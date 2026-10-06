"""Render-resolution accumulator: every learned operation runs BELOW render resolution.

Why this exists
---------------
Measured on an RTX 2070 SUPER at 960x540 -> 1920x1080, `MambaAccumulator` (v64) takes
163 ms per frame in fp16. The split is the lesson: the ported Lanczos resolve is 45 ms,
the full-resolution encoder 18 ms, the full-resolution kernel head 23 ms, the nine
full-resolution tap samples 8 ms, and ~28 ms is nothing but elementwise work on 1080p
tensors. Swapping the trunk for a U-Net removes one of those lines. A 2 ms budget needs
all of them gone.

What a millisecond buys on that card (fp16, measured):

    3x3 conv 32->32 at 540p     0.61 ms        one elementwise op on a 1080p image   0.10 ms
    the same conv at 270p       0.16 ms        bilinear grid_sample, 3ch at 1080p    3.25 ms
    the same conv at 135p       0.04 ms        softmax kernel over 11 candidates    ~13 ms

So the rules this module follows are: no convolution at render resolution or above; as few
passes over output-sized data as possible; and no per-pixel kernel over many candidates
(the kernel-prediction head is the right idea and the wrong cost here).

An output frame at integer scale ``s`` is exactly ``s*s`` render-resolution images
interleaved (`pixel_unshuffle`), so the accumulation is written at render resolution with
the sub-pixel phase as a channel axis:

- warped history colour is unshuffled into ``3*s*s`` render-resolution channels and clamped
  into the low-res neighbourhood colour range (FSR's rectification, as an AABB);
- the network sees that history next to the raw low-res frame -- history is an *input*, so
  it can tell stale history from valid history, which `MambaAccumulator` never could;
- per sub-pixel phase it predicts a colour residual on the low-res sample (the learned
  upsample) and a blend weight ``alpha``; the output is ``lerp(history, current, alpha)``.

The only output-resolution work left is the history warp and the final shuffle. In an
engine the warp is a hardware bilinear fetch; in PyTorch it is `grid_sample`, which costs
more than the whole network, so benchmarks report it separately.

Stability: history enters with weight ``1 - alpha <= 1`` and is clamped to the current
frame's colour range before both the blend and the network input, so nothing unbounded
can circulate through the recurrence.

The optional learned state is the same selective update as `mamba.py`
(``h = a*h_prev + (1-a)*x``, ``a = exp(-softplus(delta))``), kept at trunk resolution.
``n_state=0`` removes it, which is the ablation for whether it earns its cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import _lanczos2
from .resolve import phase_kernels, window_bounds

__all__ = ["FastState", "FastAccumulator"]


@dataclass
class FastState:
    color: torch.Tensor   # (1, 3, H, W) history colour, output resolution
    feat: torch.Tensor    # (1, n_state, h/2, w/2) learned state, trunk resolution
    depth: torch.Tensor   # (1, 1, h, w) previous depth, render resolution
    frame_index: int = 0
    conf: torch.Tensor | None = None   # (1, 1, H, W) accumulated sample weight (accum only)
    m1: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    osc: torch.Tensor | None = None
    luma: torch.Tensor | None = None  # previous instantaneous sample, render resolution

    age: torch.Tensor | None = None  # accumulated frames since hard reset, render resolution

    def detach(self) -> "FastState":
        return FastState(self.color.detach(), self.feat.detach(), self.depth.detach(),
                         self.frame_index, None if self.conf is None else self.conf.detach(),
                         *(None if t is None else t.detach()
                           for t in (self.m1, self.m2, self.osc, self.luma, self.age)))


class _Res(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(x)))


def _uv_grid(h: int, w: int, device) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)  # (1, h, w, 2), xy order


def _nearest_depth(depth, motion=None):
    """Reversed-Z: largest depth is nearest; max selects the first row-major tie.

    Operate on the real render extent before trunk padding. Motion is NHW2.
    """
    h, w = depth.shape[-2:]
    padded = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    taps = torch.stack([padded[..., dy:dy + h, dx:dx + w]
                        for dy in range(3) for dx in range(3)], dim=-1)
    near, index = taps.max(dim=-1)
    if motion is None:
        return near
    mp = F.pad(motion.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
    mt = torch.stack([mp[..., dy:dy + h, dx:dx + w]
                     for dy in range(3) for dx in range(3)], dim=-1)
    selected = mt.gather(-1, index.expand(-1, 2, -1, -1).unsqueeze(-1)).squeeze(-1)
    return near, selected.permute(0, 2, 3, 1)


def _thin_feature(lr):
    """Soft bright/dark ridge across either cardinal or diagonal axis.

    Both neighbours must differ on the same side by >25% of the 3x3 luma
    range. Float32 arithmetic followed by one half round matches deployment.
    """
    rgb = lr.float()
    luma = .25 * rgb[:, 0:1] + .5 * rgb[:, 1:2] + .25 * rgb[:, 2:3]
    h, w = luma.shape[-2:]
    padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    taps = [padded[..., dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]
    local = torch.stack(taps)
    span = (local.amax(0) - local.amin(0)).clamp(min=1e-3)
    ridge = torch.zeros_like(luma)
    for a, b in ((3, 5), (1, 7), (0, 8), (2, 6)):
        bright = torch.minimum(luma - taps[a], luma - taps[b])
        dark = torch.minimum(taps[a] - luma, taps[b] - luma)
        ridge = torch.maximum(ridge, torch.maximum(bright, dark))
    return ((ridge / span - .25) / .75).clamp(0, 1).to(lr.dtype)


def _coverage_evidence(lr, rng, previous, reset):
    """Pre-update moments distinguish sustained oscillation from a new change.

    Moments and instantaneous luma stay float32; only trunk features are rounded.
    A hard reset exposes no history evidence and seeds moments from this sample.
    """
    rgb = lr.float()
    luma = .25 * rgb[:, 0:1] + .5 * rgb[:, 1:2] + .25 * rgb[:, 2:3]
    m1, m2, osc, last = previous
    span = .25 * rng[:, 0:1].float() + .5 * rng[:, 1:2].float() + .25 * rng[:, 2:3].float() + .02
    valid = reset <= .5
    features = torch.cat(((m2 - m1.square()).clamp(min=0) / span, osc / span), 1)
    features = torch.where(valid, features.clamp(0, 4), 0).to(lr.dtype)
    next_state = (torch.where(valid, m1 + .25 * (luma - m1), luma),
                  torch.where(valid, m2 + .25 * (luma.square() - m2), luma.square()),
                  torch.where(valid, osc + .25 * ((luma - last).abs() - osc), 0), luma)
    return features, next_state


def _coverage_coefficient(logit, coverage_bias=-6.0):
    # Match the logit dtype before sigmoid so neutral remains exact under autocast.
    bias = torch.as_tensor(coverage_bias, device=logit.device, dtype=logit.dtype)
    floor = torch.sigmoid(bias.expand_as(logit))
    return ((torch.sigmoid(logit) - floor) / (1 - floor)).clamp(0, 1)


def _osc_reset(model, reset, mismatch, prev_d, d_max, evidence, *, training):
    """Training uses sigmoid((osc_n-T)*8); evaluation/deployment uses osc_n>T.

    Evidence is pre-update, masked only by offscreen/first-frame resets. Larger
    reversed-Z depth is nearer: a >25% foreground departure always resets.
    """
    threshold = model.soft_osc_threshold.clamp(.05, 4)
    osc_n = evidence[:, 1:2].float()
    dither = (torch.sigmoid((osc_n - threshold) * 8) if training
              else (osc_n > threshold).float())
    forced = (reset > .5) | (prev_d > d_max * 1.25)
    return torch.where(forced, 1., mismatch.float() * (1 - dither)).to(reset.dtype)


class FastAccumulator(nn.Module):
    """Same call signature as `MambaAccumulator.forward`; state from `init_state`.

    ``widths``/``depths`` describe the pyramid from the trunk resolution (half of render)
    downwards: ``widths[0]`` channels at 1/2 render, ``widths[1]`` at 1/4, and so on, with
    ``depths[i]`` residual blocks at each level.

    ``carry_raw`` accumulates samples with learned rectification, adding the free RGB
    residual only to the display. It requires ``accum`` and exposes ``_last_carry`` for loss.
    ``base_gate`` blends FSR's deringed Lanczos resolve with the current low-res sample.
    ``mv_dilate`` borrows motion from the nearest reversed-Z depth in a 3x3 window;
    ``depth_soft`` keeps depth mismatch as a trunk feature and resets only off-screen history.
    ``depth_dilate`` stores that nearest depth. ``thin_lock`` appends a soft luma ridge
    feature and widens its clamp slack by ``1 + thin_slack`` above 0.5.
    ``history_age`` adds a reprojected frame count and caps current-frame weight
    outside hard resets; alpha_min is learned and bounded to [0, 1].
    ``coverage`` adds reprojected pre-update luma variance/oscillation and a calibrated
    sigmoid head that lowers current confidence and favours the smooth base.
    ``depth_soft_osc`` requires depth_test/depth_soft/coverage and restricts soft depth
    handling to oscillating coverage, except for reversed-Z foreground departures.
    Its gate is sigmoid in train() and a strict hard step in eval() and FusedFast.
    """

    def __init__(self, render_size: tuple[int, int], output_size: tuple[int, int],
                 widths: tuple[int, ...] = (32, 64), depths: tuple[int, ...] = (1, 2),
                 n_state: int = 8, film: bool = True, depth_test: bool = False,
                 stem_kernel: int = 1, learned_clamp: bool = False,
                 resolve: str = "nearest", detail_ch: int = 0, hist_filter: str = "bilinear", accum: bool = False,
                 conf_max: float = 32.0, conf_motion: bool = False,
                 hist_residual: bool = False, nearest_sample: bool = False,
                 conf_consistent: bool = False, jitter_sign: float = 1.0, device: torch.device | str = "cpu",
                 carry_raw: bool = False, base_gate: bool = False,
                 reset_lanczos: bool = False, mv_dilate: bool = False,
                 depth_dilate: bool = False, thin_lock: bool = False,
                 depth_soft: bool = False, coverage: bool = False,
                 coverage_bias: float = -3.0, depth_soft_osc: bool = False, history_age: bool = False) -> None:
        super().__init__()
        if carry_raw and (not accum or learned_clamp or hist_residual):
            raise ValueError("carry_raw requires accum=True, learned_clamp=False and hist_residual=False")
        self.mv_dilate, self.depth_dilate, self.thin_lock = mv_dilate, depth_dilate, thin_lock
        if depth_soft and not depth_test:
            raise ValueError("depth_soft requires depth_test=True")
        self.depth_soft = depth_soft
        self.history_age = history_age
        if history_age:
            self.alpha_min = nn.Parameter(torch.tensor(.05))
        self.coverage = coverage
        if depth_soft_osc and not (depth_test and depth_soft and coverage):
            raise ValueError("depth_soft_osc requires depth_test, depth_soft and coverage")
        self.depth_soft_osc = depth_soft_osc
        if depth_soft_osc:
            self.soft_osc_threshold = nn.Parameter(torch.tensor(.5))
        if not math.isfinite(coverage_bias) or coverage_bias >= 0:
            raise ValueError("coverage bias must be negative and finite")
        if coverage:
            self.register_buffer("_coverage_marker", torch.zeros(1))
            self.register_buffer("coverage_bias", torch.tensor(float(coverage_bias)))
        if mv_dilate:
            self.register_buffer("_mv_dilate_marker", torch.zeros(1))
        if depth_dilate:
            self.register_buffer("_depth_dilate_marker", torch.zeros(1))
        if thin_lock:
            self.thin_slack = nn.Parameter(torch.tensor(1.0))
        self.base_gate = base_gate
        # reset_lanczos (with base_gate): where there is no history the output is a
        # single-frame reconstruction, and the Lanczos resolve is a far better one than the
        # nearest sample the gate starts on. Reset pixels take the Lanczos candidate.
        self.reset_lanczos = reset_lanczos
        if base_gate:
            self.register_buffer("_base_gate_marker", torch.zeros(1))
        self.carry_raw = carry_raw
        if carry_raw:
            self.register_buffer("_carry_marker", torch.zeros(1))
        h, w = render_size
        s = output_size[0] // h
        if s * h != output_size[0] or s * w != output_size[1] or s < 1:
            raise ValueError(f"need an integer uniform scale, got {render_size} -> {output_size}")
        # The trunk needs dimensions divisible by 2**levels; odd sizes (the 640x355
        # captures) are edge-padded on the way in and cropped on the way out.
        m = 2 ** len(widths)
        self._pad = ((-h) % m, (-w) % m)
        hp, wp = h + self._pad[0], w + self._pad[1]
        assert len(widths) == len(depths)
        self.render_size, self.output_size, self.scale = render_size, output_size, s
        # jitter_sign: the captures place a low-res sample at texel centre + jitter, the
        # opposite of the convention baseline._upsample was ported with (centre - jitter).
        # Measured: sampling the ground truth at centre + jitter matches the low-res frames
        # (toyshop MSE 0.66e-4), centre - jitter does not (2.09e-4, worse than no jitter).
        # -1 feeds every jitter-aware computation the corrected sign; +1 keeps the behaviour
        # of checkpoints trained before this was found.
        self.jitter_sign = float(jitter_sign)
        if self.jitter_sign != 1.0:
            self.register_buffer("_jitter_sign", torch.tensor(self.jitter_sign))
        self.n_state, self.depth_test = n_state, depth_test
        self.learned_clamp = learned_clamp
        assert resolve in ("nearest", "lanczos")
        self.resolve = resolve
        self.state_channels = n_state
        self.dev = torch.device(device)
        p = s * s
        self.p = p

        # lr 3 | clamped history 3p | colour range 3 | inverse depth | speed | reset
        in_ch = 3 + 3 * p + 3 + 1 + 1 + 1 + int(thin_lock) + 2 * int(coverage) + int(history_age)
        # learned_clamp: the hard AABB clamp throws away accumulated sub-pixel detail
        # whenever it falls outside the low-res neighbourhood range, which is exactly
        # where supersampling adds something. FSR picks the box width by hand; here the
        # network sees how far the history sat outside the box (3 channels) and predicts,
        # per sub-pixel, how much of the clamp to apply. History weight stays <= 1.
        if learned_clamp:
            in_ch += 3
        if carry_raw:
            in_ch += p
        # accum: FSR converges on static content because it carries an accumulated sample
        # weight and blends with current / (history + current) -- a running average whose
        # memory grows with every valid frame. A network that only predicts alpha has no
        # count to divide by and settles on a short exponential average (measured: mean
        # alpha 0.2-0.4 where FSR's is ~0.045), which is why FSR wins the low-motion
        # scenes. Here the weight is explicit state, reprojected with the colour; the
        # network sees it and predicts a correction and how much of it survives.
        self.accum, self.conf_max = accum, conf_max
        self.conf_motion = False
        if accum:
            in_ch += p
            ph = (torch.arange(s, dtype=torch.float32) + 0.5) / s
            py_, px_ = torch.meshgrid(ph, ph, indexing="ij")
            self.register_buffer("_acc_px", px_.reshape(p), persistent=False)
            self.register_buffer("_acc_py", py_.reshape(p), persistent=False)
            self.acc_sharp = nn.Parameter(torch.tensor(4.0))
            # conf_motion: every reprojection at a fractional offset low-passes the history,
            # but the carried weight does not know that. After a static stretch the weight
            # is at its cap, and when motion resumes the blurred history keeps winning for
            # many frames (measured on brutalism: 7 dB below FSR when the camera starts to
            # move again, recovering over 8 frames). FSR scales accumulation by velocity;
            # here the weight is divided by (1 + m * speed) with a learned m.
            self.conf_motion = conf_motion
            if conf_motion:
                self.conf_m = nn.Parameter(torch.tensor(0.5))
        # Lossless 2x2 space-to-depth, then a conv at half render resolution. A stride-2 3x3
        # on the 540p tensor costs 0.58 ms; this is ~0.1.
        self.stem = nn.Conv2d(in_ch * 4 + n_state, widths[0], stem_kernel,
                              padding=stem_kernel // 2)
        if coverage or history_age:
            with torch.no_grad():
                extra = 2 * int(coverage) + int(history_age)
                self.stem.weight[:, (in_ch - extra) * 4:in_ch * 4].zero_()
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i, (c, d) in enumerate(zip(widths, depths)):
            self.enc.append(nn.Sequential(*[_Res(c) for _ in range(d)]))
            if i + 1 < len(widths):
                self.down.append(nn.Conv2d(c, widths[i + 1], 3, stride=2, padding=1))
                # 1x1 then PixelShuffle: no transposed conv (checkerboard), no big 3x3 on
                # the way up.
                self.up.append(nn.Conv2d(widths[i + 1], c * 4, 1))
        self.fuse = nn.Conv2d(widths[0], widths[0], 3, padding=1)

        self.film = None
        if film:
            # Jitter says where the low-res sample sits inside the output pixel, which is
            # what decides the right sub-pixel reconstruction. One tiny MLP per frame.
            self.film = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2 * widths[0]))
            nn.init.zeros_(self.film[2].weight)
            nn.init.zeros_(self.film[2].bias)

        # Per render pixel: residual 3p + alpha p. Emitted at half render res, 4 phases.
        # hist_residual: the output is (1-alpha)*history + alpha*(low-res + residual), so the
        # network's only handle on errors already in the history is to raise alpha, i.e. to
        # trade them for single-frame noise (measured: alpha settles at 0.15-0.35 even on
        # static content, and a plain 3x3 blur of the output gains 0.37 dB). This adds a
        # per-phase correction to the history itself, bounded by the local colour range
        # (so it cannot run away through the recurrence), letting the model undo
        # reprojection blur and keep accumulating.
        self.hist_residual = hist_residual
        # nearest_sample: the accumulation weight of a phase measures the distance to the
        # nearest low-res sample, which for half the phase/jitter combinations belongs to a
        # NEIGHBOURING texel -- while the colour base was always the centre texel, leaving
        # the residual to transport the neighbour's colour. This gathers the sample the
        # weight is actually about.
        # conf_consistent: the learned correction changed alpha but not the stored weight;
        # here the corrected current weight is what is both blended and accumulated.
        self.nearest_sample, self.conf_consistent = nearest_sample, conf_consistent
        ph_ = (torch.arange(s, dtype=torch.float32) + 0.5) / s
        self._ph_list = [(float(ph_[i // s]), float(ph_[i % s])) for i in range(p)]  # (py, px)
        # Head order: residual | alpha | clamp | history residual | base gate | carry beta | cov | keep.
        self._base_gate_offset = (5 if learned_clamp else 4) * p + (3 * p if hist_residual else 0)
        self._carry_beta_offset = self._base_gate_offset + (p if base_gate else 0)
        self._cov_offset = self._carry_beta_offset + (p if carry_raw else 0)
        self.n_px = ((5 if learned_clamp else 4) * p + (3 * p if hist_residual else 0)
                     + (p if base_gate else 0) + (p if accum else 0) + (p if carry_raw else 0) + (p if coverage else 0))
        if hist_residual:
            self.hres_gain = nn.Parameter(torch.tensor(0.25))
        # detail_ch > 0: the half-res trunk alone cannot place sub-pixel detail (16 output
        # pixels per trunk cell from one feature vector). A thin branch at render resolution
        # -- trunk features shuffled up, next to the raw inputs -- refines the per-phase
        # outputs. A 3x3 conv costs ~C^2, so 16 channels at 540p is ~0.15 ms where 32 is 0.6.
        self.detail_ch = detail_ch
        # Bilinear reprojection low-passes the history every frame it is carried, which is
        # why FSR reprojects with a Lanczos kernel. bicubic is the sharp option here; its
        # overshoot is removed by the history clamp.
        assert hist_filter in ("bilinear", "bicubic")
        self.hist_filter = hist_filter
        n_head = detail_ch if detail_ch else self.n_px
        self.out = nn.Conv2d(widths[0], n_head * 4 + 2 * n_state, 1)
        last = self.out
        if detail_ch:
            self.detail = nn.Conv2d(detail_ch + in_ch, detail_ch, 3, padding=1)
            self.detail_out = nn.Conv2d(detail_ch, self.n_px, 3, padding=1)
            last = self.detail_out
        rep_ = 1 if detail_ch else 4      # head channels per logical output channel
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            b = torch.zeros(last.bias.shape[0])
            # alpha ~0.10: a working accumulator at init. With accum the same channels are
            # a correction to the running-average blend and start at zero.
            b[3 * p * rep_: 4 * p * rep_] = 0.0 if accum else -2.2
            if accum:
                b[(self.n_px - p) * rep_: self.n_px * rep_] = 3.0   # keep ~95% of the history weight
            if base_gate:
                q = self._base_gate_offset * rep_
                b[q: q + p * rep_] = 2.0
            if learned_clamp:
                b[4 * p * rep_: 5 * p * rep_] = 3.0  # beta ~0.95: starts as the hard clamp
            if carry_raw:
                q = self._carry_beta_offset * rep_
                b[q: q + p * rep_] = 3.0
            if coverage:
                q = self._cov_offset * rep_
                b[q: q + p * rep_] = coverage_bias
            last.bias.copy_(b)
        # History is clamped to the low-res neighbourhood range widened by this fraction.
        self.box_slack = nn.Parameter(torch.tensor(0.25))

        if resolve == "lanczos" or base_gate:
            # Geometry of FSR's resolve (upsample.h:305-307), per sub-pixel phase: where the
            # output point sits inside its low-res texel, and the 3x3 tap offsets. The
            # per-frame jitter turns these into a (p, 1, 4, 4) kernel -- one tiny conv.
            ph = (torch.arange(s, dtype=torch.float32) + 0.5) / s
            py, pxx = torch.meshgrid(ph, ph, indexing="ij")
            # 4x4 support (-2..1): FSR's 3x3 window sits at -2 or -1 depending on which
            # side of the output point the base sample falls (upsample.h:307).
            tap = torch.arange(-2.0, 2.0)
            ty, tx = torch.meshgrid(tap, tap, indexing="ij")
            self.register_buffer("_ph_x", pxx.reshape(p, 1, 1), persistent=False)
            self.register_buffer("_ph_y", py.reshape(p, 1, 1), persistent=False)
            self.register_buffer("_tap_x", tx.reshape(1, 4, 4), persistent=False)
            self.register_buffer("_tap_y", ty.reshape(1, 4, 4), persistent=False)
            # How strongly a phase that has a sample close to it this frame is pushed
            # toward the current frame (FSR's per-frame Lanczos weight, learned scale).
            if resolve == "lanczos":
                self.jit_gain = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("_uv_hr", _uv_grid(*output_size, self.dev), persistent=False)
        self.register_buffer("_uv_lr", _uv_grid(h, w, self.dev), persistent=False)
        # Trunk-resolution grid in TRUE-frame UV (may exceed 1 inside the padding), plus the
        # factor that maps true-frame UV to the padded tensor's normalised coordinates.
        self._tr_size = (hp // 2, wp // 2)
        self.register_buffer("_uv_tr", _uv_grid(hp // 2, wp // 2, self.dev)
                             * torch.tensor([wp / w, hp / h], device=self.dev), persistent=False)
        self.register_buffer("_tr_fac", torch.tensor([w / wp, h / hp]), persistent=False)
        self.register_buffer("_wh", torch.tensor([float(w), float(h)]), persistent=False)
        self.to(self.dev)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.coverage and prefix + "coverage_bias" not in state_dict:
            bias = -6.0 if prefix + "_coverage_marker" in state_dict else float(self.coverage_bias)
            state_dict[prefix + "coverage_bias"] = self.coverage_bias.new_tensor(bias)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def init_state(self, device=None) -> FastState:
        device = device or self.dev
        h, w = self.render_size
        return FastState(torch.zeros(1, 3, *self.output_size, device=device),
                         torch.zeros(1, self.n_state, *self._tr_size, device=device),
                         torch.ones(1, 1, h, w, device=device), 0,
                         torch.zeros(1, 1, *self.output_size, device=device) if self.accum else None,
                         *(torch.zeros(1, 1, h, w, device=device) if self.coverage else None
                           for _ in range(4)),
                         torch.zeros(1, 1, h, w, device=device) if self.history_age else None)

    def reproject(self, state: FastState, mv_hr: torch.Tensor) -> torch.Tensor:
        """History colour moved along the motion vectors. The one output-resolution
        resample; a texture fetch in an engine."""
        # Coordinates stay float32: in half precision a 1920-wide grid is off by up to 0.2 px.
        src = state.color.float()
        if self.accum:
            src = torch.cat((src, state.conf.float()), dim=1)   # weight rides with the colour
        return F.grid_sample(src, (self._uv_hr + mv_hr.float()) * 2 - 1,
                             mode=self.hist_filter, padding_mode="border", align_corners=False)

    def forward(self, state: FastState, lr_rgb: torch.Tensor, mv: torch.Tensor,
                depth: torch.Tensor, jitter, prev_depth_hr=None, hist=None):
        """lr_rgb (h,w,3); mv (.,.,2) and depth (.,.) at render OR output resolution.

        ``hist`` lets a caller that already reprojected the history (an engine, or a
        benchmark timing the network alone) pass it in.
        """
        h, w = self.render_size
        s, p = self.scale, self.p
        if self.jitter_sign != 1.0:
            jitter = (jitter * self.jitter_sign if torch.is_tensor(jitter)
                      else (float(jitter[0]) * self.jitter_sign, float(jitter[1]) * self.jitter_sign))
        # One working dtype for the whole step. Under autocast only the convolutions would
        # run in half precision and every elementwise pass over output-sized data would
        # stay in float32 at twice the cost (measured: 0.19 vs 0.10 ms per pass).
        dt = torch.float16 if torch.is_autocast_enabled() else lr_rgb.dtype
        lr = lr_rgb.permute(2, 0, 1).unsqueeze(0).to(dt)
        mv = mv.unsqueeze(0)
        if mv.shape[1] == h:
            mv_lr = mv
            mv_hr = None
        else:
            mv_hr = mv
            mv_lr = F.avg_pool2d(mv.permute(0, 3, 1, 2), s).permute(0, 2, 3, 1)
        d = depth[None, None].to(dt)
        if d.shape[2] != h:
            d = d[:, :, ::s, ::s]

        stored_depth = d
        if self.mv_dilate:
            near, mv_lr = _nearest_depth(d, mv_lr)
            mv_hr = None
            if self.depth_dilate:
                stored_depth = near
        elif self.depth_dilate:
            stored_depth = _nearest_depth(d)

        if hist is None:
            if mv_hr is None:
                mv_hr = F.interpolate(mv_lr.permute(0, 3, 1, 2), scale_factor=s, mode="bilinear",
                                      align_corners=False).permute(0, 2, 3, 1)
            hist = self.reproject(state, mv_hr)
        hist = hist.to(dt)
        conf_w = None
        if self.accum:
            hist, conf_w = hist[:, :3], hist[:, 3:].clamp(min=0.0)

        uv = self._uv_lr + mv_lr
        reset = (~((uv >= 0) & (uv < 1)).all(dim=-1))[:, None]
        reset_feature = None
        if self.depth_test:
            # Neighbourhood depth envelope, as `mamba.py`'s robust_disocc.
            prev_d = F.grid_sample(state.depth, (uv * 2 - 1).to(state.depth.dtype), mode="nearest",
                                   padding_mode="border", align_corners=False)
            d_max = F.max_pool2d(d, 3, stride=1, padding=1)
            d_min = -F.max_pool2d(-d, 3, stride=1, padding=1)
            mismatch = (prev_d < d_min * 0.9) | (prev_d > d_max * 1.1)
            if self.depth_soft:
                # Alternating coverage still informs the trunk without discarding history.
                reset_feature = (reset | mismatch).to(dt)
            else:
                reset = reset | mismatch
        reset = reset.to(dt)
        if state.frame_index == 0:
            reset = torch.ones_like(reset)
            reset_feature = reset

        # --- rectify: clamp history into the (widened) low-res neighbourhood range ---
        mx = F.max_pool2d(lr, 3, stride=1, padding=1)
        mn = -F.max_pool2d(-lr, 3, stride=1, padding=1)
        rng = mx - mn
        previous = None
        if self.depth_soft_osc:
            previous = torch.cat((state.m1, state.m2, state.osc, state.luma), 1)
            previous = F.grid_sample(previous, uv.float() * 2 - 1, mode="nearest",
                                     padding_mode="border", align_corners=False).chunk(4, 1)
            evidence, _ = _coverage_evidence(lr, rng, previous, reset)
            reset = _osc_reset(self, reset, mismatch, prev_d, d_max, evidence,
                               training=self.training)
        slack = rng * self.box_slack.abs()
        thin = None
        if self.thin_lock:
            thin = _thin_feature(lr)
            factor = (1 + self.thin_slack).to(dt)
            slack = slack * torch.where(thin > .5, factor, torch.ones_like(thin))
        hist_un = F.pixel_unshuffle(hist, s).reshape(1, 3, p, h, w)
        h_cl = torch.minimum(torch.maximum(hist_un, (mn - slack).unsqueeze(2)),
                             (mx + slack).unsqueeze(2))

        speed = (torch.linalg.vector_norm(mv_lr * self._wh, dim=-1) * 0.1).clamp(0, 1)[:, None].to(dt)
        parts = [lr, h_cl.reshape(1, 3 * p, h, w), rng, 1.0 / d.clamp(min=1e-2), speed,
                 reset_feature if reset_feature is not None else reset]
        if self.learned_clamp:
            parts.append((hist_un - h_cl).abs().mean(dim=2) * (1 - reset))
        if self.accum:
            conf_w = F.pixel_unshuffle(conf_w, s) * (1 - reset)        # (1, p, h, w)
            if self.conf_motion:
                spd = torch.linalg.vector_norm(mv_lr * self._wh, dim=-1).clamp(max=16.0)[:, None]
                conf_w = conf_w / (1.0 + self.conf_m.abs() * spd.to(dt))
            parts.append(torch.log1p(conf_w))
        if self.carry_raw:
            if self.nearest_sample:
                jx, jy = float(jitter[0]), float(jitter[1])
                lrp = F.pad(lr, (1, 1, 1, 1), mode="replicate")
                samples = []
                for py_, px_ in self._ph_list:
                    ny, nx = round(py_ - 0.5 + jy), round(px_ - 0.5 + jx)
                    samples.append(lrp[:, :, 1 + ny: 1 + ny + h, 1 + nx: 1 + nx + w])
                base_q = torch.stack(samples, dim=2)
            else:
                base_q = lr.unsqueeze(2)
            raw_luma = 0.25 * hist_un[:, 0] + 0.5 * hist_un[:, 1] + 0.25 * hist_un[:, 2]
            base_luma = 0.25 * base_q[:, 0] + 0.5 * base_q[:, 1] + 0.25 * base_q[:, 2]
            luma_range = 0.25 * rng[:, 0:1] + 0.5 * rng[:, 1:2] + 0.25 * rng[:, 2:3]
            parts.append(((raw_luma - base_luma) / (luma_range + 0.02)).clamp(-4, 4) * (1 - reset))
        if self.thin_lock:
            parts.append(thin)
        coverage_state = (None,) * 4
        if self.coverage:
            if previous is None:
                previous = torch.cat((state.m1, state.m2, state.osc, state.luma), 1)
                previous = F.grid_sample(previous, uv.float() * 2 - 1, mode="nearest",
                                         padding_mode="border", align_corners=False).chunk(4, 1)
            evidence, coverage_state = _coverage_evidence(lr, rng, previous, reset)
            self._last_coverage = evidence
            parts.append(evidence)
        age = new_age = None
        if self.history_age:
            age = F.grid_sample(state.age, uv.float() * 2 - 1, mode="nearest",
                                padding_mode="border", align_corners=False)
            age = torch.where(reset > .5, torch.zeros_like(age), age)
            parts.append((torch.log2(1 + age) / 5).to(dt))
            new_age = torch.where(reset > .5, torch.zeros_like(age), (age + 1).clamp(max=32))
        x = torch.cat(parts, dim=1)
        x_r = x
        ph, pw = self._pad
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x = F.pixel_unshuffle(x, 2)
        if self.n_state:
            mv_p, rs_p = mv_lr.permute(0, 3, 1, 2), reset
            if ph or pw:
                mv_p = F.pad(mv_p, (0, pw, 0, ph), mode="replicate")
                rs_p = F.pad(rs_p, (0, pw, 0, ph), mode="replicate")
            mv_tr = F.avg_pool2d(mv_p, 2).permute(0, 2, 3, 1)
            keep = 1 - F.max_pool2d(rs_p, 2)
            g_tr = (self._uv_tr + mv_tr) * self._tr_fac * 2 - 1
            feat_prev = F.grid_sample(state.feat, g_tr.to(state.feat.dtype),
                                      mode="bilinear", padding_mode="border",
                                      align_corners=False).to(dt) * keep
            x = torch.cat((x, feat_prev), dim=1)
        x = F.relu(self.stem(x))

        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                skips.append(x)
                x = F.relu(self.down[i](x))
        for i in range(len(self.down) - 1, -1, -1):
            x = skips[i] + F.pixel_shuffle(self.up[i](x), 2)
        x = F.relu(self.fuse(x))
        if self.film is not None:
            j = jitter if torch.is_tensor(jitter) else torch.tensor(jitter, device=lr.device)
            g, b = self.film(j.to(x.dtype).view(1, 2)).chunk(2, dim=1)
            x = x * (1 + g.view(1, -1, 1, 1)) + b.view(1, -1, 1, 1)
        o = self.out(x)

        # --- blend, per sub-pixel phase ---
        n_head = self.detail_ch if self.detail_ch else self.n_px
        px = F.pixel_shuffle(o[:, : n_head * 4], 2)[..., :h, :w]     # (1, n_head, h, w)
        if self.detail_ch:
            px = self.detail_out(F.relu(self.detail(torch.cat((px, x_r.to(px.dtype)), dim=1))))
        px = px.to(dt)
        cov = (_coverage_coefficient(px[:, self._cov_offset:self._cov_offset + p], self.coverage_bias)
               if self.coverage else None)
        a_logit = px[:, 3 * p: 4 * p]
        if self.base_gate:
            j, k0, _, _, prior, win = phase_kernels(
                jitter, lr.device, self._ph_x, self._ph_y, self._tap_x, self._tap_y)
            taps = F.pad(lr.reshape(3, 1, h, w), (2, 1, 2, 1), mode="replicate")
            lanczos = F.conv2d(taps, k0.to(dt)).unsqueeze(0)
            lo, hi = window_bounds(taps, win, h, w)
            lanczos = torch.minimum(torch.maximum(lanczos, lo), hi)
            if self.carry_raw:
                nearest = base_q
            elif self.nearest_sample:
                jx, jy = float(j[0]), float(j[1])
                lrp = F.pad(lr, (1, 1, 1, 1), mode="replicate")
                samples = []
                for py_, px_ in self._ph_list:
                    ny, nx = round(py_ - 0.5 + jy), round(px_ - 0.5 + jx)
                    samples.append(lrp[:, :, 1 + ny: 1 + ny + h, 1 + nx: 1 + nx + w])
                nearest = torch.stack(samples, dim=2)
            else:
                nearest = lr.unsqueeze(2)
            q = self._base_gate_offset
            gate = torch.sigmoid(px[:, q: q + p]).unsqueeze(1)
            if self.coverage:
                gate = gate * (1 - cov).unsqueeze(1)
            if self.reset_lanczos:
                gate = gate * (1 - reset).unsqueeze(2)
            base = torch.lerp(lanczos, nearest, gate)
            cur = base + px[:, :3 * p].reshape(1, 3, p, h, w)
            if self.carry_raw:
                base_q = base
            if self.resolve == "lanczos":
                a_logit = a_logit + (self.jit_gain * prior).to(dt)
        elif self.resolve == "lanczos":
            # Jitter-aware base: each phase is a Lanczos-2 resolve of the 3x3 low-res taps
            # at their true (un-jittered) sample positions, deringed into the tap range.
            j = jitter if torch.is_tensor(jitter) else torch.tensor(jitter, device=lr.device)
            j = j.float()
            ox = self._tap_x + 0.5 - j[0] - self._ph_x
            oy = self._tap_y + 0.5 - j[1] - self._ph_y
            # Window start per phase: -2 if the base sample lies beyond the output point.
            sx = torch.where(0.5 - j[0] - self._ph_x > 0, -2.0, -1.0)
            sy = torch.where(0.5 - j[1] - self._ph_y > 0, -2.0, -1.0)
            inwin = ((self._tap_x >= sx) & (self._tap_x <= sx + 2)
                     & (self._tap_y >= sy) & (self._tap_y <= sy + 2))
            k = _lanczos2(torch.sqrt(ox * ox + oy * oy)) * inwin    # (p, 4, 4)
            ksum = k.sum(dim=(1, 2), keepdim=True)
            kern = (k / ksum.clamp(min=1e-4)).unsqueeze(1).to(dt)
            base = F.conv2d(F.pad(lr.reshape(3, 1, h, w), (2, 1, 2, 1), mode="replicate"), kern)
            base = torch.minimum(torch.maximum(base.unsqueeze(0), mn.unsqueeze(2)), mx.unsqueeze(2))
            cur = base + px[:, : 3 * p].reshape(1, 3, p, h, w)
            prior = torch.log(ksum.clamp(min=1e-4) / ksum.mean()).reshape(1, p, 1, 1)
            a_logit = a_logit + (self.jit_gain * prior).to(dt)
        elif self.nearest_sample:
            jx, jy = float(jitter[0]), float(jitter[1])
            lrp = F.pad(lr, (1, 1, 1, 1), mode="replicate")
            base = []
            for py_, px_ in self._ph_list:
                ny, nx = round(py_ - 0.5 + jy), round(px_ - 0.5 + jx)
                base.append(lrp[:, :, 1 + ny: 1 + ny + h, 1 + nx: 1 + nx + w])
            cur = torch.stack(base, dim=2) + px[:, : 3 * p].reshape(1, 3, p, h, w)
        else:
            cur = lr.unsqueeze(2) + px[:, : 3 * p].reshape(1, 3, p, h, w)
        new_conf = None
        if self.accum:
            # Weight of this frame's sample for each phase: how close the nearest low-res
            # sample centre (texel + 0.5 - jitter, any neighbouring texel) is to the phase.
            j = jitter if torch.is_tensor(jitter) else torch.tensor(jitter, device=lr.device)
            j = j.float()
            dx = (0.5 - j[0] - self._acc_px + 0.5) % 1.0 - 0.5
            dy = (0.5 - j[1] - self._acc_py + 0.5) % 1.0 - 0.5
            w_c = torch.exp(-self.acc_sharp.abs() * (dx * dx + dy * dy)).reshape(1, p, 1, 1).to(dt)
            if self.coverage:
                w_c = w_c * (1 - .9 * cov)
            keep = torch.sigmoid(px[:, -p:])
            w_h = conf_w * keep
            if self.conf_consistent:
                w_ce = w_c * torch.exp(a_logit.float().clamp(-2.0, 2.0)).to(dt)
                a_logit = (torch.log(w_ce) - torch.log(w_h + 1e-3)).to(dt)
                new_conf = (w_h + w_ce).clamp(max=self.conf_max)
            else:
                # autocast runs log in float32; come back to the working dtype.
                a_logit = (a_logit + torch.log(w_c) - torch.log(w_h + 1e-3)).to(dt)
                new_conf = (w_h + w_c).clamp(max=self.conf_max)
        # A disoccluded pixel takes the current frame outright (an exact select: a large
        # logit offset can still be out-voted).
        current_weight = torch.sigmoid(a_logit)
        if self.history_age:
            limit = torch.maximum(1 / (1 + age), self.alpha_min.clamp(0, 1)).to(dt)
            current_weight = torch.minimum(current_weight, limit)
        alpha = torch.where(reset > 0.5, torch.ones_like(a_logit), current_weight).unsqueeze(1)
        hist_b = h_cl
        if self.hist_residual:
            k0 = (5 if self.learned_clamp else 4) * p
            hr = torch.tanh(px[:, k0: k0 + 3 * p]).reshape(1, 3, p, h, w)
            hist_b = h_cl + hr * (rng.unsqueeze(2) * self.hres_gain.abs()).to(dt)
        if self.learned_clamp:
            beta = torch.sigmoid(px[:, 4 * p: 5 * p]).unsqueeze(1)
            hist_b = torch.lerp(hist_un, hist_b, beta)
        if self.carry_raw:
            q = self._carry_beta_offset
            beta = torch.sigmoid(px[:, q: q + p]).unsqueeze(1)
            h_mix = torch.lerp(hist_un, h_cl, beta)
            out_un = torch.lerp(h_mix, base_q, alpha)
        else:
            out_un = torch.lerp(hist_b, cur, alpha)
        self._last_reset = reset
        self._last_alpha = alpha
        out = F.pixel_shuffle(out_un.reshape(1, 3 * p, h, w), s)   # (1, 3, H, W)
        carry = out
        if self.carry_raw:
            self._last_carry = carry
            display_un = out_un + px[:, :3 * p].reshape(1, 3, p, h, w)
            out = F.pixel_shuffle(display_un.reshape(1, 3 * p, h, w), s)

        feat = state.feat
        if self.n_state:
            n = self.n_state
            so = o[:, n_head * 4:].float()
            a = torch.exp(-F.softplus(so[:, :n]))
            feat = (a * feat_prev.float() + (1 - a) * torch.tanh(so[:, n:])).to(state.feat.dtype)
        if new_conf is not None:
            new_conf = F.pixel_shuffle(new_conf.float(), s)
        new = FastState(carry, feat, stored_depth, state.frame_index + 1, new_conf, *coverage_state, new_age)
        return out[0].permute(1, 2, 0).clamp(min=0.0), new
