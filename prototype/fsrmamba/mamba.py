"""A learned recurrent accumulator: selective state-space update per pixel.

This is the drop-in replacement for `baseline.FSRAccumulator`. Same call
signature, same role in the pipeline -- it just replaces the hand-tuned update
rules with learned ones and widens the recurrent state from FSR's 4 channels to
`state_channels`.

Design decisions worth defending
--------------------------------

**The spatial resolve is kept, not learned.** We reuse FSR's Lanczos resolve as
a fixed front-end and learn only the temporal update. Two reasons: it isolates
the contribution (any win is attributable to the recurrence, not to a fancier
spatial filter), and it keeps the compute budget honest, since the resolve is
already tuned to run in microseconds. Learning the spatial path too is a later
experiment, not a phase-one one.

**The recurrence is a diagonal selective SSM, one step per frame.** Mamba's
associative parallel scan exists to parallelise training over a known sequence.
At inference, frames arrive one at a time, so a single elementwise recurrent step
is all that is needed -- which is what makes this viable in a real-time budget at
all. Training pays for this: we unroll frame by frame with the warp in the loop
and backpropagate through time, forfeiting the scan speedup. That cost is real
and lands entirely on training.

**The decay gate is the learned analogue of FSR's accumulation weight.** In
`baseline.py`, `ComputeBaseAccumulationWeight` decides how much to trust memory
using hand-written rules. Here that is `A_bar = exp(-softplus(delta))` with
`delta` predicted from the input -- input-dependent forgetting, which is exactly
what the "selective" in selective SSM means. The mapping between the two is the
cleanest way to explain this project to someone who knows either half.

**The model is a strict generalisation of FSR's accumulator.** The first three
state channels hold history colour in YCoCg, exactly as FSR does; the remaining
channels are free learned state. The output is
``lerp(warped_history, resolved, alpha) + residual`` where ``alpha`` is predicted
per pixel -- structurally identical to `accumulate.h:35-36`, with a learned blend
weight in place of ``fUpsampledWeight / fHistoryWeight``.

This matters more than it looks. An earlier version kept the state fully
abstract and predicted a residual on the resolve; it started at *spatial-only*
quality and had to rediscover temporal accumulation from scratch, which it did
very slowly. Keeping FSR's structure means the model starts near baseline
behaviour and spends its capacity on improving the blend decision -- which is
the actual thesis -- rather than on relearning that history exists.

It also makes the comparison honest in the other direction: since FSR's
heuristic is inside this model's hypothesis space, "the learned version wins"
cannot be an artifact of the learned version having a different architecture.

Not yet implemented: expert routing. That is deliberately phase two, so the
recurrence can be measured on its own first.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .sepconv import conv2d as _conv
import torch.nn.functional as F

from .baseline import FSRAccumulator
from .colorspace import rgb_to_ycocg, ycocg_to_rgb

__all__ = ["MambaState", "MambaAccumulator"]


@dataclass
class MambaState:
    """The recurrent state -- FSR's float4 widened to N learned channels.

    Stored as (C, H, W) to stay in torch's conv layout; the baseline's (H, W, C)
    convention stops at the module boundary.
    """

    hidden: torch.Tensor
    frame_index: int = 0

    @classmethod
    def zeros(cls, size: tuple[int, int], channels: int, device="cpu") -> "MambaState":
        h, w = size
        return cls(hidden=torch.zeros((channels, h, w), device=device))

    def detach(self) -> "MambaState":
        """Cut the autograd graph -- used to truncate BPTT between windows."""
        return MambaState(self.hidden.detach(), self.frame_index)


class MambaAccumulator(nn.Module):
    def __init__(
        self,
        render_size: tuple[int, int],
        output_size: tuple[int, int],
        state_channels: int = 8,
        feature_channels: int = 24,
        num_experts: int = 1,
        sharpen: bool = False,
        rectify: bool = False,
        box_max: float = 3.0,
        learned_resolve: bool = False,
        swin_resolve: bool = False,
        swin_dim: int = 48,
        lr_upsampler: bool = False,
        lr_dim: int = 64,
        lr_film: bool = False,
        separable: bool = False,
        unet: int = 0,
        kernel_predict: bool = False,
        robust_disocc: bool = False,
        lock_feature: bool = False,
        alpha_scale: float = 1.0,
        encoder_depth: int = 2,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.render_size = render_size
        self.output_size = output_size
        self.state_channels = state_channels
        self.num_experts = num_experts
        self.sharpen = sharpen
        self.box_max = box_max
        self.robust_disocc = robust_disocc
        self.lock_feature = lock_feature
        self.alpha_scale = alpha_scale
        self.dev = torch.device(device)

        # Fixed, non-learned front-end: FSR's Lanczos resolve and its
        # rectification box. We borrow the implementation wholesale.
        self._resolve = FSRAccumulator(render_size, output_size, device=device)

        # Input features per pixel:
        #   3 upsampled YCoCg
        #   3 rectification box centre
        #   3 rectification box stddev  (local contrast -- how risky is blending?)
        #   1 disocclusion
        #   1 normalised velocity magnitude
        #   1 inverse depth
        #   +1 thin-feature (lock) confidence, when enabled
        in_ch = 13 if lock_feature else 12
        c = feature_channels
        n = state_channels

        # Feature trunk. Two shapes available:
        #
        # FLAT (default): two convs at OUTPUT resolution. Simple, and the reason this
        # model costs 592 GFLOP/frame -- a single 48->48 3x3 at 1080p is 86 GFLOP.
        #
        # U-NET (unet=N downsampling levels): pool to render resolution, run a pyramid
        # whose deeper levels sit at 1/2, 1/4, 1/8 of that, PixelShuffle back. Conv cost
        # is channels^2 x pixels, so a level that doubles channels while quartering
        # pixels costs the SAME -- capacity grows exponentially with depth for linear
        # cost. Measured: 906k parameters (2.4x the flat model) for 28.9 GFLOP (20x
        # less) = 1.69 ms instead of 34.6 on this GPU.
        #
        # This is the shape DLSS 4.5 and FSR 4.1 use, and it is why they fit
        # million-parameter networks inside a frame budget where this model could not
        # fit 372k: their weights run at 1/8 resolution, not at full.
        self.unet = unet
        if unet:
            from .unet import UNetTrunk
            # c*4 output channels so PixelShuffle(2) restores output resolution.
            self.encoder_unet = UNetTrunk(in_ch, c * 4, base=c, levels=unet,
                                          separable=separable)
            self.encoder_up = nn.PixelShuffle(2)
        else:
            layers: list[nn.Module] = [_conv(in_ch, c, 3, separable=separable), nn.GELU()]
            for _ in range(encoder_depth - 1):
                layers += [_conv(c, c, 3, separable=separable), nn.GELU()]
            self.encoder = nn.Sequential(*layers)
        # Selective SSM parameter projections. 1x1 convs: the spatial context is
        # already baked in by the encoder, and per-pixel projections keep this
        # cheap enough to be portable to a shader later.
        # State layout: [colour(3) | router logits(K, if routing) | learned(rest)]
        k = num_experts
        self.n_router = k if k > 1 else 0
        self.n_learned = n - 3 - self.n_router
        assert self.n_learned >= 1, (
            f"state_channels={n} too small for 3 colour + {self.n_router} router channels"
        )

        self.to_delta = nn.Conv2d(c, self.n_learned, 1)
        self.to_b = nn.Conv2d(c, self.n_learned, 1)
        self.to_c = nn.Conv2d(c, self.n_learned, 1)
        self.to_x = nn.Conv2d(c, self.n_learned, 1)
        # Learned initialisation for disoccluded pixels -- FSR just zeroes them.
        self.disocclusion_init = nn.Parameter(torch.zeros(self.n_learned))

        # --- history rectification (RectifyHistory, accumulate.h:44) --------
        # FSR's actual detail-preservation mechanism, which the pure learned
        # blend lacks: warped history is clamped into the current neighbourhood's
        # anisotropic YCoCg colour box, so stale smooth history over a now-sharp
        # region is pulled back to the box and the edge survives. FSR picks the
        # box width (box_scale in [1, box_max]) from hand-tuned velocity/
        # accumulation rules; here it is a per-pixel *learned* gate, keeping the
        # model a strict generalisation of FSR (wide box == trust history).
        # NB (v8 diagnostic): with box_max=3 the learned scale drifts to ~2.84
        # (nearly no clamp) because L1 rewards trusting smooth history. Capping
        # box_max low (e.g. 1.8) forces the clamp to actually bite -> more SSIM.
        self.rectify = rectify
        if rectify:
            self.to_boxscale = nn.Conv2d(c, 1, 1)
            # Bias toward the *tight* end so rectification is active from the
            # start; the model can still widen locally under motion where a hard
            # clamp would cause instability. sigmoid(-0.4)=0.40.
            nn.init.zeros_(self.to_boxscale.weight)
            nn.init.constant_(self.to_boxscale.bias, -0.4)
            # Anisotropy: luma gets 1.7x the chroma slack, exactly as FSR.
            self.register_buffer(
                "_aniso", torch.tensor([1.7, 1.0, 1.0]).view(1, 3, 1, 1)
            )

        # --- learned spatial resolve refiner -------------------------------
        # Every tuning/capacity/loss lever plateaued at the fixed-Lanczos-resolve
        # ceiling (~0.9425 SSIM, matching hand-tuned FSR math). The one thing kept
        # fixed the whole time was the spatial resolve. This is a small dilated-conv
        # refiner on the Lanczos upsample (fed the neighbourhood box so it knows the
        # local contrast/structure it may reconstruct), producing a residual on the
        # resolve. Dilations 1/2/3 give a wide receptive field to reconstruct edge
        # structure a single 3x3 Lanczos tap cannot. Zero-init last layer => starts
        # as exactly the Lanczos resolve, then learns to sharpen the reconstruction.
        # How far either resolve-refiner (conv or Swin) may push the resolve, in units
        # of local neighbourhood stddev. Shared so both paths get the same discipline.
        if learned_resolve or swin_resolve:
            self.refine_gain = nn.Parameter(torch.tensor(0.5))

        self.learned_resolve = learned_resolve
        if learned_resolve:
            rc = feature_channels
            self.resolve_refine = nn.Sequential(
                nn.Conv2d(9, rc, 3, padding=1, dilation=1), nn.GELU(),
                nn.Conv2d(rc, rc, 3, padding=2, dilation=2), nn.GELU(),
                nn.Conv2d(rc, rc, 3, padding=3, dilation=3), nn.GELU(),
                nn.Conv2d(rc, 3, 3, padding=1),
            )
            nn.init.zeros_(self.resolve_refine[-1].weight)
            nn.init.zeros_(self.resolve_refine[-1].bias)

        # --- windowed-attention (Swin) resolve refiner ----------------------
        # Phase 2. The conv refiner above shares the weakness of every other lever
        # tried: purely local reconstruction. Self-attention lets a pixel consult
        # structurally similar pixels elsewhere in its window, which is the direct
        # counter to the diagnosed failure (fine *repetitive* texture -- chess
        # marble/checkerboard). Windowed + shifted so cost stays linear at 1080p.
        self.swin_resolve = swin_resolve
        if swin_resolve:
            from .swin import SwinRefiner
            self.swin_refine = SwinRefiner(in_ch=9, dim=swin_dim, heads=3, window=8, depth=2)

        # --- LR-domain learned upsampler ------------------------------------
        # The one component every previous variant lacked: direct access to the
        # raw low-res image. See lrnet.py for why that is likely the real cap.
        self.separable = separable
        self.lr_upsampler = lr_upsampler
        if lr_upsampler:
            from .lrnet import LRUpsampler
            s_h = output_size[0] // render_size[0]
            s_w = output_size[1] // render_size[1]
            if s_h != s_w or s_h * render_size[0] != output_size[0]:
                raise ValueError(f"lr_upsampler needs an integer uniform scale, got "
                                 f"{render_size} -> {output_size}")
            self.lr_up = LRUpsampler(scale=s_h, dim=lr_dim, depth=4, film=lr_film,
                                     separable=separable)
            # How far the learned residual may push the resolve, in units of local
            # neighbourhood stddev. Learnable but starts modest; see forward().
            self.lr_gain = nn.Parameter(torch.tensor(0.5))

        # --- experts -------------------------------------------------------
        # Each expert owns a blend-weight head and a residual head. The blend
        # weight is the natural thing to specialise: "how much do I trust
        # history here" is a genuinely different question for in-world text than
        # for a grass texture, which is the whole motivation for routing.
        self.alpha_heads = nn.ModuleList(
            [_conv(self.n_learned + c, 1, 3, separable=separable) for _ in range(k)]
        )
        for head in self.alpha_heads:
            nn.init.zeros_(head.weight)
            # Bias so the initial alpha ~= 0.045, matching FSR's converged blend
            # rate. Without this the model starts at alpha = 0.5 and throws away
            # most of its history every frame.
            nn.init.constant_(head.bias, -3.05)

        # Kernel-prediction output head. Replaces `lerp(hist, resolve, alpha) +
        # residual` with a softmax-weighted combination of real candidate colours,
        # so the output is a convex combination and cannot drift through the
        # recurrence. See kpn.py for the full rationale.
        self.kernel_predict = kernel_predict
        if kernel_predict:
            from .kpn import KernelPredictHead, lr_taps_at_output_res, phase_planes
            self._lr_taps = lr_taps_at_output_res
            self._phase_planes = phase_planes
            self.kpn_head = KernelPredictHead(self.n_learned + c, separable=separable)

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    _conv(self.n_learned + 3, c, 3, separable=separable),
                    nn.GELU(),
                    _conv(c, 3, 3, separable=separable),
                )
                for _ in range(k)
            ]
        )
        # Start as a no-op: the initial model is exactly a temporal accumulator
        # with a constant blend weight, and nothing else.
        for dec in self.decoders:
            nn.init.zeros_(dec[-1].weight)
            nn.init.zeros_(dec[-1].bias)

        if k > 1:
            self.router = nn.Conv2d(c, k, 3, padding=1)
            # Router hysteresis. This is the crux of the routing contribution:
            # a router that decides afresh every frame will flip its mind on
            # borderline pixels and that flicker reads as shimmer -- which is
            # worse than not routing at all. Blending this frame's logits with
            # the *reprojected* logits from last frame gives the decision
            # memory, and costs one sigmoid.
            #
            # Prior patch-routing work (ClassSR, SkipVSR et al.) never had to
            # solve this, because it is all single-frame. That is precisely
            # where the novelty of this project sits.
            self.router_inertia = nn.Parameter(torch.tensor(1.0))

        # --- sharpening head (RCAS analogue) --------------------------------
        # FSR wins SSIM/structure via RCAS: a contrast-adaptive sharpen applied
        # as a *separate output pass*, never fed back into history. We mirror
        # that as a learned unsharp mask -- high-freq = out - lowpass(out), added
        # back with a per-pixel, non-negative (sharpen-only) gain the loss
        # trains. Applied to the display output only; history stays unsharpened
        # so ringing can't compound across frames (the reason FSR keeps RCAS out
        # of the accumulation loop).
        if sharpen:
            # A real detail-synthesis head, not a single-gain unsharp: fed the
            # output *and* its high-frequency band (bias toward high-freq ops),
            # it has the capacity+receptive field to reconstruct structure a
            # 3x3 gain cannot. Near-zero final init => starts as a no-op and
            # learns to add detail. Applied to the display output only.
            # 9-ch input: output(3) + output high-freq(3) + current-frame
            # resolve high-freq(3). The last is the key: it is sharp detail from
            # THIS frame's Lanczos resolve, which the temporal blend washes out
            # of the accumulated output -- the head re-injects it.
            self.sharpen_net = nn.Sequential(
                nn.Conv2d(9, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 3, 3, padding=1),
            )
            nn.init.zeros_(self.sharpen_net[-1].weight)
            nn.init.zeros_(self.sharpen_net[-1].bias)

        # --- sub-pixel alignment correction ---------------------------------
        # Diagnostic (diag_align.py) showed the resolved output is offset from GT
        # by ~half a pixel (SSIM peaks at a nonzero shift), which no content loss
        # can fix -- a coordinate-convention error in the torch resolve/warp that
        # real FSR doesn't have. Two learnable output-pixel offsets, applied as a
        # final resample, let the model discover and cancel the constant offset.
        # Starts at zero (no-op).
        self.align_shift = nn.Parameter(torch.zeros(2))

        self.to(self.dev)

    def _apply_align(self, ycocg: torch.Tensor) -> torch.Tensor:
        """Resample (1,3,H,W) by the learned sub-pixel offset to align to GT."""
        h, w = self.output_size
        shift = self.align_shift / torch.tensor([float(w), float(h)], device=ycocg.device)
        uv = self._resolve._hr_uv + shift  # (H,W,2) pixel-centre UV + offset
        grid = (uv * 2 - 1).unsqueeze(0)
        return F.grid_sample(ycocg, grid, mode="bilinear", padding_mode="border",
                             align_corners=False)

    def _sharpen(self, ycocg: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Learned detail head on (1,3,H,W). Re-injects current-frame detail.

        ``ref`` is the sharp current-frame resolve (pre-blend); its high-freq is
        the detail the accumulator blurred away, so the head can synthesise it
        back aligned to this frame instead of amplifying absent signal.
        """
        high = ycocg - F.avg_pool2d(ycocg, 3, stride=1, padding=1)
        ref_high = ref - F.avg_pool2d(ref, 3, stride=1, padding=1)
        return ycocg + self.sharpen_net(torch.cat([ycocg, high, ref_high], dim=1))

    # -- auxiliary losses ---------------------------------------------------

    def load_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer-style load balancing over the last forward pass.

        Without this, routers reliably collapse onto a subset of experts: the
        first expert to become slightly better attracts more pixels, which makes
        it better still, and the rest starve. Measured on the first 4-expert run
        here, usage was [0.554, 0.032, 0.004, 0.411] -- two live experts and two
        dead ones, which is a 2-expert model paying 4 experts' compute.

        The loss is K * sum_i (fraction routed to i) * (mean weight of i), which
        is minimised when both are uniform.
        """
        w = getattr(self, "_last_weights", None)
        if w is None:
            return torch.zeros((), device=self.dev)
        k = w.shape[1]
        frac = torch.zeros(k, device=w.device)
        win = w.argmax(dim=1).flatten()
        frac.scatter_add_(0, win, torch.ones_like(win, dtype=w.dtype))
        frac = frac / win.numel()
        mean_w = w.mean(dim=(0, 2, 3))
        return k * (frac * mean_w).sum()

    # -- warp --------------------------------------------------------------

    def _warp(self, hidden: torch.Tensor, mv_hr: torch.Tensor) -> torch.Tensor:
        """Move the state along the motion vectors.

        Same operation as the baseline's history reprojection, applied to N
        learned channels instead of RGB+lock. This is the step that degrades
        state a little every frame, and the open question flagged in
        ORIENTATION.md -- whether interpolating between two learned state
        vectors yields a meaningful state vector -- lives right here.
        """
        uv = self._resolve._hr_uv + mv_hr
        grid = (uv * 2.0 - 1.0).unsqueeze(0)
        # Bilinear. (2026-07-23: tried bicubic to fight accumulation blur -- it
        # did not move SSIM at all AND hit a CUDA "misaligned address" bug in
        # bicubic grid_sample at 1080p. The SSIM gap is not the warp interp; see
        # the ssim-ceiling memory.)
        return F.grid_sample(
            hidden.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]

    @staticmethod
    def _thin_feature_confidence(luma: torch.Tensor) -> torch.Tensor:
        """FSR's lock signal: is this pixel part of a thin high-contrast feature?

        Real FSR runs a separate lock pass (ffx_fsr3upscaler_lock.h) that finds thin
        features via a luma-neighbourhood test and *locks* them so accumulation cannot
        blend them away. `baseline.py` ports the lock *decay* but never creates new
        locks (see its docstring), so thin-feature protection has been missing on both
        sides of this project -- and thin features are exactly the fine detail that
        reads as "soft" in the output. Post-hoc sharpening cannot recover it: measured
        three times (v8, the physical FSR reimpl, and v20), amplification always made
        PSNR *and* SSIM worse, because the detail is absent rather than merely damped.

        This is the spirit of ComputeThinFeatureConfidence, not a bit-exact port: for
        each of the 4 axes through a pixel, test whether the centre is a luma extremum
        against its two opposite neighbours (with FSR's 1.05 similarity margin). A thin
        bright or dark line is an extremum across its width but not along its length,
        so ridges score high and flat regions score zero. Input/output: (1,1,H,W).
        """
        t = 1.05
        p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        h, w = luma.shape[-2:]
        conf = torch.zeros_like(luma)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = p[..., 1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
            b = p[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            hi = (luma > a * t) & (luma > b * t)
            lo = (luma * t < a) & (luma * t < b)
            conf = conf + (hi | lo).to(luma.dtype)
        return conf * 0.25

    # -- one step ----------------------------------------------------------

    def forward(
        self,
        state: MambaState,
        lr_rgb: torch.Tensor,
        mv_hr: torch.Tensor,
        depth_hr: torch.Tensor,
        jitter: tuple[float, float],
        prev_depth_hr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MambaState]:
        upsampled, _, box_center, box_stddev = self._resolve._upsample(lr_rgb, jitter)

        # --- disocclusion, same depth test the baseline uses ---
        uv = self._resolve._hr_uv + mv_hr
        inside = (
            (uv[..., 0] >= 0) & (uv[..., 0] < 1) & (uv[..., 1] >= 0) & (uv[..., 1] < 1)
        )
        if prev_depth_hr is not None:
            warped_prev = F.grid_sample(
                prev_depth_hr[None, None],
                (uv * 2 - 1).unsqueeze(0),
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            )[0, 0]
            if self.robust_disocc:
                # Neighbourhood depth test. The naive per-pixel test below compares
                # warped previous depth against the depth of exactly this pixel; at a
                # depth discontinuity the warp straddles two surfaces, so the test
                # fails and a perfectly valid surface is declared "newly disoccluded".
                # Because alpha = max(alpha, disocclusion), that throws history away
                # exactly on geometric edges -- which is where temporal AA comes from.
                # Measured on locomotive: the naive test fires on 35.6% of edge pixels
                # vs 5.6% of flat ones, and the resulting jaggies are visible.
                #
                # Fix: accept history if the warped depth matches ANY depth in a 3x3
                # neighbourhood of the current pixel (min/max envelope + tolerance).
                # At an edge the envelope spans both surfaces, so history survives; in
                # the interior the envelope is tight, so real disocclusions still fire.
                d = depth_hr[None, None]
                d_max = F.max_pool2d(d, 3, stride=1, padding=1)[0, 0]
                d_min = -F.max_pool2d(-d, 3, stride=1, padding=1)[0, 0]
                tol = 0.1
                lo = d_min * (1.0 - tol)
                hi = d_max * (1.0 + tol)
                disocclusion = ((warped_prev < lo) | (warped_prev > hi)).float()
            else:
                rel = (warped_prev - depth_hr).abs() / depth_hr.clamp(min=1e-3)
                disocclusion = (rel > 0.1).float()
        else:
            disocclusion = torch.zeros_like(depth_hr)
        is_new = (~inside) | (state.frame_index == 0)
        disocclusion = torch.maximum(disocclusion, is_new.float())

        # 4K-normalised velocity, derived from the ACTUAL output size rather than a
        # hardcoded 3840x2160.
        #
        # This was a real train/test mismatch. `crop_sequence` rescales motion vectors
        # into crop-local UV (x rw/cw = 3.75x for a 256 crop of a 960-wide render), so
        # multiplying by a fixed 3840 made the velocity feature ~3.75x too large during
        # crop training while full-frame inference saw the true value -- measured 1.84x
        # difference in the post-clamp mean the encoder actually sees. Velocity gates
        # motion-dependent behaviour (rectification box width, blend weight), so it was
        # miscalibrated at full resolution, and the full-frame PSNR deficit is uniform
        # across the frame, which is what a global feature shift predicts (a border
        # problem would concentrate at the edges -- measured, it does not).
        #
        # mv_hr is UV in whatever space we are running, so `2 * output_size` converts to
        # 4K-equivalent pixels in both regimes: full frame 2*1920 = 3840, and a 512-wide
        # crop 2*512 = 1024, which equals mv_full * 3840 after the 3.75x crop rescale.
        vel_scale = torch.tensor(
            [2.0 * self.output_size[1], 2.0 * self.output_size[0]], device=mv_hr.device
        )
        vel = torch.linalg.vector_norm(mv_hr * vel_scale, dim=-1)

        extra = ()
        if self.lock_feature:
            # Computed at render resolution from the LR luma (as FSR's lock pass does),
            # then expanded to output resolution alongside the other inputs.
            lr_luma = rgb_to_ycocg(lr_rgb)[..., 0][None, None]
            conf = self._thin_feature_confidence(lr_luma)
            conf_hr = F.interpolate(conf, size=upsampled.shape[:2], mode="bilinear",
                                    align_corners=False)[0, 0]
            extra = (conf_hr.unsqueeze(-1),)

        feats = torch.cat(
            (
                upsampled,
                box_center,
                box_stddev,
                disocclusion.unsqueeze(-1),
                (vel / 20.0).clamp(0, 1).unsqueeze(-1),
                (1.0 / depth_hr.clamp(min=1e-2)).unsqueeze(-1),
            ) + extra,
            dim=-1,
        )
        x = feats.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
        if self.unet:
            # Pool to render resolution before the pyramid: the only full-resolution
            # work left is the PixelShuffle and the elementwise blend, both trivial.
            # Little is lost -- these features derive from the low-res input anyway,
            # since the resolve is itself an upsample of it.
            enc = self.encoder_up(self.encoder_unet(F.avg_pool2d(x, 2)))
        else:
            enc = self.encoder(x)

        # --- warp the whole state along the motion vectors ---
        warped = self._warp(state.hidden, mv_hr).unsqueeze(0)
        hist_color = warped[:, :3]  # YCoCg history, FSR's 3 channels
        prev_logits = warped[:, 3 : 3 + self.n_router]  # routing decision, carried
        h_prev = warped[:, 3 + self.n_router :]  # free learned channels

        reset = disocclusion.view(1, 1, *disocclusion.shape)
        init = self.disocclusion_init.view(1, -1, 1, 1)
        h_prev = torch.lerp(h_prev, init.expand_as(h_prev), reset)

        # --- selective SSM step on the learned channels ---
        # A_bar in (0, 1): the learned "how much do I trust memory" gate. This is
        # the direct counterpart of ComputeBaseAccumulationWeight in baseline.py.
        delta = F.softplus(self.to_delta(enc))
        a_bar = torch.exp(-delta)
        h = a_bar * h_prev + (1.0 - a_bar) * self.to_b(enc) * self.to_x(enc)
        y = self.to_c(enc) * h

        # --- routing, with the decision carried through time ---
        if self.num_experts > 1:
            raw_logits = self.router(enc)
            # Hysteresis against the reprojected decision. On a disoccluded
            # pixel there is no prior decision to respect, so take this frame's
            # logits outright.
            inertia = torch.sigmoid(self.router_inertia) * (1.0 - reset)
            logits = torch.lerp(raw_logits, prev_logits, inertia)
            weights = torch.softmax(logits, dim=1)
            # Stash for the load-balancing loss (see `load_balance_loss`).
            self._last_weights = weights
        else:
            logits = prev_logits  # empty
            weights = None
            self._last_weights = None

        # --- blend, exactly the shape of accumulate.h:35-36 ---
        up_nchw = upsampled.permute(2, 0, 1).unsqueeze(0)
        if self.lr_upsampler:
            # Learned upsample straight from the raw LR image, added to the fixed
            # Lanczos resolve. This is the only path with access to LR detail the
            # 3x3 Lanczos taps discarded.
            #
            # The residual is **bounded by local neighbourhood contrast**:
            # tanh(raw) * box_stddev * gain. Without this the run collapses (v15,
            # v16, v17 all did) -- an unbounded residual on the resolve is fed into
            # the recurrent history and compounds frame over frame until the
            # accumulation is destroyed. Scaling by the box stddev says: you may
            # add detail in proportion to the detail actually present locally, and
            # essentially nothing in flat regions. Same physical logic as FSR's
            # rectification box, and it keeps the module a bounded perturbation of
            # the Lanczos resolve rather than a free-running generator.
            lr_y = rgb_to_ycocg(lr_rgb).permute(2, 0, 1).unsqueeze(0)
            jit = jitter if torch.is_tensor(jitter) else torch.tensor(
                jitter, device=lr_rgb.device)
            raw = self.lr_up(lr_y, jit)
            bs_r = box_stddev.permute(2, 0, 1).unsqueeze(0)
            up_nchw = up_nchw + torch.tanh(raw) * bs_r * self.lr_gain.abs()

        if self.learned_resolve or self.swin_resolve:
            # Refine the fixed Lanczos resolve with a learned residual (fed the
            # neighbourhood colour box) before it enters the blend, so a sharper
            # reconstruction propagates into the accumulated history too.
            bc_r = box_center.permute(2, 0, 1).unsqueeze(0)
            bs_r = box_stddev.permute(2, 0, 1).unsqueeze(0)
            rin = torch.cat((up_nchw, bc_r, bs_r), dim=1)
            refiner = self.swin_refine if self.swin_resolve else self.resolve_refine
            # Bounded by local contrast, exactly as the LR upsampler is. v15/v16 added
            # this residual raw and both collapsed: anything added to the resolve is fed
            # into the recurrent history and compounds frame over frame, so an unbounded
            # generator destroys the accumulation. tanh(raw) * box_stddev * gain lets the
            # refiner add detail in proportion to the detail actually present locally and
            # essentially nothing in flat regions -- the same discipline as FSR's
            # rectification box. Those two divergences were a stability bug, not evidence
            # that attention cannot work here.
            bs_bound = box_stddev.permute(2, 0, 1).unsqueeze(0)
            up_nchw = up_nchw + torch.tanh(refiner(rin)) * bs_bound * self.refine_gain.abs()
        alpha_in = torch.cat((y, enc), dim=1)

        if weights is None:
            alpha = torch.sigmoid(self.alpha_heads[0](alpha_in))
        else:
            # Dense combination: every expert is evaluated and mixed by weight.
            # This measures the *quality* ceiling of routing. A shippable
            # version would route whole tiles to avoid paying for every expert
            # on every wave -- see ORIENTATION.md on wave divergence. Quality
            # first, sparsity second.
            stacked = torch.cat([torch.sigmoid(hd(alpha_in)) for hd in self.alpha_heads], dim=1)
            alpha = (stacked * weights).sum(dim=1, keepdim=True)

        # --- RectifyHistory (accumulate.h:44), learned box width -------------
        # Clamp warped history into the neighbourhood colour box before blending.
        # This is the detail-preservation step the pure learned blend was missing:
        # smooth stale history over a now-sharp region is pulled to the box edge,
        # so the current sharp resolve dominates there. The box half-width is a
        # learned per-pixel gate in [1,3]x the neighbourhood stddev (FSR's
        # hand-tuned box_scale, generalised).
        if self.rectify:
            bc = box_center.permute(2, 0, 1).unsqueeze(0)
            bs = box_stddev.permute(2, 0, 1).unsqueeze(0)
            box_scale = 1.0 + (self.box_max - 1.0) * torch.sigmoid(self.to_boxscale(enc))
            scaled = (bs * self._aniso * box_scale).clamp(min=1.193e-7)
            transformed = (hist_color - bc) / scaled
            norm = torch.linalg.vector_norm(transformed, dim=1, keepdim=True)
            outside = (norm > 1.0).float()
            clamped = transformed / norm.clamp(min=1e-8) * scaled + bc
            hist_color = outside * clamped + (1.0 - outside) * hist_color

        # A disoccluded pixel has no history worth blending: take the resolve.
        alpha = torch.maximum(alpha, reset)
        # Stashed for diagnostics: temporal anti-aliasing comes from *accumulating*
        # jittered sub-pixel samples, so if alpha saturates high on edges the model
        # is throwing history away exactly where AA should come from -> jaggies.
        # alpha_scale (default 1.0 = unchanged): shrinks the weight on the current
        # frame, forcing deeper accumulation. Measured motivation: in flat regions the
        # trained model uses alpha ~0.165 while FSR converges to ~0.045, so it averages
        # over ~3.7x fewer jittered samples. The Lanczos resolve has a jitter-dependent
        # tap-window bias that only cancels when many phases are averaged, which is why
        # our sub-pixel phase spread is ~2x FSR's (43% vs 23%) and reads as 2x2
        # pixelation. Disocclusion is re-applied after, so genuinely new pixels still
        # take the resolve at full weight and cannot smear.
        if self.alpha_scale != 1.0:
            alpha = alpha * self.alpha_scale
            alpha = torch.maximum(alpha, reset)
        self._last_alpha = alpha
        self._last_reset = reset
        blended = torch.lerp(hist_color, up_nchw, alpha)

        if self.kernel_predict:
            # Predict per-pixel weights over real candidate colours instead of a
            # colour residual. `alpha` and the rectified history are still computed
            # above -- the disocclusion reset is folded in by handing the head the
            # already-rectified history and letting it learn to ignore it -- but the
            # additive decoder path is bypassed entirely, so nothing unbounded is
            # ever fed back into the recurrent state.
            lr_y_k = rgb_to_ycocg(lr_rgb).permute(2, 0, 1).unsqueeze(0)
            out_hw = (up_nchw.shape[-2], up_nchw.shape[-1])
            taps = self._lr_taps(lr_y_k, out_hw)
            phase = self._phase_planes(out_hw, self.render_size, jitter,
                                       up_nchw.device, up_nchw.dtype)
            out_ycocg = self.kpn_head(alpha_in, taps, hist_color, up_nchw, phase)
            # On a genuinely disoccluded pixel there is no valid history at all, so
            # force the resolve rather than trusting the head to have learned it.
            out_ycocg = reset * up_nchw + (1.0 - reset) * out_ycocg
        else:
            dec_in = torch.cat((y, blended), dim=1)
            if weights is None:
                residual = self.decoders[0](dec_in)
            else:
                stacked_r = torch.stack([dec(dec_in) for dec in self.decoders], dim=1)
                residual = (stacked_r * weights.unsqueeze(2)).sum(dim=1)

            out_ycocg = blended + residual
        # Sharpen the display output only; the un-sharpened out_ycocg is what
        # feeds history below (RCAS discipline -- no ringing build-up).
        # NB: align_shift was a dead end -- the SSIM ceiling is NOT a sub-pixel
        # offset (grid-search: best fixed shift gained +0.001 SSIM). Kept behind
        # `sharpen` only so the sharpen-experiment checkpoints still load.
        if self.sharpen:
            disp_ycocg = self._apply_align(self._sharpen(out_ycocg, up_nchw))
        else:
            disp_ycocg = out_ycocg
        out_rgb = ycocg_to_rgb(disp_ycocg[0].permute(1, 2, 0)).clamp(min=0.0)

        # History channels carry the output forward, as FSR does at
        # accumulate.h:165; the router logits and SSM state ride alongside.
        parts = [out_ycocg] + ([logits] if self.num_experts > 1 else []) + [h]
        new_hidden = torch.cat(parts, dim=1)[0]
        return out_rgb, MambaState(hidden=new_hidden, frame_index=state.frame_index + 1)
