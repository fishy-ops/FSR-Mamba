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
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.render_size = render_size
        self.output_size = output_size
        self.state_channels = state_channels
        self.num_experts = num_experts
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
        in_ch = 12
        c = feature_channels
        n = state_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
        )
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

        # --- experts -------------------------------------------------------
        # Each expert owns a blend-weight head and a residual head. The blend
        # weight is the natural thing to specialise: "how much do I trust
        # history here" is a genuinely different question for in-world text than
        # for a grass texture, which is the whole motivation for routing.
        self.alpha_heads = nn.ModuleList(
            [nn.Conv2d(self.n_learned + c, 1, 3, padding=1) for _ in range(k)]
        )
        for head in self.alpha_heads:
            nn.init.zeros_(head.weight)
            # Bias so the initial alpha ~= 0.045, matching FSR's converged blend
            # rate. Without this the model starts at alpha = 0.5 and throws away
            # most of its history every frame.
            nn.init.constant_(head.bias, -3.05)

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.n_learned + 3, c, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c, 3, 3, padding=1),
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

        self.to(self.dev)

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
        return F.grid_sample(
            hidden.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]

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
            rel = (warped_prev - depth_hr).abs() / depth_hr.clamp(min=1e-3)
            disocclusion = (rel > 0.1).float()
        else:
            disocclusion = torch.zeros_like(depth_hr)
        is_new = (~inside) | (state.frame_index == 0)
        disocclusion = torch.maximum(disocclusion, is_new.float())

        vel = torch.linalg.vector_norm(
            mv_hr * torch.tensor([3840.0, 2160.0], device=mv_hr.device), dim=-1
        )

        feats = torch.cat(
            (
                upsampled,
                box_center,
                box_stddev,
                disocclusion.unsqueeze(-1),
                (vel / 20.0).clamp(0, 1).unsqueeze(-1),
                (1.0 / depth_hr.clamp(min=1e-2)).unsqueeze(-1),
            ),
            dim=-1,
        )
        x = feats.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
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

        # A disoccluded pixel has no history worth blending: take the resolve.
        alpha = torch.maximum(alpha, reset)
        blended = torch.lerp(hist_color, up_nchw, alpha)

        dec_in = torch.cat((y, blended), dim=1)
        if weights is None:
            residual = self.decoders[0](dec_in)
        else:
            stacked_r = torch.stack([dec(dec_in) for dec in self.decoders], dim=1)
            residual = (stacked_r * weights.unsqueeze(2)).sum(dim=1)

        out_ycocg = blended + residual
        out_rgb = ycocg_to_rgb(out_ycocg[0].permute(1, 2, 0)).clamp(min=0.0)

        # History channels carry the output forward, as FSR does at
        # accumulate.h:165; the router logits and SSM state ride alongside.
        parts = [out_ycocg] + ([logits] if self.num_experts > 1 else []) + [h]
        new_hidden = torch.cat(parts, dim=1)[0]
        return out_rgb, MambaState(hidden=new_hidden, frame_index=state.frame_index + 1)
