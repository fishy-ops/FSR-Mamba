"""FSR 3.1.4's temporal accumulator, ported to PyTorch.

This is the thing to beat. It is a deliberately faithful transcription of

    fsr-upstream/sdk/include/FidelityFX/gpu/fsr3upscaler/
        ffx_fsr3upscaler_accumulate.h      (the recurrence itself)
        ffx_fsr3upscaler_reproject.h       (history warp)
        ffx_fsr3upscaler_upsample.h        (Lanczos resolve + rectification box)
        ffx_fsr3upscaler_common.h          (YCoCg, rectification box maths)

Line references appear next to each ported block so you can diff against the HLSL.

Why port it at all, rather than just calling FSR?
-------------------------------------------------
Three reasons, in order of importance:

1. It is the baseline. Any claim that a learned accumulator is better is
   meaningless without a same-pipeline, same-data reference number.
2. It validates the data pipeline. If this implementation produces sensible
   output on synthetic scenes, the scenes and their motion vectors are correct.
   If it produces garbage, the bug is in the data, not the model -- and finding
   that out now is much cheaper than finding it out after training something.
3. It forces genuine understanding of the heuristics being replaced.

What is faithful and what is not
--------------------------------
Faithful: the recurrence structure, the YCoCg rectification box (weighted mean
and standard deviation), the anisotropic box scaling, the history-clamp
blend, the accumulation weight update, and the Lanczos-2 resolve.

Simplified, and these matter:

  * **The four masks** (reactive / disocclusion / shading-change / accumulation)
    are produced by upstream passes not ported here. Disocclusion is computed
    from a depth test, accumulation from a frame counter, and reactive /
    shading-change are stubbed to zero. On synthetic scenes with no transparency
    and no shading that is nearly right; on engine data it would not be.
  * **The lock mechanism** is approximated. Real FSR detects thin features in a
    separate pass (ffx_fsr3upscaler_lock.h) using a luma-neighbourhood test.
    Here `UpdateLockStatus`'s decay logic is ported but new locks are never
    created, so thin-feature protection is weaker than the real thing.
  * **Luma instability** (ffx_fsr3upscaler_luma_instability.h) is not ported;
    the factor is held at zero.

Net effect: this baseline is somewhat *worse* than real FSR 3.1.4, mostly on
thin features. Treat its numbers as a floor, not as "FSR's score" -- and do not
publish a comparison against it as though it were AMD's shipping quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .colorspace import rgb_to_ycocg, ycocg_to_rgb

__all__ = ["FSRState", "FSRAccumulator"]

_EPSILON = 1e-6
_FP16_MIN = 6.10e-5

# common.h:46-48. The scale is the crux of the whole algorithm: each frame's
# resolve contributes only ~0.046 of weight against a history weight of up to
# 1.0, so a pixel converges over roughly 20 frames rather than being overwritten
# every frame. Drop this factor and temporal accumulation silently stops working
# while everything still *looks* plausible -- which is exactly what happened on
# the first version of this port.
_UPSAMPLE_LANCZOS_WEIGHT_SCALE = 1.0 / 16.0
_AVERAGE_LANCZOS_WEIGHT_PER_FRAME = 0.74 * _UPSAMPLE_LANCZOS_WEIGHT_SCALE

# common.h:55-56
_LOCK_THRESHOLD = 1.0
_LOCK_MAX = 2.0

# upsample.h:609. The rectification box uses a Gaussian-ish falloff, NOT the
# Lanczos kernel used for the resolve. They are different filters for different
# jobs and mixing them up widens the clamp box incorrectly.
_RECTIFICATION_CURVE_BIAS = -2.3


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


def _lanczos2(x: torch.Tensor) -> torch.Tensor:
    """Lanczos kernel with 2 lobes, as used by the resolve.

    FSR approximates this with a LUT (FFX_FSR3UPSCALER_GET_LANCZOS_SAMPLER1D);
    we evaluate it directly, which is marginally more accurate and irrelevant
    to quality at this stage.
    """
    x = x.abs().clamp(max=2.0)
    pix = math.pi * x
    # sinc(x) * sinc(x/2), guarding x -> 0.
    safe = torch.where(x < 1e-5, torch.ones_like(pix), pix)
    val = torch.where(
        x < 1e-5,
        torch.ones_like(x),
        (torch.sin(safe) / safe) * (torch.sin(safe / 2) / (safe / 2)),
    )
    return torch.where(x >= 2.0, torch.zeros_like(val), val)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


@dataclass
class FSRState:
    """The recurrent state -- FSR's is exactly 4 channels wide.

    See accumulate.h:165:
        StoreInternalColorAndWeight(iPxHrPos, float4(fHistoryColor, fLock))

    `history_rgb` is (H, W, 3) linear RGB, `lock` is (H, W).

    This is the structure a Mamba accumulator would widen: same per-pixel
    recurrence, same reprojection, but N learned channels instead of 3+1 and a
    learned update rule instead of the heuristics below.
    """

    history_rgb: torch.Tensor
    lock: torch.Tensor
    accumulation: torch.Tensor  # per-pixel frames-accumulated, normalised to [0,1]
    frame_index: int = 0

    @classmethod
    def zeros(cls, size: tuple[int, int], device="cpu") -> "FSRState":
        h, w = size
        return cls(
            history_rgb=torch.zeros((h, w, 3), device=device),
            lock=torch.zeros((h, w), device=device),
            accumulation=torch.zeros((h, w), device=device),
        )


# --------------------------------------------------------------------------
# Accumulator
# --------------------------------------------------------------------------


class FSRAccumulator:
    """One frame of FSR's temporal accumulation, as a callable object."""

    def __init__(
        self,
        render_size: tuple[int, int],
        output_size: tuple[int, int],
        device: torch.device | str = "cpu",
        lock_max: float = _LOCK_MAX,
        lock_threshold: float = _LOCK_THRESHOLD,
    ) -> None:
        self.render_size = render_size
        self.output_size = output_size
        self.device = torch.device(device)
        self.lock_max = lock_max
        self.lock_threshold = lock_threshold

        h, w = output_size
        ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
        self._hr_x, self._hr_y = torch.meshgrid(xs, ys, indexing="xy")
        self._hr_uv = torch.stack((self._hr_x / w, self._hr_y / h), dim=-1)

    # -- resolve -----------------------------------------------------------

    def _upsample(
        self,
        lr_rgb: torch.Tensor,
        jitter: tuple[float, float],
        kernel_bias: torch.Tensor | None = None,
    ):
        """Lanczos-2 resolve of jittered low-res samples onto the output grid.

        Ports ComputeUpsampledColorAndWeight (upsample.h:301, reference path at
        :585-641) together with the rectification box (common.h:149-182).

        Three details here are easy to get wrong and all three matter:

        * The Lanczos kernel is **radial** -- ``Lanczos2(length(offset))`` --
          not a separable product of two 1D kernels (upsample.h:36).
        * The rectification box uses a **different** kernel,
          ``exp(-2.3 * |offset|^2)`` (upsample.h:611).
        * The returned weight is scaled by ``fAverageLanczosWeightPerFrame``
          (upsample.h:629), which is what makes this a *sliced* filter
          accumulated over many frames rather than a per-frame resolve.

        Returns (upsampled_ycocg, upsampled_weight, box_center, box_stddev).
        """
        lr_h, lr_w = self.render_size
        out_h, out_w = self.output_size
        dev = self.device

        # HR pixel centre expressed in low-res pixel space (upsample.h:305).
        src_x = self._hr_x * (lr_w / out_w)
        src_y = self._hr_y * (lr_h / out_h)
        base_x = torch.floor(src_x)
        base_y = torch.floor(src_y)

        # Un-jittered position of the base sample (upsample.h:307), which decides
        # whether the 3x3 window sits at offset -1 or -2.
        unjit_x = base_x + 0.5 - jitter[0]
        unjit_y = base_y + 0.5 - jitter[1]
        off_tl_x = torch.where(unjit_x > src_x, -2.0, -1.0)
        off_tl_y = torch.where(unjit_y > src_y, -2.0, -1.0)

        if kernel_bias is None:
            kernel_bias = torch.ones((out_h, out_w), device=dev)

        lr_ycocg = rgb_to_ycocg(lr_rgb)

        color_acc = torch.zeros((out_h, out_w, 3), device=dev)
        weight_acc = torch.zeros((out_h, out_w), device=dev)
        box_sum = torch.zeros((out_h, out_w, 3), device=dev)
        box_sqsum = torch.zeros((out_h, out_w, 3), device=dev)
        box_w = torch.zeros((out_h, out_w), device=dev)
        aabb_min = torch.full((out_h, out_w, 3), float("inf"), device=dev)
        aabb_max = torch.full((out_h, out_w, 3), float("-inf"), device=dev)

        for row in range(3):
            for col in range(3):
                px = base_x + off_tl_x + col
                py = base_y + off_tl_y + row

                on_screen = (
                    (px >= 0) & (px < lr_w) & (py >= 0) & (py < lr_h)
                ).float()
                qx = px.clamp(0, lr_w - 1).long()
                qy = py.clamp(0, lr_h - 1).long()
                sample = lr_ycocg[qy, qx]

                # Offset from this texel's un-jittered centre to the output point.
                ox = (px + 0.5 - jitter[0]) - src_x
                oy = (py + 0.5 - jitter[1]) - src_y
                off_sq = ox * ox + oy * oy

                # Resolve weight: radial Lanczos-2 of the bias-scaled offset.
                w = _lanczos2(torch.sqrt(off_sq) * kernel_bias) * on_screen
                color_acc += sample * w.unsqueeze(-1)
                weight_acc += w

                # Box weight: Gaussian-ish falloff, a different kernel entirely.
                bw = torch.exp(_RECTIFICATION_CURVE_BIAS * off_sq) * on_screen
                box_sum += sample * bw.unsqueeze(-1)
                box_sqsum += sample * sample * bw.unsqueeze(-1)
                box_w += bw

                valid = on_screen.bool().unsqueeze(-1)
                aabb_min = torch.where(valid, torch.minimum(aabb_min, sample), aabb_min)
                aabb_max = torch.where(valid, torch.maximum(aabb_max, sample), aabb_max)

        # RectificationBoxComputeVarianceBoxData (common.h:175-182).
        safe_w = torch.where(box_w.abs() > _FP16_MIN, box_w, torch.ones_like(box_w))
        center = box_sum / safe_w.unsqueeze(-1)
        mean_sq = box_sqsum / safe_w.unsqueeze(-1)
        stddev = torch.sqrt((mean_sq - center * center).abs())

        # upsample.h:626-633: normalise the colour, then scale the weight down so
        # this frame is one slice of a filter spread across many.
        has_weight = weight_acc > _EPSILON
        safe_weight = torch.where(has_weight, weight_acc, torch.ones_like(weight_acc))
        upsampled = color_acc / safe_weight.unsqueeze(-1)
        # Deringing (upsample.h:23-26): clamp to the neighbourhood AABB.
        upsampled = torch.clamp(torch.maximum(upsampled, aabb_min), max=aabb_max)
        upsampled = torch.where(has_weight.unsqueeze(-1), upsampled, center)

        weight = torch.where(
            has_weight,
            weight_acc * _AVERAGE_LANCZOS_WEIGHT_PER_FRAME,
            torch.zeros_like(weight_acc),
        )
        return upsampled, weight, center, stddev

    # -- warp --------------------------------------------------------------

    def _reproject(
        self, state: FSRState, mv_hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the state along the motion vectors.

        Ports ReprojectHistoryColor (reproject.h:63) and ComputeReprojectedUVs
        (reproject.h:56). FSR resamples with a Lanczos-weighted bicubic; we use
        `grid_sample`'s bicubic, which is close enough for a baseline and is the
        same *class* of filter -- lossy interpolation of state, applied afresh
        every frame.

        Returns (history_rgb, lock, accumulation, is_existing_sample).
        """
        reproj_uv = self._hr_uv + mv_hr  # reproject.h:58
        inside = (
            (reproj_uv[..., 0] >= 0)
            & (reproj_uv[..., 0] < 1)
            & (reproj_uv[..., 1] >= 0)
            & (reproj_uv[..., 1] < 1)
        )

        grid = (reproj_uv * 2.0 - 1.0).unsqueeze(0)  # NHWC in [-1, 1]
        packed = torch.cat(
            (state.history_rgb, state.lock.unsqueeze(-1), state.accumulation.unsqueeze(-1)),
            dim=-1,
        )
        sampled = F.grid_sample(
            packed.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        )[0].permute(1, 2, 0)

        return sampled[..., :3], sampled[..., 3], sampled[..., 4], inside

    # -- the recurrence ----------------------------------------------------

    def __call__(
        self,
        state: FSRState,
        lr_rgb: torch.Tensor,
        mv_hr: torch.Tensor,
        depth_hr: torch.Tensor,
        jitter: tuple[float, float],
        prev_depth_hr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, FSRState]:
        """Run one accumulation step. Mirrors Accumulate() at accumulate.h:141.

        Returns (upscaled_rgb, next_state).
        """
        hist_rgb, lock, accumulation, existing = self._reproject(state, mv_hr)

        is_new = (~existing) | (state.frame_index == 0)

        # --- masks (simplified -- see module docstring) ---
        if prev_depth_hr is not None:
            warped_prev_depth = F.grid_sample(
                prev_depth_hr[None, None],
                ((self._hr_uv + mv_hr) * 2 - 1).unsqueeze(0),
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            )[0, 0]
            rel = (warped_prev_depth - depth_hr).abs() / depth_hr.clamp(min=1e-3)
            disocclusion = (rel > 0.1).float()
        else:
            disocclusion = torch.zeros_like(depth_hr)
        disocclusion = torch.maximum(disocclusion, is_new.float())
        reactive = torch.zeros_like(disocclusion)
        shading_change = torch.zeros_like(disocclusion)
        luma_instability = torch.zeros_like(disocclusion)

        # UpdateAccumulation (prepare_reactivity.h:204-238). Note the two-value
        # dance: `accumulation` is what *this* frame uses, and it is the
        # reprojected value from previous frames -- zero on the first frame and
        # after a disocclusion. The +0.333 produces the value stored for the
        # *next* frame. Collapsing these into one variable makes a fresh pixel
        # behave as though it already had a frame of history.
        accumulation = torch.where(
            disocclusion > 0.5, torch.zeros_like(accumulation), accumulation
        ).clamp(0, 1)
        accumulation = accumulation * (torch.round(accumulation * 100.0) > 1.0).float()
        is_initial = accumulation == 0.0
        # AccumulationAddedPerFrame, default 0.333 -- converged in ~3 frames.
        accumulation_next = (accumulation + 0.333).clamp(max=1.0)

        hist_ycocg = rgb_to_ycocg(hist_rgb)

        # --- 4K-normalised velocity, used to widen the clamp box under motion ---
        vel_px = torch.linalg.vector_norm(
            mv_hr * torch.tensor([3840.0, 2160.0], device=self.device), dim=-1
        )

        # --- UpdateLockStatus (accumulate.h:72) ---
        lock = lock * (~is_new).float()
        lifetime_decrease_factor = torch.maximum(
            shading_change.clamp(0, 1), torch.maximum(reactive, disocclusion)
        )
        lock = (lock - lifetime_decrease_factor * self.lock_max).clamp(min=0.0)
        lock_contribution = (
            ((lock - self.lock_threshold).clamp(0, 1) * (self.lock_max - self.lock_threshold))
            .clamp(0, 1)
        )
        # NOTE: new locks are never added -- the detection pass is not ported.
        lifetime_decrease = (0.1 / 16.0) * (1.0 - lifetime_decrease_factor)
        lock = (lock - lifetime_decrease).clamp(min=0.0)

        # --- ComputeBaseAccumulationWeight (accumulate.h:95) ---
        base_accum = accumulation
        base_accum = torch.minimum(
            base_accum,
            torch.lerp(base_accum, torch.full_like(base_accum, 0.15), (vel_px / 0.5).clamp(0, 1)),
        )
        history_weight = base_accum

        # --- ComputeUpsampledColorAndWeight (accumulate.h:155) ---
        # Ordered after the accumulation weight because the kernel bias depends
        # on it (upsample.h:439-448): a well-converged pixel gets a wider,
        # sharper kernel; a freshly disoccluded one gets a narrow, safe kernel.
        lr_h, lr_w = self.render_size
        out_h, out_w = self.output_size
        kernel_bias_max = min(1.99, 1.0 + (out_w / lr_w - 1.0))
        kernel_bias_min = max(1.0, (1.0 + kernel_bias_max) * 0.3)
        kernel_bias_weight = torch.minimum(
            1.0 - disocclusion * 0.5,
            torch.minimum(1.0 - shading_change, (history_weight * 5.0).clamp(0, 1)),
        )
        kernel_bias = torch.lerp(
            torch.full_like(kernel_bias_weight, kernel_bias_min),
            torch.full_like(kernel_bias_weight, kernel_bias_max),
            kernel_bias_weight,
        )
        upsampled, upsampled_weight, box_center, box_stddev = self._upsample(
            lr_rgb, jitter, kernel_bias
        )

        # upsample.h:635-641: an initial sample has no history to slice against,
        # so it takes the box centre outright at full weight.
        init = is_initial.unsqueeze(-1)
        upsampled = torch.where(init, box_center, upsampled)
        upsampled_weight = torch.where(
            is_initial, torch.ones_like(upsampled_weight), upsampled_weight
        )
        history_weight = torch.where(
            is_initial, torch.zeros_like(history_weight), history_weight
        )

        # --- RectifyHistory (accumulate.h:44) ---
        f4k = (vel_px / 20.0).clamp(0, 1)
        distance_factor = (0.75 - depth_hr / 20.0).clamp(0, 1)
        accumulation_factor = 1.0 - accumulation
        reactive_factor = reactive.clamp(min=0).sqrt()
        box_scale_t = torch.maximum(
            f4k,
            torch.maximum(
                distance_factor,
                torch.maximum(accumulation_factor, torch.maximum(reactive_factor, shading_change)),
            ),
        )
        box_scale = torch.lerp(
            torch.full_like(box_scale_t, 3.0), torch.ones_like(box_scale_t), box_scale_t
        )
        # Anisotropic: luma is allowed 1.7x the chroma slack (accumulate.h:57).
        aniso = torch.tensor([1.7, 1.0, 1.0], device=self.device)
        scaled_box = box_stddev * aniso * box_scale.unsqueeze(-1)
        clamped_box = scaled_box.clamp(min=1.193e-7)

        transformed = (hist_ycocg - box_center) / clamped_box
        norm = torch.linalg.vector_norm(transformed, dim=-1, keepdim=True)
        outside = (norm > 1.0).float()
        clamped_hist = (transformed / norm.clamp(min=1e-8)) * scaled_box + box_center
        history_contribution = (
            torch.maximum(luma_instability, lock_contribution) * accumulation * (1 - disocclusion)
        ).clamp(0, 1)
        rectified = torch.lerp(clamped_hist, hist_ycocg, history_contribution.unsqueeze(-1))
        hist_ycocg = outside * rectified + (1 - outside) * hist_ycocg

        # --- Accumulate (accumulate.h:23) ---
        history_weight = history_weight * (history_weight > _FP16_MIN).float()
        history_weight = (history_weight + upsampled_weight).clamp(min=_EPSILON)
        alpha = (upsampled_weight / history_weight).clamp(0, 1)
        blended = torch.lerp(hist_ycocg, upsampled, alpha.unsqueeze(-1))

        # No special case for new samples is needed here: the initial-sample
        # path above sets history_weight=0 and upsampled_weight=1, so alpha=1
        # and the blend already resolves to the upsampled colour.
        out_rgb = ycocg_to_rgb(blended).clamp(min=0.0)

        return out_rgb, FSRState(
            history_rgb=out_rgb,
            lock=lock,
            accumulation=accumulation_next,
            frame_index=state.frame_index + 1,
        )
