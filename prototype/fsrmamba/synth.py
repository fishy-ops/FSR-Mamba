"""Analytic synthetic scenes: paired training data without a GPU or a game engine.

Why this exists
---------------
Training a learned accumulator needs matched pairs: a jittered low-resolution
frame plus its motion vectors and depth, alongside a perfect high-resolution
version of the same instant. You cannot download that -- ordinary video datasets
have no motion vectors -- and capturing it from a real engine needs Windows and
a discrete GPU, which is a hard blocker on an Apple Silicon machine.

So instead we define scenes *analytically*: procedural 2D textures on layers that
move along known paths. Because the content is a closed-form function of position
and time, we can

  * sample it at any resolution, so ground truth is just heavy supersampling;
  * derive motion vectors exactly, with zero estimation error;
  * place layers at different depths, which produces real occlusion and
    disocclusion;

...all on the CPU or MPS, today.

The tradeoff is honest and worth stating: these scenes have no shading, no
specular, no transparency, and perfectly truthful motion vectors. Real engine
data has none of those luxuries, and motion vectors that *lie* are precisely the
hard case (shadows, reflections, VFX). So synthetic data validates the mechanism
-- does a learned recurrence beat hand-tuned rules when the inputs are clean? --
but it cannot validate robustness. Engine capture is still required before any
quality claim means anything.

Conventions
-----------
Motion vectors follow FSR's convention (ffx_fsr3upscaler_reproject.h:58):

    reprojected_uv = uv + motion_vector

i.e. they are *backward* vectors in UV space, pointing from where a surface is
now to where it was last frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

__all__ = [
    "Layer",
    "Scene",
    "Frame",
    "halton_jitter",
    "checkerboard",
    "zone_plate",
    "thin_lines",
    "smooth_gradient",
    "glyph_blocks",
    "default_scene",
    "game_like_scene",
    "random_scene",
]

# A texture maps layer-local pixel coordinates to linear RGB in [0, 1].
Texture = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------
# Jitter
# --------------------------------------------------------------------------


def _halton(index: int, base: int) -> float:
    """One element of the Halton low-discrepancy sequence (1-indexed)."""
    result, f = 0.0, 1.0
    while index > 0:
        f /= base
        result += f * (index % base)
        index //= base
    return result


def halton_jitter(frame_index: int, phase_count: int = 16) -> tuple[float, float]:
    """FSR-style sub-pixel jitter, in render-resolution pixels, in [-0.5, 0.5].

    FSR uses a Halton(2, 3) sequence; see ffx_fsr3upscaler.cpp's
    ffxFsr3UpscalerGetJitterOffset. `phase_count` should match the jitter
    sequence length the upscaler is configured with -- it scales with the
    upscale ratio in the real thing.
    """
    i = (frame_index % phase_count) + 1
    return _halton(i, 2) - 0.5, _halton(i, 3) - 0.5


# --------------------------------------------------------------------------
# Procedural textures
# --------------------------------------------------------------------------


def checkerboard(period: float = 16.0, colors=((0.9, 0.9, 0.9), (0.05, 0.05, 0.05))) -> Texture:
    """Hard-edged checks. Aliases viciously under motion -- the classic stress case."""
    c0 = torch.tensor(colors[0])
    c1 = torch.tensor(colors[1])

    def tex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        parity = (torch.floor(x / period) + torch.floor(y / period)) % 2.0
        sel = parity.unsqueeze(-1)
        return c0.to(x.device) * (1.0 - sel) + c1.to(x.device) * sel

    return tex


def zone_plate(scale: float = 0.0025, tint=(1.0, 0.95, 0.85)) -> Texture:
    """sin(r^2) rings: contains every spatial frequency at once.

    The standard test pattern for resampling artifacts. Frequency rises with
    radius, so a single image shows exactly where an upscaler stops resolving
    detail and starts producing moire.
    """
    t = torch.tensor(tint)

    def tex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r2 = x * x + y * y
        v = 0.5 + 0.5 * torch.sin(scale * r2)
        return v.unsqueeze(-1) * t.to(x.device)

    return tex


def thin_lines(spacing: float = 24.0, width: float = 1.0, angle_deg: float = 22.5) -> Texture:
    """Sub-pixel-width diagonal lines: the case FSR's `lock` machinery protects.

    Thin features are what a naive temporal filter erases first, so this is the
    direct test of whether a learned accumulator rediscovers the lock heuristic.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)

    def tex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = (x * ca + y * sa) % spacing
        on = (d < width).float().unsqueeze(-1)
        return on * 1.0 + (1.0 - on) * 0.08

    return tex


def smooth_gradient(period: float = 220.0) -> Texture:
    """Low-frequency colour ramp. Should be trivially easy -- a control case."""

    def tex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = 0.5 + 0.5 * torch.sin(2 * math.pi * x / period)
        v = 0.5 + 0.5 * torch.cos(2 * math.pi * y / period)
        return torch.stack((u, v, 0.6 * (u + v) * 0.5), dim=-1)

    return tex


def glyph_blocks(cell: float = 14.0, seed: int = 0) -> Texture:
    """Blocky pseudo-glyphs standing in for in-world text (signage, decals).

    Not real text -- just a deterministic high-contrast block pattern with the
    rough spatial statistics of small type. Enough to expose the failure mode
    without dragging a font rasteriser into the data pipeline.
    """

    def tex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cx = torch.floor(x / cell)
        cy = torch.floor(y / cell)
        # Cheap deterministic hash -> on/off, biased toward "ink" ~35% of cells.
        h = torch.sin(cx * 127.1 + cy * 311.7 + seed * 74.7) * 43758.5453
        on = ((h - torch.floor(h)) < 0.35).float().unsqueeze(-1)
        return on * 0.02 + (1.0 - on) * 0.95

    return tex


# --------------------------------------------------------------------------
# Scene description
# --------------------------------------------------------------------------


@dataclass
class Layer:
    """One movable textured plane.

    Attributes
    ----------
    texture:
        Content function, evaluated in layer-local pixel coordinates.
    depth:
        Distance in metres. Smaller is nearer; nearer layers occlude further
        ones. FSR consumes depth mainly for disocclusion detection.
    velocity:
        (vx, vy) in reference pixels per frame. Constant linear motion.
    extent:
        (x0, y0, x1, y1) bounding rect in reference coordinates at t=0, or None
        for an infinite backdrop. Finite extents are what create disocclusion.
    """

    texture: Texture
    depth: float
    velocity: tuple[float, float] = (0.0, 0.0)
    extent: tuple[float, float, float, float] | None = None

    def offset_at(self, t: float) -> tuple[float, float]:
        return self.velocity[0] * t, self.velocity[1] * t


@dataclass
class Frame:
    """One rendered instant.

    color: (H, W, 3) linear RGB
    depth: (H, W)    metres
    mv:    (H, W, 2) backward motion vector in UV units, FSR convention
    """

    color: torch.Tensor
    depth: torch.Tensor
    mv: torch.Tensor
    jitter: tuple[float, float]
    frame_index: int


@dataclass
class Scene:
    """A stack of layers over a reference coordinate space."""

    layers: Sequence[Layer]
    reference_size: tuple[int, int]  # (H, W)
    background: tuple[float, float, float] = (0.02, 0.02, 0.03)
    far_depth: float = 1000.0
    device: torch.device | str = "cpu"
    _sorted: list[Layer] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Nearest first, so the first hit wins during compositing.
        self._sorted = sorted(self.layers, key=lambda l: l.depth)

    # -- internals ---------------------------------------------------------

    def _sample_world(
        self, wx: torch.Tensor, wy: torch.Tensor, t: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Composite all layers at world positions (wx, wy) at time `t`.

        Returns (rgb, depth, velocity) where velocity is the reference-pixels-
        per-frame motion of whichever layer ended up visible.
        """
        dev = wx.device
        rgb = torch.tensor(self.background, device=dev).expand(*wx.shape, 3).clone()
        depth = torch.full(wx.shape, self.far_depth, device=dev)
        vel = torch.zeros((*wx.shape, 2), device=dev)
        filled = torch.zeros(wx.shape, dtype=torch.bool, device=dev)

        for layer in self._sorted:
            ox, oy = layer.offset_at(t)
            lx, ly = wx - ox, wy - oy

            if layer.extent is None:
                inside = torch.ones(wx.shape, dtype=torch.bool, device=dev)
            else:
                x0, y0, x1, y1 = layer.extent
                inside = (lx >= x0) & (lx < x1) & (ly >= y0) & (ly < y1)

            # Nearest layer wins: only write where nothing nearer has landed.
            take = inside & ~filled
            if not bool(take.any()):
                continue

            sampled = layer.texture(lx, ly)
            rgb = torch.where(take.unsqueeze(-1), sampled, rgb)
            depth = torch.where(take, torch.full_like(depth, layer.depth), depth)
            v = torch.tensor(layer.velocity, device=dev).expand(*wx.shape, 2)
            vel = torch.where(take.unsqueeze(-1), v, vel)
            filled = filled | take

        return rgb, depth, vel

    # -- public API --------------------------------------------------------

    def render(
        self,
        frame_index: int,
        size: tuple[int, int],
        jitter: tuple[float, float] = (0.0, 0.0),
        supersample: int = 1,
    ) -> Frame:
        """Render one frame.

        Parameters
        ----------
        size:
            (H, W) output resolution.
        jitter:
            Sub-pixel offset in *output* pixels. Pass (0, 0) for ground truth.
        supersample:
            NxN box supersampling. 1 reproduces a real renderer's single jittered
            sample per pixel (aliased, which is the point). Use 8-16 for ground
            truth.

        Depth and motion vectors are always taken from the pixel *centre*, never
        supersampled -- matching a real G-buffer, which stores one value per
        pixel rather than a filtered average.
        """
        h, w = size
        ref_h, ref_w = self.reference_size
        dev = torch.device(self.device)
        sx, sy = ref_w / w, ref_h / h
        t = float(frame_index)

        # --- colour, optionally supersampled ---
        n = max(1, supersample)
        js = (torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n - 0.5
        cols = torch.arange(w, device=dev, dtype=torch.float32) + 0.5 + jitter[0]
        rows = torch.arange(h, device=dev, dtype=torch.float32) + 0.5 + jitter[1]

        acc = torch.zeros((h, w, 3), device=dev)
        for dy in js:
            for dx in js:
                gx, gy = torch.meshgrid((cols + dx) * sx, (rows + dy) * sy, indexing="xy")
                acc += self._sample_world(gx, gy, t)[0]
        color = acc / (n * n)

        # --- depth and motion, from pixel centres ---
        gx, gy = torch.meshgrid(cols * sx, rows * sy, indexing="xy")
        _, depth, vel = self._sample_world(gx, gy, t)

        # Backward MV in UV: the surface here was at (p - velocity) last frame.
        mv = torch.stack(
            (-vel[..., 0] / float(ref_w), -vel[..., 1] / float(ref_h)), dim=-1
        )

        return Frame(color=color, depth=depth, mv=mv, jitter=jitter, frame_index=frame_index)

    def sequence(
        self,
        num_frames: int,
        render_size: tuple[int, int],
        output_size: tuple[int, int],
        gt_supersample: int = 8,
        jitter_phases: int = 16,
    ):
        """Yield (low_res_input, ground_truth) pairs for consecutive frames.

        The low-res frame carries the jitter; ground truth is unjittered and
        heavily supersampled at the target resolution.
        """
        for i in range(num_frames):
            jx, jy = halton_jitter(i, jitter_phases)
            lr = self.render(i, render_size, jitter=(jx, jy), supersample=1)
            gt = self.render(i, output_size, jitter=(0.0, 0.0), supersample=gt_supersample)
            yield lr, gt


def random_scene(
    seed: int, reference_size: tuple[int, int] = (1080, 1920), device="cpu"
) -> Scene:
    """A randomised game-like scene, for building disjoint train/val splits.

    Exists because holding out a *scene* is the only honest split here. Holding
    out frames from a sequence does not work: consecutive frames of the same
    content are near-duplicates, and a recurrent model carries state across them
    anyway, so a frame-wise split leaks almost everything.

    Varying velocities, layer placement, and texture scale by seed gives
    genuinely unseen content while keeping the difficulty distribution stable.
    """
    rng = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(1, generator=rng).item()

    h, w = reference_size
    return Scene(
        layers=[
            Layer(smooth_gradient(period=w / u(2.5, 3.5)), depth=100.0,
                  velocity=(u(-0.5, 0.5), u(-0.1, 0.1))),
            Layer(
                checkerboard(period=w / u(18.0, 30.0),
                             colors=((u(0.6, 0.8), u(0.6, 0.75), u(0.5, 0.7)),
                                     (u(0.15, 0.25), u(0.15, 0.28), u(0.2, 0.3)))),
                depth=45.0,
                velocity=(u(-0.9, -0.3), u(-0.1, 0.1)),
                extent=(-0.1 * w, u(0.5, 0.62) * h, 1.4 * w, 1.2 * h),
            ),
            Layer(
                glyph_blocks(cell=w / u(38.0, 55.0), seed=seed),
                depth=14.0,
                velocity=(u(0.15, 0.55), u(0.05, 0.25)),
                extent=(u(0.06, 0.16) * w, u(0.12, 0.24) * h,
                        u(0.40, 0.50) * w, u(0.36, 0.46) * h),
            ),
            Layer(
                thin_lines(spacing=w / u(24.0, 38.0), width=w / u(350.0, 480.0),
                           angle_deg=u(10.0, 55.0)),
                depth=9.0,
                velocity=(u(-1.0, -0.5), u(0.1, 0.35)),
                extent=(u(0.54, 0.62) * w, u(0.48, 0.56) * h,
                        u(0.86, 0.94) * w, u(0.82, 0.90) * h),
            ),
        ],
        reference_size=reference_size,
        device=device,
    )


def game_like_scene(reference_size: tuple[int, int] = (1080, 1920), device="cpu") -> Scene:
    """A milder scene whose content mostly sits *below* Nyquist.

    `default_scene` is a torture test: its zone-plate backdrop is above Nyquist
    across most of the frame, where the correct answer is flat grey. That is
    useful for finding artifacts but useless as a benchmark, because it rewards
    whichever method blurs hardest and compresses everyone into the same low
    PSNR band.

    This scene is the one to quote numbers from. Content is recoverable, so PSNR
    differences reflect reconstruction quality rather than who blurred most.
    Motion is slower and more camera-like, which is closer to real gameplay than
    objects rocketing across the screen.
    """
    h, w = reference_size
    return Scene(
        layers=[
            Layer(smooth_gradient(period=w / 3.0), depth=100.0, velocity=(-0.35, 0.0)),
            Layer(
                checkerboard(period=w / 24.0, colors=((0.72, 0.68, 0.60), (0.20, 0.22, 0.26))),
                depth=45.0,
                velocity=(-0.6, 0.0),
                extent=(-0.1 * w, 0.55 * h, 1.4 * w, 1.2 * h),
            ),
            Layer(
                glyph_blocks(cell=w / 45.0),
                depth=14.0,
                velocity=(0.35, 0.12),
                extent=(0.10 * w, 0.18 * h, 0.44 * w, 0.40 * h),
            ),
            Layer(
                thin_lines(spacing=w / 30.0, width=w / 400.0),
                depth=9.0,
                velocity=(-0.75, 0.2),
                extent=(0.58 * w, 0.52 * h, 0.90 * w, 0.86 * h),
            ),
        ],
        reference_size=reference_size,
        device=device,
    )


def default_scene(reference_size: tuple[int, int] = (1080, 1920), device="cpu") -> Scene:
    """A scene exercising the four failure modes we care about.

    Backdrop of pure aliasing (zone plate), a panning checkerboard, a thin-line
    card, and a block of pseudo-text -- the last two moving at different speeds
    and depths so they occlude and disocclude each other.
    """
    h, w = reference_size
    return Scene(
        layers=[
            Layer(zone_plate(), depth=100.0, velocity=(0.0, 0.0)),
            Layer(
                checkerboard(period=w / 60.0),
                depth=40.0,
                velocity=(-1.7, 0.0),
                # Finite, or it would blanket the zone plate behind it.
                extent=(-0.1 * w, 0.45 * h, 1.4 * w, 1.1 * h),
            ),
            Layer(
                glyph_blocks(cell=w / 140.0),
                depth=12.0,
                velocity=(0.9, 0.45),
                extent=(0.12 * w, 0.15 * h, 0.48 * w, 0.42 * h),
            ),
            Layer(
                thin_lines(spacing=w / 80.0, width=w / 1600.0),
                depth=8.0,
                velocity=(-2.4, 0.6),
                extent=(0.55 * w, 0.5 * h, 0.92 * w, 0.88 * h),
            ),
        ],
        reference_size=reference_size,
        device=device,
    )
