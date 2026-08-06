"""Windowed (Swin-style) self-attention for the spatial reconstruction path.

Why attention here, and why *windowed*
--------------------------------------
Every convolutional/tuning lever plateaued at ~0.9435 full-frame SSIM, and the
per-scene diagnostic put the deficit in **fine, repetitive, high-frequency
texture** (chess marble/checkerboard: 0.876 vs FSR's 0.921). That is precisely
the pattern self-attention is built for: a token can look at *other* tokens that
share its structure and reconstruct itself consistently with them, which a local
3x3 kernel cannot do. It is also what DLSS 4 moved to.

Global attention is impossible at output resolution -- 1920x1080 is ~2M tokens and
attention is quadratic. Swin's answer, used here: attend only inside small windows,
then **shift** the window grid on alternating blocks so information still crosses
window borders. Cost becomes linear in pixels.

Faithful to Swin: relative position bias per head, and the shifted-window attention
mask so rolled-in edge tokens don't attend across the wrap seam.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwinRefiner"]


def window_partition(x: torch.Tensor, w: int) -> torch.Tensor:
    """(B,H,W,C) -> (B*num_windows, w*w, C)"""
    b, h, wd, c = x.shape
    x = x.view(b, h // w, w, wd // w, w, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, w * w, c)


def window_reverse(windows: torch.Tensor, w: int, h: int, wd: int) -> torch.Tensor:
    """(B*num_windows, w*w, C) -> (B,H,W,C)"""
    b = int(windows.shape[0] / (h * wd / w / w))
    x = windows.view(b, h // w, wd // w, w, w, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, wd, -1)


class WindowAttentionBlock(nn.Module):
    """One Swin block: windowed MHSA + MLP, both residual, pre-norm."""

    def __init__(self, dim: int, heads: int, window: int, shift: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim, self.heads, self.window, self.shift = dim, heads, window, shift
        self.scale = (dim // heads) ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # Relative position bias: lets a head prefer particular spatial offsets,
        # which is what makes windowed attention good at regular texture.
        self.rel_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)  # 2,N
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()  # N,N,2
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)  # N,N
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _attn_mask(self, h: int, w: int, device) -> torch.Tensor | None:
        """Shifted-window mask: after the roll, tokens from opposite image edges
        share a window; they must not attend to each other."""
        if self.shift == 0:
            return None
        key = (h, w)
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        img = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        spans = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
        for hs in spans:
            for ws in spans:
                img[:, hs, ws, :] = cnt
                cnt += 1
        mw = window_partition(img, self.window).view(-1, self.window * self.window)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, 0.0)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,H,W,C)
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))

        win = window_partition(x, self.window)                     # nW, N, C
        nw, n, _ = win.shape
        qkv = self.qkv(win).view(nw, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # nW, heads, N, hd
        attn = (q * self.scale) @ k.transpose(-2, -1)               # nW, heads, N, N
        attn = attn + self.rel_bias[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        mask = self._attn_mask(h, w, x.device)
        if mask is not None:
            attn = (attn.view(-1, mask.shape[0], self.heads, n, n) + mask[None, :, None]).view(
                -1, self.heads, n, n)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(nw, n, c)
        out = self.proj(out)

        x = window_reverse(out, self.window, h, w)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SwinRefiner(nn.Module):
    """Attention refiner on the spatial resolve.

    Takes the Lanczos resolve plus the neighbourhood colour box (centre + stddev),
    which tell it the local contrast it is allowed to reconstruct, and predicts a
    residual. The output projection is zero-initialised, so the module starts as an
    exact identity: training can only move it away from plain Lanczos if that helps,
    which matters a lot when it sits inside a recurrent accumulation.
    """

    def __init__(self, in_ch: int = 9, dim: int = 48, heads: int = 3,
                 window: int = 8, depth: int = 2):
        super().__init__()
        self.window = window
        self.embed = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            WindowAttentionBlock(dim, heads, window, shift=0 if i % 2 == 0 else window // 2)
            for i in range(depth)
        ])
        self.out = nn.Conv2d(dim, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (1,in_ch,H,W) -> (1,3,H,W)
        x = self.embed(feats)
        _, _, h0, w0 = x.shape
        ph = (self.window - h0 % self.window) % self.window
        pw = (self.window - w0 % self.window) % self.window
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x = x.permute(0, 2, 3, 1)                    # NHWC for attention
        for blk in self.blocks:
            # Gradient checkpointing: the per-window attention maps
            # (num_windows x heads x N x N) dominate memory and would push an 8 GB
            # card into system-RAM spill at 512x512 with BPTT. Recomputing them in
            # the backward pass trades ~30% compute for several GB.
            if self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = x.permute(0, 3, 1, 2)                    # back to NCHW
        if ph or pw:
            x = x[:, :, :h0, :w0]
        return self.out(x)
