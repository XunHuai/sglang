# SPDX-License-Identifier: Apache-2.0
"""插值放大：对 H3 视频 latent（全网格 [1,24,T,H,W]）做空间插值。

对齐规则移植自 ComfyUI-H3LatentUpscale-jingchen573：
- VAE 压缩 16x、DiT patch 2x2 → 像素宽高须被 32 整除 → latent 宽高须为偶数
- 短边向下取偶，实际倍率不超设定；长边取最接近等比缩放的合法偶数
- 时间维与通道不参与插值（逐帧独立空间插值）

插值核建议 nearest（映射 torch "nearest-exact" 语义用 mode="nearest"，
配合 align 无关；不发明新值，latent 统计分布保持最好）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# latent 全网格的合法步进（= patch 2x2）
_LATENT_ALIGN = 2
# 与 ComfyUI 节点同名方法的映射
_METHOD_MAP = {
    "nearest-exact": "nearest",
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
}


def aligned_output_size(h: int, w: int, scale: float) -> tuple[int, int]:
    """jingchen573 对齐算法：返回 (h_out, w_out)，均为偶数且倍率不超上限。"""
    if h <= 0 or w <= 0:
        raise ValueError(f"latent 尺寸必须为正, got {h}x{w}")
    long_in, short_in = (w, h) if w >= h else (h, w)
    short_out = max(_LATENT_ALIGN, int(short_in * scale) // _LATENT_ALIGN * _LATENT_ALIGN)
    short_eff = short_out / short_in
    ideal_long = long_in * short_eff
    long_cap = max(_LATENT_ALIGN, int(long_in * scale) // _LATENT_ALIGN * _LATENT_ALIGN)
    lower = max(_LATENT_ALIGN, int(ideal_long) // _LATENT_ALIGN * _LATENT_ALIGN)
    candidates = {c for c in (lower, lower + _LATENT_ALIGN, long_cap)
                  if _LATENT_ALIGN <= c <= long_cap}
    long_out = min(candidates, key=lambda c: (abs(c - ideal_long), c))
    return (short_out, long_out) if h <= w else (long_out, short_out)


def upscale_video_latent(
    latent: torch.Tensor,
    scale: float,
    method: str = "nearest-exact",
    target_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict]:
    """放大 [1,24,T,H,W] 的空间维。返回 (新 latent, 信息 dict)。

    只放大 H/W；T、通道原样。target_hw 优先（对齐到指定 latent 尺寸，
    如 pass2 画布反推值）；否则按 scale 经 32 像素（latent 偶数）对齐算法。
    """
    if latent.dim() != 5 or latent.shape[0] != 1:
        raise ValueError(f"期望 [1,24,T,H,W]，got {tuple(latent.shape)}")
    mode = _METHOD_MAP.get(method)
    if mode is None:
        raise ValueError(f"未知插值方法 {method!r}，可选 {sorted(_METHOD_MAP)}")
    _, c, t, h, w = (int(x) for x in latent.shape)
    if target_hw is not None:
        h_out, w_out = int(target_hw[0]), int(target_hw[1])
    else:
        h_out, w_out = aligned_output_size(h, w, scale)

    x = latent.to(torch.float32).permute(0, 2, 1, 3, 4).reshape(t, c, h, w)
    y = F.interpolate(x, size=(h_out, w_out), mode=mode)
    out = y.reshape(1, t, c, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
    info = {
        "in_latent_hw": (h, w), "out_latent_hw": (h_out, w_out),
        "in_pixels": (h * 16, w * 16), "out_pixels": (h_out * 16, w_out * 16),
        "effective_scale": (w_out / w, h_out / h), "method": method,
    }
    return out.to(latent.dtype), info
