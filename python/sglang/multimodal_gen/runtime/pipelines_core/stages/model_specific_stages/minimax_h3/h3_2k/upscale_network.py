# SPDX-License-Identifier: Apache-2.0
"""学习型网络放大：LBH-123-AI Minimax_h3_latent_Upscaler 的 3D 网络移植。

HF 仓库实际发布的权重为 3D 版（minimax_h3_latent_upscaler_3d_fp16.safetensors，
ComfyUI 官方工作流模板同样使用 3D 版），本模块结构与其逐键一致（strict 加载）。

架构（LatentResizer3D，宽度 512）：
  conv_in(24->512, 3x3x3) -> 12x ResBlockEmb3D（每 2 块后串 TemporalConv）
  -> trilinear 上采样到目标 (t,h,w)   ← 时间维传原 t，等效不变
  -> 12x ResBlockEmb3D（同样穿插 TemporalConv）
  -> GroupNorm/SiLU/conv_out(512->24)
  scale-1 经 MLP 得 FiLM 嵌入注入每个残差块 → 单网络支持 1.0~4.0 任意倍率。

权重下载（约 658MB）：
  https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
  https://hf-mirror.com/LBH-123-AI/Minimax_h3_latent_Upscaler
放到 config.UPSCALER_WEIGHTS_DIR（默认 MiniMax-H3/models/latent_upscale_models/）。
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upscale_interp import aligned_output_size

# H3 24 通道 latent 逐通道归一化统计（原仓库训练常数）
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946881835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * channels))
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=0.1), nn.Conv3d(channels, channels, 3, padding=1))
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.dim() < h.dim():
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return x + self.out_layers(h)


class _TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(32, channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel_size, 1, 1),
                                padding=(pad, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    """与原仓库 LatentResizer3D 逐键一致（strict=True 可加载）。"""

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(),
                                   nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(_ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(_ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv(channels, temporal_kernel))
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, scale: float, target_thw) -> torch.Tensor:
        emb = self.embed(torch.tensor([scale - 1.0], dtype=x.dtype, device=x.device)[None])
        emb = emb.expand(x.shape[0], -1)
        z = self.conv_in(x)
        for blk in self.in_blocks:
            z = blk(z, emb) if isinstance(blk, _ResBlockEmb3D) else blk(z)
        z = F.interpolate(z, size=target_thw, mode="trilinear", align_corners=False)
        for blk in self.out_blocks:
            z = blk(z, emb) if isinstance(blk, _ResBlockEmb3D) else blk(z)
        return self.conv_out(F.silu(self.norm_out(z)))


def _load_state(path) -> dict:
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(str(path), device="cpu")
    else:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
            for k, v in sd.items()}


def _detect(sd: dict) -> dict:
    """从 state_dict 反推结构（对齐原仓库 _detect_arch）。"""
    import re

    cfg = {"in_channels": 24, "channels": 512, "temporal_every": 2, "temporal_kernel": 5}
    if "conv_in.weight" in sd:
        cfg["in_channels"] = sd["conv_in.weight"].shape[1]
        cfg["channels"] = sd["conv_in.weight"].shape[0]
    in_ids = {int(m.group(1)) for k in sd
              if (m := re.match(r"in_blocks\.(\d+)\.in_layers\.", k))}
    out_ids = {int(m.group(1)) for k in sd
               if (m := re.match(r"out_blocks\.(\d+)\.in_layers\.", k))}
    cfg["in_blocks"] = len(in_ids) or 12
    cfg["out_blocks"] = len(out_ids) or 12
    has_temporal = any(".dwconv.weight" in k for k in sd)
    cfg["temporal_every"] = 2 if has_temporal else 0
    for k in sd:
        if k.endswith(".dwconv.weight"):
            cfg["temporal_kernel"] = sd[k].shape[2]
            break
    return cfg


@lru_cache(maxsize=1)
def _cached_model(weights_path: str, device: str, dtype: str):
    sd = _load_state(weights_path)
    cfg = _detect(sd)
    model = LatentResizer3D(**{k: v for k, v in cfg.items() if k in
                               LatentResizer3D.__init__.__code__.co_varnames})
    model.load_state_dict(sd, strict=True)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    model = model.to(device=device, dtype=dt).eval().requires_grad_(False)
    return model, dt


def release_cached_model() -> None:
    """释放放大网络的 GPU 常驻权重，给后续 pass2 降噪让出显存。"""
    import gc

    _cached_model.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_upscale_fn(weights_name: str | None = None, device: str = "cuda",
                    dtype: str = "fp16"):
    """返回 callable(latent5d, scale) -> (latent5d', info)。

    weights_name 不给时自动扫描 config.UPSCALER_WEIGHTS_DIR 下第一个
    .safetensors/.pth。输入输出为 [1,24,T,H,W] 全网格 latent。
    """
    from . import config

    weights_dir = config.UPSCALER_WEIGHTS_DIR
    if weights_name is None:
        candidates = sorted(list(weights_dir.glob("*.safetensors")) + list(weights_dir.glob("*.pth")))
        if not candidates:
            raise FileNotFoundError(
                f"未找到放大网络权重，请从 https://hf-mirror.com/LBH-123-AI/"
                f"Minimax_h3_latent_Upscaler 下载 minimax_h3_latent_upscaler_3d_fp16.safetensors "
                f"后放入 {weights_dir}"
            )
        weights_name = candidates[0].name
    path = weights_dir / weights_name
    model, dt = _cached_model(str(path), device, dtype)
    mean = torch.tensor(LATENTS_MEAN, dtype=dt, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dt, device=device).view(1, -1, 1, 1, 1)

    def _upscale(latent: torch.Tensor, scale: float, target_hw=None):
        _, c, t, h, w = (int(x) for x in latent.shape)
        if target_hw is not None:
            h_out, w_out = int(target_hw[0]), int(target_hw[1])
            emb_scale = (w_out / w + h_out / h) / 2.0  # FiLM 用实际等效倍率
        else:
            h_out, w_out = aligned_output_size(h, w, scale)
            emb_scale = scale
        x = latent.to(device=device, dtype=dt)
        x = (x - mean) / std
        with torch.inference_mode():
            y = model(x, scale=emb_scale, target_thw=(t, h_out, w_out))
        y = y * std + mean
        info = {
            "in_latent_hw": (h, w), "out_latent_hw": (h_out, w_out),
            "in_pixels": (h * 16, w * 16), "out_pixels": (h_out * 16, w_out * 16),
            "effective_scale": (w_out / w, h_out / h), "method": f"network:{weights_name}",
        }
        return y.to("cpu", latent.dtype), info

    return _upscale
