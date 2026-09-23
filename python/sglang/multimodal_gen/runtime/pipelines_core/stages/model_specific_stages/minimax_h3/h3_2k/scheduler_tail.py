# SPDX-License-Identifier: Apache-2.0
"""调度截断 + 流匹配部分加噪（SDEdit 核心）。

sigma 表用官方 time_request.minimax_h3_time_shift_sigmas 生成
（视频 shift=12、音频 shift=3，同点数 → 同索引天然锁定），
截断时按同一索引取尾段，锁定关系保持。
加噪公式与 latent_preparation 的噪声语义一致（CPU fp32、独立 generator）。
"""
from __future__ import annotations

import torch

from . import config


def build_full_sigmas(
    num_steps: int,
    video_shift: float = config.VIDEO_FLOW_SHIFT,
    audio_shift: float = config.AUDIO_FLOW_SHIFT,
) -> dict[str, list[float]]:
    """生成完整（从 sigma=1 到 0）的双流调度表，格式同官方 extras。"""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    return {
        "video": minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=video_shift),
        "audio": minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=audio_shift),
    }


def tail_sigmas(
    full: dict[str, list[float]],
    sigma_start: float,
) -> dict[str, list[float]]:
    """按视频表上第一个 sigma <= sigma_start 的索引 k 截断两表尾段。

    返回 {"video": full.video[k:], "audio": full.audio[k:]}，
    保证两点：起点噪声水平 ~ sigma_start；音视频索引锁定不变。
    """
    video = full["video"]
    audio = full["audio"]
    if len(video) != len(audio):
        raise ValueError("video/audio 表长度不一致")
    k = next((i for i, s in enumerate(video) if s <= sigma_start), None)
    if k is None or len(video) - k < 2:
        raise ValueError(
            f"sigma_start={sigma_start} 太小，截断后不足 2 个调度点"
            f"（video 表: {video[:3]}...{video[-3:]}）"
        )
    return {"video": video[k:], "audio": audio[k:]}


def flow_mixture(latent: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """流匹配部分加噪：x = sigma*eps + (1-sigma)*x（逐元素独立同参）。"""
    gen = torch.Generator().manual_seed(int(seed))
    eps = torch.randn(latent.shape, generator=gen, dtype=torch.float32)
    return (sigma * eps + (1.0 - sigma) * latent.to(torch.float32))


def make_pass2_noise_state(
    video_latent: torch.Tensor,
    audio_latent_native: torch.Tensor,
    tail: dict[str, list[float]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """对放大后的视频网格 latent 与音频网格 latent 做部分加噪。

    audio_latent_native: [2,32,T40]（batch.audio_latents 布局）。
    返回 (video_noised, audio_noised, sigma_v_start, sigma_a_start)。
    """
    sigma_v = float(tail["video"][0])
    sigma_a = float(tail["audio"][0])
    v = flow_mixture(video_latent, sigma_v, seed)
    a = flow_mixture(audio_latent_native, sigma_a, seed + 1)  # 音频独立流
    return v, a, sigma_v, sigma_a


def pack_audio_native(native: torch.Tensor) -> torch.Tensor:
    """[2,32,T] -> [2*T, 32] packed rows（unpack_audio_tokens 的逆）。"""
    if native.dim() != 3:
        raise ValueError(f"期望 [C=2,32,T]，got {tuple(native.shape)}")
    return native.permute(0, 2, 1).reshape(-1, native.shape[1]).contiguous()
