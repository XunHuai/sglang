# SPDX-License-Identifier: Apache-2.0
"""h3_2k 配置：阈值、默认参数、放大方法注册表、路径。

全部支持环境变量覆盖，便于不改代码调整。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 画布阈值
# 官方 768p 基线：短边 768、面积上限 768*1344（resolved_plan.py 的原值）。
NORMAL_SHORT_EDGE = 768
NORMAL_MAX_PIXELS = NORMAL_SHORT_EDGE * 1344

# 大画幅上限：1536*2720 ≈ 4.16MP，覆盖 1536 短边 16:9（1536x2720 已对齐 32）。
LARGE_MAX_PIXELS = int(
    os.environ.get("H3_2K_MAX_PIXELS", str(1536 * 2720))
)

# 判定"大画幅请求"的规则：target.short_edge > NORMAL_SHORT_EDGE 即走两遍编排。
def is_large_target(target: dict) -> bool:
    short_edge = int((target or {}).get("short_edge", NORMAL_SHORT_EDGE))
    return short_edge > NORMAL_SHORT_EDGE

# ---------------------------------------------------------------- 采样默认值
# sigma 起点：视频调度表上截断的起始噪声水平。
# 注意 shift=12 的表前半集中在高噪声区，0.90 约等价 ComfyUI denoise≈0.5。
SIGMA_START_DEFAULT = float(os.environ.get("H3_2K_SIGMA_START", "0.90"))
# 第二遍步数（含终点 0 的调度点数 = 步数+1，与官方约定一致）
PASS2_STEPS_DEFAULT = int(os.environ.get("H3_2K_PASS2_STEPS", "6"))
PASS1_STEPS_DEFAULT = int(os.environ.get("H3_2K_PASS1_STEPS", "8"))
# H3 官方双流 shift
VIDEO_FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0

# ---------------------------------------------------------------- 分片默认值
# 瓦片边长像素（另一边相同）与重叠比例；瓦片按 latent 偶数网格对齐。
TILE_PX_DEFAULT = int(os.environ.get("H3_2K_TILE_PX", "512"))
TILE_OVERLAP_DEFAULT = float(os.environ.get("H3_2K_TILE_OVERLAP", "0.25"))
# tiling 模式 sigma_start 上限：瓦片是独立请求（RoPE 按 512x512 1:1 重新
# 归一化、参考条件按瓦片重编码），重绘幅度过高时各瓦片按自己的画布世界
# 观重新构图，画面会被切成多个面。钳制后瓦片退化为低幅度细节精修。
TILING_SIGMA_START_MAX = float(
    os.environ.get("H3_2K_TILING_SIGMA_MAX", "0.90")
)
# 缝区重采样上限（原插件 seam_denoise 推荐 0.5~0.8）：重叠带渐变段的
# 最终自由度 = seam_cap，值越低缝区越保守
SEAM_CAP_DEFAULT = float(os.environ.get("H3_2K_SEAM_CAP", "0.6"))
# probe 门控二道缝修补（corr<0.85 或 |dc|>0.03 触发）
SEAM_POLISH_DEFAULT = os.environ.get("H3_2K_SEAM_POLISH", "1") == "1"
# 修补窗口半宽（latent 单位）
POLISH_BAND_LATENT = int(os.environ.get("H3_2K_POLISH_BAND", "8"))
# 时间分块（默认关：5s/124f 视频 2 块成本翻倍；长视频再开）
TIME_TILE_DEFAULT = os.environ.get("H3_2K_TIME_TILE", "0") == "1"
# identity 锚：每 stride 个 token 钉一整帧（防内容漂移）
IDENTITY_STRIDE_DEFAULT = int(os.environ.get("H3_2K_IDENTITY_STRIDE", "7"))

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(os.environ.get("H3_2K_BASE_DIR", "/home/liaoruihao/Github/MiniMax-H3"))
MODEL_PATH = Path(os.environ.get("H3_2K_MODEL_PATH", str(BASE_DIR)))
TRANSFORMER_PATH = Path(
    os.environ.get(
        "H3_2K_TRANSFORMER_PATH",
        "/home/liaoruihao/Github/MiniMax-H3-quant/diffusion_models/"
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    )
)
LORA_PATH = Path(
    os.environ.get(
        "H3_2K_LORA_PATH",
        "/home/liaoruihao/Github/MiniMax-H3-Turbo/"
        "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    )
)
# 学习型放大网络权重目录（LBH-123-AI/Minimax_h3_latent_Upscaler 下载放置处）
UPSCALER_WEIGHTS_DIR = Path(
    os.environ.get("H3_2K_UPSCALER_WEIGHTS", str(BASE_DIR / "models/latent_upscale_models"))
)
# 中间产物（latent 快照）目录
WORK_DIR = Path(os.environ.get("H3_2K_WORK_DIR", str(BASE_DIR / "h3_2k_work")))

# ---------------------------------------------------------------- 放大方法注册表
# name -> (module, factory)。延迟导入，用到才加载（网络放大依赖 torch/safetensors）。
def get_upscaler(name: str):
    """按名取放大函数：callable(latent5d, scale) -> latent5d'。"""
    if name == "interp":
        from . import upscale_interp

        return upscale_interp.upscale_video_latent
    if name == "network":
        from . import upscale_network

        return upscale_network.load_upscale_fn()
    raise ValueError(f"未知的放大方法 {name!r}，可选: interp / network")

UPSCALER_NAMES = ("interp", "network")
