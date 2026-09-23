# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 大画幅（2K）二次采样：单 API 自动两遍扩展。

对外只有现有 POST /v1/videos 一个接口：
- target.short_edge <= 768 → 原路径，零影响
- target.short_edge  > 768 且携带可选 "h3_2k" 配置 → 自动两遍：
  pass1(768p) -> latent 放大 -> sigma 截断部分加噪 -> pass2(大画幅)

请求示例（"h3_2k" 为可选字段，缺省用默认值）：
    {"task":"t2va","prompt":"...","target":{"short_edge":1536,...},
     "h3_2k":{"upscale":"network","scale":2.0,"sigma_start":0.90,
              "pass2_steps":6,"freeze_audio":true}}

模块布局（全部只新增文件；原 sglang 文件仅在 minimax_h3_pipeline.py 与
video_api.py 各加一处 try-import 挂载，容错不破坏原功能）：
    config           阈值/路径/方法注册表
    canvas_patch     画布上限放宽
    latent_hooks     捕获/音频冻结/pass2 行替换
    upscale_interp   插值放大（32 像素对齐）
    upscale_network  学习型 3D 网络放大（LBH 移植）
    scheduler_tail   sigma 截断 + 流匹配加噪
    tiling           分片算法（瓦片网格/蒙版/色校）
    orchestrator     worker 侧两遍执行体（wrap MiniMaxH3Pipeline.forward）
    api_hook         HTTP 侧大画幅标记提取 + pass1 改写（wrap video_api）
"""
from __future__ import annotations

import logging

# 共享 logger：继承服务 root handler（时间戳/级别/颜色），输出进 sglang 日志
logger = logging.getLogger("sglang.h3_2k")

# extras key：HTTP -> worker 的两遍计划（api_hook 塞入）
H3_2K_PLAN_KEY = "h3_2k_plan"
# extras key：HTTP -> worker 的 pass2 初始行替换载荷（api_hook 塞入）
H3_2K_PASS2_KEY = "h3_2k_pass2_state"
# extras key：decode-only 请求（tiling 融合后的最终解码，跳过 DenoisingStage）
H3_2K_DECODE_ONLY_KEY = "h3_2k_decode_only"

_installed_worker = False
_installed_api = False


def install_worker_hooks() -> None:
    """在 scheduler/worker 进程安装两遍编排（幂等）。挂载点：minimax_h3_pipeline.py。"""
    global _installed_worker
    if _installed_worker:
        return
    from . import worker_hooks

    worker_hooks.install()
    _installed_worker = True


def install_api_hooks() -> None:
    """在 HTTP 进程安装请求改写（幂等）。挂载点：openai/video_api.py。"""
    global _installed_api
    if _installed_api:
        return
    from . import api_hook

    api_hook.install()
    _installed_api = True
