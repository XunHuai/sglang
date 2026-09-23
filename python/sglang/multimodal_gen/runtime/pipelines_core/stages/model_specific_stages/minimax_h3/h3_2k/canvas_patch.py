# SPDX-License-Identifier: Apache-2.0
"""画布上限放宽：运行时替换 resolved_plan.MAX_PIXELS（不改库文件）。

resolved_plan 的面积 clamp 直接引用模块级 MINIMAX_H3_MAX_PIXELS，
Python 模块属性是动态查找，运行时替换即生效。
"""
from __future__ import annotations

from . import config

_ORIGINAL = None


def raise_canvas_cap(max_pixels: int | None = None) -> None:
    """把 H3 画布面积上限放宽到 config.LARGE_MAX_PIXELS（或指定值）。"""
    global _ORIGINAL
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        resolved_plan,
    )

    if _ORIGINAL is None:
        _ORIGINAL = resolved_plan.MINIMAX_H3_MAX_PIXELS
    resolved_plan.MINIMAX_H3_MAX_PIXELS = int(
        max_pixels if max_pixels is not None else config.LARGE_MAX_PIXELS
    )


def restore_canvas_cap() -> None:
    """恢复官方上限（幂等）。"""
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        resolved_plan,
    )

    resolved_plan.MINIMAX_H3_MAX_PIXELS = _ORIGINAL
    _ORIGINAL = None


class canvas_cap:
    """上下文管理器：with canvas_cap(): ... 期间放宽，退出恢复。"""

    def __init__(self, max_pixels: int | None = None):
        self._max_pixels = max_pixels

    def __enter__(self):
        raise_canvas_cap(self._max_pixels)
        return self

    def __exit__(self, *exc):
        restore_canvas_cap()
        return False
