# SPDX-License-Identifier: Apache-2.0
"""worker 侧常驻钩子（每个 rank 进程独立安装，幂等）：

1. wrap MiniMaxH3Pipeline.forward
   请求带 h3_2k_plan 标记（大画幅两遍的 pass1）时捕获最终 latent，
   把 {video/audio rows, 网格, plan} 挂到 OutputBatch.h3_2k_state 随
   pickle 回传 HTTP 进程（rank0 的回传被使用，rank1 的被丢弃）。
   普通请求零改动、零开销。

2. wrap MiniMaxH3LatentPreparationStage.forward（常驻）
   请求 extra 带 h3_2k_pass2_state 时替换 target 初始行（条件行保留
   管线按大画布重新编码的结果）；若带冻结音频行则同时登记冻结。

3. wrap denoise_loop（常驻）
   冻结登记存在时，每步 Euler 更新后把音频行钉回冻结值。
"""
from __future__ import annotations

import torch

from . import H3_2K_DECODE_ONLY_KEY, H3_2K_PASS2_KEY, H3_2K_PLAN_KEY, logger

# 模块级冻结登记（LatentPreparation 写入，denoise_loop 读取；同进程内）
# "rows": 音频整段钉回；"video": 视频行锚（noise_mask 等效）
_FREEZE: dict = {"rows": None, "video": None}


def install() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        denoise_loop as denoise_loop_mod,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
        MiniMaxH3DecodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
        MiniMaxH3LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
        MiniMaxH3Pipeline,
    )

    # ---------- 1) pipeline.forward：pass1 捕获 + pass2 任务级 LoRA ----------
    _orig_forward = MiniMaxH3Pipeline.forward

    def _forward(self, batch, server_args):
        payload2 = (batch.extra or {}).get(H3_2K_PASS2_KEY)
        lora_override = (payload2 or {}).get("lora_strength")
        if lora_override is None:
            return _forward_nolora_override(self, batch, server_args)
        # 任务级 LoRA：本 forward 内临时切换，结束恢复（scheduler 串行
        # forward，其它请求的 forward 看到的始终是服务级原状态）
        nickname = getattr(server_args, "lora_nickname", None)
        restore_strength = float(getattr(server_args, "lora_scale", 1.0) or 1.0)
        if not nickname:
            logger.info("[h3_2k] pass2_lora=off 但服务未挂 LoRA，忽略")
            return _forward_nolora_override(self, batch, server_args)
        logger.info(f"[h3_2k] pass2 LoRA 任务级切换: {nickname} "
              f"strength {restore_strength} -> {lora_override}")
        self.set_lora(nickname, None, "all", strength=float(lora_override))
        try:
            return _forward_nolora_override(self, batch, server_args)
        finally:
            self.set_lora(nickname, None, "all", strength=restore_strength)
            logger.info(f"[h3_2k] pass2 LoRA 已恢复 strength={restore_strength}")

    def _forward_nolora_override(self, batch, server_args):
        plan = (batch.extra or {}).get(H3_2K_PLAN_KEY)
        if plan is None:
            return _orig_forward(self, batch, server_args)
        captured: dict = {}
        p_capture = _install_capture(captured)
        try:
            ob = _orig_forward(self, batch, server_args)
        finally:
            _restore(p_capture)
        captured["plan"] = plan
        captured["server_args"] = server_args
        captured["seed"] = int(getattr(batch.sampling_params, "seed", 42) or 42)
        try:
            ob.h3_2k_state = captured
        except Exception:
            pass
        return ob

    MiniMaxH3Pipeline.forward = _forward

    # ---------- 2) LatentPreparation：pass2 行替换 + 冻结登记 ----------
    _orig_prep = MiniMaxH3LatentPreparationStage.forward

    def _prep(self, batch, server_args):
        result = _orig_prep(self, batch, server_args)
        payload = (batch.extra or {}).get(H3_2K_PASS2_KEY)
        if payload is not None:
            _apply_pass2_rows(batch, payload)
            frozen = payload.get("freeze_audio_rows")
            _FREEZE["rows"] = frozen if frozen is not None else None
            # 视频行锚（noise_mask 等效）：idx 为 target 段内局部索引，
            # 这里偏移为全局行索引（跳过条件/参考前缀）
            va = payload.get("video_anchor")
            if va is not None and va.get("idx") is not None:
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
                    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
                )

                st = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
                target_n = (int(st["latent_t"]) * (int(st["latent_h"]) // 2)
                            * (int(st["latent_w"]) // 2))
                prefix = int(st["initial_video_rows"].shape[0]) - target_n
                _FREEZE["video"] = {
                    "idx": va["idx"].to(torch.long) + prefix,
                    "val": va["val"].to(torch.float32),
                    "alpha": va["alpha"].to(torch.float32),
                }
            else:
                _FREEZE["video"] = None
        return result

    MiniMaxH3LatentPreparationStage.forward = _prep

    # ---------- 4) DenoisingStage：decode-only 跳过 ----------
    # tiling 融合后的最终解码请求带标记：不跑模型（省全幅 attention 显存
    # 峰值），batch.latents 已是注入的干净行，直接进 DecodingStage
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        MiniMaxH3DenoisingStage,
    )

    _orig_denoise = MiniMaxH3DenoisingStage.forward

    def _denoise(self, batch, server_args):
        if (batch.extra or {}).get(H3_2K_DECODE_ONLY_KEY):
            logger.info("[h3_2k] decode-only: 跳过 DenoisingStage，直接解码")
            # 复刻官方出口转换（denoising.py _publish_full_loop_outputs）：
            # 阶段输出契约要求 5 维 latents / 3 维 audio_latents，
            # 注入的打包行必须先 unpatchify，否则输出校验不过
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
                MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
            )
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
                minimax_h3_unpatchify_video_tokens,
                minimax_h3_unpack_audio_tokens,
            )

            state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
            lt, lh, lw = (int(state[k]) for k in ("latent_t", "latent_h", "latent_w"))
            audio_t = int(state["audio_t"])
            target_n = lt * (lh // 2) * (lw // 2)
            # 必须搬上 CUDA：官方注释明确 decode autocast 只对 CUDA 输入
            # 生效，CPU 张量会在 VAE conv3d 处触发 slow_conv3d 后端错误
            rows_v = batch.latents[batch.latents.shape[0] - target_n:].cuda()
            batch.latents = minimax_h3_unpatchify_video_tokens(
                rows_v,
                latent_shape=[lt, lh // 2, lw // 2, 24],
                patch_size=[1, 2, 2],
            )
            batch.audio_latents = minimax_h3_unpack_audio_tokens(
                batch.audio_latents.cuda(), audio_t=audio_t * 2, audio_channel=2)
            return batch
        return _orig_denoise(self, batch, server_args)

    MiniMaxH3DenoisingStage.forward = _denoise

    # ---------- 3) denoise_loop：音频钉回 + 视频行锚 ----------
    _orig_loop = denoise_loop_mod.minimax_h3_denoise_loop

    def _loop(**kwargs):
        frozen_rows = _FREEZE["rows"]
        video_anchor = _FREEZE.get("video")
        if frozen_rows is None and video_anchor is None:
            return _orig_loop(**kwargs)
        device = kwargs.get("device")
        frozen_dev = (frozen_rows.to(device=device, dtype=torch.float32)
                      if frozen_rows is not None else None)
        va_dev = None
        if video_anchor is not None:
            va_dev = {
                "idx": video_anchor["idx"].to(device),
                "val": video_anchor["val"].to(device=device, dtype=torch.float32),
                "alpha": video_anchor["alpha"].to(
                    device=device, dtype=torch.float32).view(-1, 1),
            }
        user_on_step = kwargs.get("on_step")

        def _on_step(step, video_rows, audio_rows):
            with torch.no_grad():
                if frozen_dev is not None:
                    audio_rows.copy_(frozen_dev)
                if va_dev is not None:
                    # alpha 混合：冻结段 alpha=1 完全钉回初始值，
                    # 渐变段部分混合（等效 noise_mask 的连续 mask）
                    cur = video_rows[va_dev["idx"]].to(torch.float32)
                    video_rows[va_dev["idx"]] = (
                        va_dev["alpha"] * va_dev["val"]
                        + (1.0 - va_dev["alpha"]) * cur)
            if user_on_step is not None:
                user_on_step(step, video_rows, audio_rows)

        kwargs["on_step"] = _on_step
        try:
            return _orig_loop(**kwargs)
        finally:
            _FREEZE["rows"] = None
            _FREEZE["video"] = None

    denoise_loop_mod.minimax_h3_denoise_loop = _loop


# ---------------------------------------------------------------- 捕获实现
def _install_capture(out: dict) -> list:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        denoise_loop as denoise_loop_mod,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
        MiniMaxH3DecodingStage,
    )

    patches = []

    _loop = denoise_loop_mod.minimax_h3_denoise_loop

    def _loop_spy(**kwargs):
        v_rows, a_rows = _loop(**kwargs)
        out["video_rows_packed"] = v_rows.detach().to("cpu", torch.float32).clone()
        out["audio_rows_packed"] = a_rows.detach().to("cpu", torch.float32).clone()
        return v_rows, a_rows

    patches.append(_Patch(denoise_loop_mod, "minimax_h3_denoise_loop", _loop_spy))

    _dec = MiniMaxH3DecodingStage.forward

    def _dec_spy(self, batch, server_args):
        state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY) or {}
        out["video_latent"] = batch.latents.detach().cpu().clone()
        out["audio_latent"] = batch.audio_latents.detach().cpu().clone()
        for k in ("latent_t", "latent_h", "latent_w", "audio_t"):
            out[k] = int(state.get(k, 0))
        return _dec(self, batch, server_args)

    patches.append(_Patch(MiniMaxH3DecodingStage, "forward", _dec_spy))
    return patches


def _apply_pass2_rows(batch, payload) -> None:
    """替换 target 初始行（条件行保留管线默认值）。"""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )

    state = batch.extra.get(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY)
    if not isinstance(state, dict):
        raise ValueError("h3_2k: 官方 denoise state 缺失，无法注入 pass2 行")
    lt, lh, lw = int(state["latent_t"]), int(state["latent_h"]), int(state["latent_w"])
    target_n = lt * (lh // 2) * (lw // 2)
    video_target = payload["video_rows_target"].to(torch.float32)
    init_rows = state["initial_video_rows"]
    if init_rows.shape[0] < target_n or video_target.shape[0] != target_n:
        raise ValueError(
            f"h3_2k: 注入 target 行数 {video_target.shape[0]} != 布局 {target_n}"
            f"（latent {lt}x{lh}x{lw}）——检查放大尺寸与请求 target 一致性"
        )
    merged = init_rows.clone()
    merged[merged.shape[0] - target_n:] = video_target.to(merged.dtype)
    state["initial_video_rows"] = merged
    audio_rows = payload.get("audio_rows")
    if audio_rows is not None:
        state["initial_audio_rows"] = audio_rows.to(state["initial_audio_rows"].dtype)
    batch.latents = state["initial_video_rows"]
    batch.audio_latents = state["initial_audio_rows"]
    batch.raw_latent_shape = (1, 24, lt, lh, lw)
    batch.raw_audio_latent_shape = (2, 32, int(state["audio_t"]))


class _Patch:
    def __init__(self, obj, name, value):
        self._obj, self._name = obj, name
        self._saved = getattr(obj, name)
        setattr(obj, name, value)

    def restore(self):
        setattr(self._obj, self._name, self._saved)


def _restore(patches) -> None:
    for p in reversed(patches):
        p.restore()
