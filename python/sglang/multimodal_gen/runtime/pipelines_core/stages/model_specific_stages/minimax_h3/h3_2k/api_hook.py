# SPDX-License-Identifier: Apache-2.0
"""HTTP 侧钩子：单 API 两遍编排（全部安装在 HTTP 进程，幂等）。

三个 wrap：
1. video_api._build_video_sampling_params
   识别大画幅（target.short_edge>768 且请求带 "h3_2k" 配置或直接大画幅），
   取出配置、保存原始 target、把 pass1 改写为 768p。
2. video_api.prepare_request
   把两遍计划塞进 req.extra[H3_2K_PLAN_KEY]（随 pickle 抵达各 rank worker，
   worker 侧据此捕获 pass1 latent 并挂到 OutputBatch.h3_2k_state 回传）。
3. AsyncSchedulerClient.forward
   pass1 返回后读取 h3_2k_state，在 HTTP 进程完成放大/截断/加噪，
   构造 pass2 请求再走一次原生 forward（TP 广播安全），返回 2K 结果。
   tiling 模式下改为逐空间瓦片二次采样 + 融合 + 单次解码。

请求协议（对外仍是 POST /v1/videos 一个接口）：
    {"task":"t2va","prompt":"...","target":{"short_edge":1152,"aspect_ratio":"16:9"},
     "h3_2k":{"upscale":"network","sigma_start":0.90,
              "pass2_steps":6,"freeze_audio":true,
              "pass2_lora":"keep|off",
              "tiling":false}}
"h3_2k" 可省略（大画幅请求默认启用两遍，参数取默认值）。
tiling 可为 true（默认瓦片参数）或 {"tile_px":512,"overlap":0.25}；
仅支持 t2va/ref2va（fl2va/i2va 的逐帧条件需空间对齐裁剪，未实现）。

画幅语义：最终尺寸完全由 target.short_edge + target.aspect_ratio 决定
（官方画布解析，如 16:9+1152 -> 2048x1152）；latent 放大倍率是内部
派生值（目标画布 / pass1 实际画布），不对外暴露。
"""
from __future__ import annotations

import asyncio
import dataclasses

import torch

from . import H3_2K_PASS2_KEY, H3_2K_PLAN_KEY
from . import config, logger, scheduler_tail


def _short_edge(target) -> int:
    try:
        return int((target or {}).get("short_edge", config.NORMAL_SHORT_EDGE))
    except Exception:
        return config.NORMAL_SHORT_EDGE


def _override_plan_canvas(req, px_h: int, px_w: int, verbose: bool = True) -> None:
    """把 resolved plan 的画布改写为指定尺寸（瓦片/校验期望同步用）。

    plan 是 frozen msgspec Struct；shape 中的 width/height 是下游（条件编码、
    latent 网格、解码裁剪）唯一读取的几何真值，同步改写 batch.width/height。
    """
    import msgspec

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    )

    plan = req.extra.get(MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
    if plan is None:
        raise ValueError("h3_2k: prequeue 未生成 resolved plan，无法覆写画布")
    shape = dict(plan.shape)
    old = (int(shape["height"]), int(shape["width"]))
    shape["height"] = int(px_h)
    shape["width"] = int(px_w)
    shape["effective_short_edge"] = min(int(px_h), int(px_w))
    req.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = msgspec.structs.replace(
        plan, shape=shape)
    req.height, req.width = int(px_h), int(px_w)
    if verbose:
        logger.info(f"[h3_2k] 画布覆写: {old[1]}x{old[0]} -> {px_w}x{px_h}")


def install() -> None:
    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
    from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient

    _orig_build = video_api._build_video_sampling_params
    _orig_prepare = video_api.prepare_request
    _orig_forward = AsyncSchedulerClient.forward
    _orig_dispatch = video_api._dispatch_job_async
    _pending: dict[str, dict] = {}

    # ---------- 0) 任务状态中间态（官方只在完成/失败时写 status，期间恒 queued） ----------
    async def _dispatch(job_id, batch, **kw):
        from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE

        await VIDEO_STORE.update_fields(job_id, {"status": "in_progress", "progress": 1})
        return await _orig_dispatch(job_id, batch, **kw)

    # ---------- 1) 识别 + pass1 改写 ----------
    def _build(request_id, request):
        plan_cfg = None
        extra = getattr(request, "model_extra", None) or {}
        if isinstance(extra, dict):
            plan_cfg = extra.pop("h3_2k", None)
        target = getattr(request, "target", None)
        if isinstance(target, dict):
            target_dict = target
        elif target is not None:
            target_dict = dict(getattr(target, "model_dump", dict)())
        else:
            target_dict = {}
        if not config.is_large_target(target_dict):
            return _orig_build(request_id, request)  # 正常分辨率：原路径
        if plan_cfg is None or plan_cfg is True:
            plan_cfg = {}
        if not isinstance(plan_cfg, dict):
            raise ValueError('h3_2k 配置必须是对象，如 {"upscale":"network"}')
        plan_cfg.pop("scale", None)  # 已废弃：画幅由 target 唯一决定

        orig_target = dict(target_dict)
        try:
            if isinstance(target, dict):
                target["short_edge"] = config.NORMAL_SHORT_EDGE
            else:
                target.short_edge = config.NORMAL_SHORT_EDGE
        except Exception as e:
            logger.error(f"[h3_2k] pass1 改写失败({e!r})，放弃两遍")
            return _orig_build(request_id, request)

        sigma_start = plan_cfg.get("sigma_start")
        pass2_steps = plan_cfg.get("pass2_steps")
        pass2_lora = str(plan_cfg.get("pass2_lora", "keep"))
        if pass2_lora not in ("keep", "off"):
            raise ValueError('h3_2k.pass2_lora 只支持 "keep"（沿用服务级 LoRA）'
                             '或 "off"（仅 pass2 任务级卸载）')
        tiling_cfg = plan_cfg.get("tiling", False)
        if tiling_cfg is True:
            tiling_cfg = {}
        if tiling_cfg and str(getattr(request, "task", "t2va")) not in (
                "t2va", "ref2va"):
            raise ValueError("h3_2k.tiling 目前仅支持 t2va/ref2va"
                             "（fl2va/i2va 的逐帧条件需空间对齐裁剪，未实现）")

        _pending[request_id] = {
            "orig_target": orig_target,
            "upscale": str(plan_cfg.get("upscale", "interp")),
            "sigma_start": float(sigma_start) if sigma_start is not None else None,
            "pass2_steps": int(pass2_steps) if pass2_steps is not None else None,
            "freeze_audio": bool(plan_cfg.get("freeze_audio", True)),
            "lora_strength": 0.0 if pass2_lora == "off" else None,
            "tiling": dict(tiling_cfg) if tiling_cfg else None,
        }
        logger.info(f"[h3_2k] 大画幅请求 {request_id}: pass1 改写为 768p, "
              f"orig_target={orig_target}")
        return _orig_build(request_id, request)

    # ---------- 2) 计划塞进 req.extra ----------
    def _prepare(server_args, sampling_params, external_trace_header=None):
        req = _orig_prepare(server_args, sampling_params, external_trace_header)
        rid = getattr(sampling_params, "request_id", None)
        plan = _pending.get(rid)
        if plan is not None:
            req.extra[H3_2K_PLAN_KEY] = plan
        return req

    # ---------- 3) pass1 返回后的两遍编排 ----------
    async def _forward(self, *args, **kwargs):
        result = await _orig_forward(self, *args, **kwargs)
        state = getattr(result, "h3_2k_state", None)
        if not state or not isinstance(state, dict) or "plan" not in state:
            return result
        requests = args[0] if args else kwargs.get("requests")
        return await _run_second_pass(
            self, _orig_forward, result, state, requests)

    video_api._build_video_sampling_params = _build
    video_api.prepare_request = _prepare
    video_api._dispatch_job_async = _dispatch
    AsyncSchedulerClient.forward = _forward


# ---------------------------------------------------------------- 公共帮手
async def _fire_pass2(
    client, orig_forward, server_args, sp1, px_hw, video_rows, audio_rows,
    freeze, lora_strength, sigmas, out_name, capture=False, decode_only=False,
    video_anchor=None,
):
    """构造一次 pass2 形态的 forward（768p 合法 target 载体 + 画布覆写）。

    px_hw=(h,w) 像素；video_rows 为该画布下的加噪 target 行；capture=True
    时附加捕获标记（返回 ob 携带 h3_2k_state.rows）。返回 (ob, req)。
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_SIGMAS_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from .canvas_patch import canvas_cap

    t1 = getattr(sp1, "target", None) or {}
    duration = t1.get("duration_seconds", 5.0) if isinstance(t1, dict) else (
        getattr(t1, "duration_seconds", 5.0))
    sp2 = dataclasses.replace(
        sp1,
        target={"short_edge": config.NORMAL_SHORT_EDGE,
                "aspect_ratio": "1:1",
                "duration_seconds": duration},
        # >=2 是官方 PartitionAdmission 预检硬性要求（即便 sigma extras
        # 单点表只跑 0 步，参数也要给 2 过检）
        num_inference_steps=max(2, len(sigmas["video"]) - 1),
        output_file_name=out_name,
        num_outputs_per_prompt=1,
    )
    req2 = prepare_request(server_args=server_args, sampling_params=sp2)
    with canvas_cap():
        await asyncio.to_thread(sp2.prepare_video_request_for_queue, req2)
    _override_plan_canvas(req2, px_hw[0], px_hw[1], verbose=False)
    req2.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = sigmas
    req2.extra[H3_2K_PASS2_KEY] = {
        "video_rows_target": video_rows.to(torch.float32),
        "audio_rows": audio_rows.to(torch.float32),
        "freeze_audio_rows": audio_rows.to(torch.float32) if freeze else None,
        "lora_strength": lora_strength,
        "video_anchor": video_anchor,
    }
    if capture:
        req2.extra[H3_2K_PLAN_KEY] = {"tile": True}
    if decode_only:
        from . import H3_2K_DECODE_ONLY_KEY

        req2.extra[H3_2K_DECODE_ONLY_KEY] = True
    ob = await orig_forward(client, [req2])
    try:
        _cleanup_temp_dirs(req2)
    except Exception:
        pass
    return ob, req2


async def _run_second_pass(client, orig_forward, ob1, state, requests):
    """放大 + 截断 + 加噪 + 构造 pass2 请求并再次 forward。"""
    plan = state["plan"]
    req1 = requests[0] if requests else None
    if req1 is None:
        logger.info("[h3_2k] 拿不到原始 Req，放弃两遍，返回 768p 结果")
        return ob1
    sp1 = req1.sampling_params
    if int(getattr(sp1, "num_outputs_per_prompt", 1) or 1) > 1:
        logger.info("[h3_2k] 多输出 + 大画幅暂不支持，返回 768p 结果")
        return ob1

    # HTTP 进程的权威 server_args（回传副本的嵌套 config 会退化为 addict.Dict）
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except Exception:
        server_args = state.get("server_args")
    if server_args is None:
        logger.info("[h3_2k] 拿不到 server_args，放弃两遍")
        return ob1

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_SIGMAS_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    try:
        # 阶段状态：pass1 已完成，进入放大/二次采样
        await _update_status(sp1, "refining", 60)

        # 1) 构造 pass2 请求：target 恢复为用户原始值（画幅唯一来源，
        #    官方画布解析，如 16:9+1152 -> 2048x1152）。
        out_name = _suffix_out_name(getattr(sp1, "output_file_name", None))
        orig_target = plan["orig_target"]
        sp2 = dataclasses.replace(
            sp1,
            target={
                "short_edge": int(orig_target.get("short_edge",
                                                  config.NORMAL_SHORT_EDGE)),
                "aspect_ratio": orig_target.get("aspect_ratio", "auto"),
                "duration_seconds": orig_target.get("duration_seconds", 5.0),
            },
            num_inference_steps=4,  # 占位，截断表就绪后覆写
            output_file_name=out_name,
            num_outputs_per_prompt=1,
        )
        req2 = prepare_request_safe(server_args, sp2)

        # 2) prequeue（线程池）。期间放宽画布上限，避免大画幅被 clamp 回 768p
        from .canvas_patch import canvas_cap
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
        )

        with canvas_cap():
            await asyncio.to_thread(sp2.prepare_video_request_for_queue, req2)
        plan2 = req2.extra.get(MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY)
        if plan2 is None:
            raise ValueError("h3_2k: pass2 prequeue 未生成 resolved plan")
        px_w, px_h = int(plan2.shape["width"]), int(plan2.shape["height"])

        # 3) 放大：对齐到 target 解析出的画布（倍率为派生值，内部计算）
        upscaler = config.get_upscaler(plan["upscale"])
        target_hw = (px_h // 16, px_w // 16)
        grid_up, info = upscaler(state["video_latent"], 0.0, target_hw=target_hw)
        # network upscaler 约占数 GB 显存，pass2 前必须同时释放闭包和缓存模型。
        if plan["upscale"] == "network":
            del upscaler
            from . import upscale_network

            upscale_network.release_cached_model()
        _, _, _, h2, w2 = (int(x) for x in grid_up.shape)
        if (h2, w2) != target_hw:
            raise ValueError(
                f"h3_2k: 放大输出 ({h2},{w2}) != target 画布 latent "
                f"{target_hw}——放大器 target_hw 未精确对齐")
        logger.info(f"[h3_2k] pass1 完成, 放大: {info} -> {px_w}x{px_h}（target 解析）")

        # 4) sigma 截断 + 部分加噪（tiling 钳制重绘幅度，见 config 注释）
        if plan.get("tiling"):
            ss = float(plan.get("sigma_start") or config.SIGMA_START_DEFAULT)
            if ss > config.TILING_SIGMA_START_MAX:
                logger.warning(
                    f"[h3_2k] tiling 模式 sigma_start {ss:.2f} 已钳制为 "
                    f"{config.TILING_SIGMA_START_MAX:.2f}——瓦片是独立请求，"
                    "高重绘会让各瓦片按自身画布重新构图，画面碎裂")
                plan = {**plan, "sigma_start": config.TILING_SIGMA_START_MAX}
        sigmas = _build_tail_sigmas(plan)
        seed = int(getattr(sp1, "seed", 42) or 42)
        v_noised, a_noised, sv, sa = scheduler_tail.make_pass2_noise_state(
            grid_up, state["audio_latent"], sigmas, seed)
        logger.info(f"[h3_2k] pass2: sigma_start v={sv:.4f} a={sa:.4f} "
              f"steps={len(sigmas['video']) - 1}")

        freeze = bool(plan.get("freeze_audio", True))
        audio_rows = (state["audio_rows_packed"].clone() if freeze
                      else scheduler_tail.pack_audio_native(a_noised))
        video_rows = minimax_h3_patchify_video_latent(
            v_noised, patch_size=[1, 2, 2])
        lora_strength = plan.get("lora_strength")

        if plan.get("tiling"):
            return await _finish_tiled(
                client, orig_forward, server_args, sp1, ob1, req1,
                v_noised, audio_rows, freeze, lora_strength, sigmas,
                (px_h, px_w), out_name, plan)

        req2.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = sigmas
        req2.extra[H3_2K_PASS2_KEY] = {
            "video_rows_target": video_rows.to(torch.float32),
            "audio_rows": audio_rows.to(torch.float32),
            "freeze_audio_rows": audio_rows.to(torch.float32) if freeze else None,
            "lora_strength": lora_strength,
        }
        sp2.num_inference_steps = len(sigmas["video"]) - 1

        # 5) 二次 forward。req2 的 plan 已是大画布（target 原生解析，无需
        #    覆写）；仅同步原请求的校验期望——最终成片校验用的是原 batch
        _override_plan_canvas(req1, px_h, px_w, verbose=False)
        ob2 = await orig_forward(client, [req2])

        # 6) 元数据对齐 + 清理
        _align_metadata(ob1, ob2)
        try:
            _cleanup_temp_dirs(req2)
        except Exception:
            pass
        logger.info("[h3_2k] pass2 完成")
        return ob2
    except Exception as e:
        # fail-fast：让任务显式失败并暴露原因（避免静默回落 768p 造成
        # "显示成功实为低清"的迷惑；需要宽松回退语义时再改回 return ob1）
        import traceback

        logger.error(f"[h3_2k] 两遍编排失败: {e!r}\n{traceback.format_exc()}")
        raise


# ---------------------------------------------------------------- 分片路径
async def _finish_tiled(
    client, orig_forward, server_args, sp1, ob1, req1,
    v_noised, audio_rows, freeze, lora_strength, sigmas,
    full_px_hw, out_name, plan,
):
    """分片二次采样（MMH3 Split Upscale 完整移植）。

    机制（对照原插件）：
    - 冻结带行锚  ~= noise_mask：重叠带冻结段 alpha=1 每步钉回初始，
      渐变段 1->1-seam_cap（原 fade_mask + seam_denoise 的连续等效）
    - identity 锚：token 维每隔 stride 钉一整帧（防内容漂移/变脸）
    - 时间分块：17n+5 网格切 token 块，重叠区取前块结果做锚（motion 连续）
    - 缝检测修补 ~= seam_polish：corr<0.85 或 |dc|>0.03 才对缝窗口重采样
    - 逐瓦片中值色校 + cross-fade 融合 ~= dc_correct + blend
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )
    from . import tiling as tiling_mod

    plan_tiling = plan.get("tiling") or {}
    tile_px = int(plan_tiling.get("tile_px", config.TILE_PX_DEFAULT))
    ol_ratio = float(plan_tiling.get("overlap", config.TILE_OVERLAP_DEFAULT))
    seam_cap = float(plan_tiling.get("seam_cap", config.SEAM_CAP_DEFAULT))
    do_polish = bool(plan_tiling.get("seam_polish", config.SEAM_POLISH_DEFAULT))
    time_tile = bool(plan_tiling.get("time_tile", config.TIME_TILE_DEFAULT))
    id_stride = int(plan_tiling.get("identity_stride",
                                    config.IDENTITY_STRIDE_DEFAULT))

    _, _, t_len, h_full, w_full = (int(x) for x in v_noised.shape)
    th = max(4, tile_px // 16)
    th -= th % 2
    tw = th
    rows, cols, rlen, clen, rovl, covl = tiling_mod.compute_spatial_grid(
        h_full, w_full, th, tw, ol_ratio)
    n_tiles = len(rows) * len(cols)

    # 时间分块（token 区间，块长约束 ≡2 mod 5 保证 duration 合法）
    total_frames = tiling_mod.frames_for_tokens(t_len)
    blocks = (tiling_mod.time_grid(t_len)
              if time_tile else [(0, t_len)])

    logger.info(f"[h3_2k] tiling: {h_full}x{w_full} latent, T={t_len}"
          f"({total_frames}f) -> {len(rows)}x{len(cols)}={n_tiles} 瓦片 x "
          f"{len(blocks)} 时间块（tile {th}x{tw}, ov {ol_ratio:.0%}, "
          f"seam_cap {seam_cap:.2f}, polish {do_polish}, "
          f"{len(sigmas['video']) - 1} 步/瓦片）")

    canvas = v_noised.clone()
    total_jobs = n_tiles * len(blocks)
    done = 0

    async def _run_tile(t0, t1, r0, c0, tr, tc, ovh, ovw, anchor_extra=None):
        """发一个瓦片请求并融合回 canvas。返回捕获的 out（未融合前）。"""
        tile = canvas[:, :, t0:t1, r0:r0 + tr, c0:c0 + tc].clone()
        rows_tile = minimax_h3_patchify_video_latent(tile, patch_size=[1, 2, 2])
        if anchor_extra is not None:
            anchor = anchor_extra
        else:
            # 行锚合并：冻结/渐变带 + identity 帧（每 stride 个 token 钉整帧，
            # 防止长序列内容漂移；alpha 取 max 等价于逐行最保守保留）
            merge: dict[int, float] = {}
            if ovh > 0 or ovw > 0:
                idx, alpha = tiling_mod.tile_anchor_rows(
                    t1 - t0, tr, tc, ovh, ovw, fade=0.5, seam_cap=seam_cap)
                for i, a in zip(idx.tolist(), alpha.tolist()):
                    merge[i] = a
            hp, wp = tr // 2, tc // 2
            for ti in range(0, t1 - t0, id_stride):
                base = ti * hp * wp
                for j in range(hp * wp):
                    merge[base + j] = 1.0
            if merge:
                keys = sorted(merge)
                anchor = {
                    "idx": torch.tensor(keys, dtype=torch.long),
                    "val": rows_tile[keys].clone().to(torch.float32),
                    "alpha": torch.tensor([merge[k] for k in keys],
                                          dtype=torch.float32),
                }
            else:
                anchor = None
        ob_t, _ = await _fire_pass2(
            client, orig_forward, server_args, sp1,
            (tr * 16, tc * 16), rows_tile, audio_rows, freeze,
            lora_strength, sigmas, None, capture=True,
            video_anchor=anchor)
        state_t = getattr(ob_t, "h3_2k_state", None) or {}
        out = state_t.get("video_latent")
        want = (1, 24, t1 - t0, tr, tc)
        if out is None or tuple(out.shape) != want:
            raise ValueError(
                f"h3_2k tiling: 瓦片捕获失败，video_latent 形状 "
                f"{tuple(out.shape) if out is not None else None} != {want}")
        return out.to(canvas.dtype)

    for bi, (t0, t1) in enumerate(blocks):
        if len(blocks) > 1:
            logger.info(f"[h3_2k] tiling 时间块 {bi + 1}/{len(blocks)}: "
                        f"token [{t0},{t1})")
        for ri, r0 in enumerate(rows):
            for cj, c0 in enumerate(cols):
                tr, tc = rlen[ri], clen[cj]
                ovh, ovw = rovl[ri], covl[cj]
                out = await _run_tile(t0, t1, r0, c0, tr, tc, ovh, ovw)
                out = tiling_mod.dc_correct(
                    out, tiling_mod.overlap_refs(canvas, out, r0, c0, ovh, ovw))
                tiling_mod.blend_into(canvas, out, r0, c0, ovh, ovw)
                done += 1
                await _update_status(sp1, "refining",
                                     60 + int(30 * done / total_jobs))
                logger.info(f"[h3_2k] tiling 瓦片 {done}/{total_jobs} 完成")

    # ---- probe 门控缝修补（seam_polish）：只处理空间内缝 ----
    if do_polish:
        band = config.POLISH_BAND_LATENT
        seams = [("w", c0) for c0 in cols[1:]] + [("h", r0) for r0 in rows[1:]]
        polished = 0
        for axis, pos in seams:
            needs, corr, dc = tiling_mod.seam_quality(canvas, axis, pos)
            if not needs:
                continue
            lo = max(0, pos - band)
            hi = (w_full if axis == "w" else h_full)
            hi = min(hi, pos + band)
            wh = (hi - lo) if axis == "h" else h_full
            ww = w_full if axis == "w" else (hi - lo)
            r0_p, c0_p = (0, lo) if axis == "w" else (lo, 0)
            # 锚用 patch 网格掩膜；融合用 latent 网格（patch 2x2 上采样）
            mask_p = tiling_mod.polish_window_mask(wh // 2, ww // 2)
            mask_l = mask_p.repeat_interleave(2, 0).repeat_interleave(2, 1)
            idx, alpha = tiling_mod.anchor_rows_from_mask(mask_p)
            tile = canvas[:, :, :, r0_p:r0_p + wh, c0_p:c0_p + ww].clone()
            rows_p = minimax_h3_patchify_video_latent(tile, patch_size=[1, 2, 2])
            anchor = {"idx": idx, "val": rows_p[idx].clone().to(torch.float32),
                      "alpha": alpha}
            logger.info(f"[h3_2k] seam polish: {axis}@{pos} corr={corr:.3f} "
                        f"dc={dc:.4f} -> 重采样窗口 [{lo},{hi})")
            ob_t, _ = await _fire_pass2(
                client, orig_forward, server_args, sp1, (wh * 16, ww * 16),
                rows_p, audio_rows, freeze, lora_strength, sigmas, None,
                capture=True, video_anchor=anchor)
            out = (getattr(ob_t, "h3_2k_state", None) or {}).get("video_latent")
            if out is None:
                continue
            out = out.to(canvas.dtype)
            # 按 (1-alpha) 采纳：中心全采纳新值，边缘保持 canvas
            region = canvas[:, :, :, r0_p:r0_p + wh, c0_p:c0_p + ww]
            a3 = mask_l.view(1, 1, 1, wh, ww)
            region.copy_(a3 * region + (1.0 - a3) * out)
            polished += 1
            await _update_status(sp1, "refining", 93)
        logger.info(f"[h3_2k] seam polish 完成：{polished}/{len(seams)} 条缝触发修补")

    # 最终单次解码：正规 2 点表 [1.0, 0.0] 过官方校验（等长 list 且 >=2），
    # decode_only 标记让 worker 跳过 DenoisingStage——零模型 forward，
    # 直接 VAE 解码注入的干净行
    final_rows = minimax_h3_patchify_video_latent(canvas, patch_size=[1, 2, 2])
    decode_sigmas = {"video": [1.0, 0.0], "audio": [1.0, 0.0]}
    ob2, req_f = await _fire_pass2(
        client, orig_forward, server_args, sp1, full_px_hw, final_rows,
        audio_rows, freeze, None, decode_sigmas, out_name, capture=False,
        decode_only=True)
    _override_plan_canvas(req1, full_px_hw[0], full_px_hw[1], verbose=False)
    _align_metadata(ob1, ob2)
    logger.info("[h3_2k] tiling 完成，已融合解码")
    return ob2


# ---------------------------------------------------------------- 小工具
def _build_tail_sigmas(plan) -> dict:
    full = scheduler_tail.build_full_sigmas(50)
    sigma_start = plan.get("sigma_start") or config.SIGMA_START_DEFAULT
    tail = scheduler_tail.tail_sigmas(full, sigma_start)
    steps2 = plan.get("pass2_steps") or config.PASS2_STEPS_DEFAULT
    points = min(int(steps2) + 1, len(tail["video"]))
    return {"video": tail["video"][-points:], "audio": tail["audio"][-points:]}


def prepare_request_safe(server_args, sp2):
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    return prepare_request(server_args=server_args, sampling_params=sp2)


def _suffix_out_name(out_name):
    # 文件名 _2k 插在扩展名前（xxx.mp4 -> xxx_2k.mp4）；
    # 后缀拼在最后会让 ffmpeg 无法推断输出格式而 BrokenPipe
    if not out_name:
        return None
    from pathlib import Path

    p = Path(out_name)
    return f"{p.stem}_2k{p.suffix}" if p.suffix else f"{out_name}_2k"


async def _update_status(sp1, status: str, progress: int) -> None:
    try:
        from sglang.multimodal_gen.runtime.entrypoints.openai.stores import VIDEO_STORE

        await VIDEO_STORE.update_fields(
            str(getattr(sp1, "request_id", "") or ""),
            {"status": status, "progress": progress})
    except Exception:
        pass


def _align_metadata(ob1, ob2) -> None:
    for field in ("rid", "id", "request_id"):
        v = getattr(ob1, field, None)
        if v is not None and hasattr(ob2, field):
            try:
                setattr(ob2, field, v)
            except Exception:
                pass


def _cleanup_temp_dirs(req) -> None:
    """清理 pass2 prequeue 登记的临时素材目录（尽力而为）。"""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        prequeue,
    )

    for name in dir(prequeue):
        if "cleanup" in name.lower():
            getattr(prequeue, name)(req)
            return
