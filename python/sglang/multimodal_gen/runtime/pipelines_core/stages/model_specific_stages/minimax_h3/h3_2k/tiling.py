# SPDX-License-Identifier: Apache-2.0
"""分片（瓦片化）算法：网格切分、渐变蒙版、中值色校、融合写回。

算法移植自 LBH-123-AI MMH3_Split_Upscale（ComfyUI），作用在
全网格 latent [1,24,T,H,W] 的 H/W 维上。refine_fn 由调用方提供
（对单瓦片 latent 跑一次部分去噪），本模块只负责切/融/校色。

注意：ref2va/fl2va 的参考条件行裁剪未包含在本模块（需要同步裁
条件 rows，二期）；t2va 或无条件精修可直接使用。
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------- 网格
def grid_1d(size: int, tile: int, overlap: int, min_tile: int = 0):
    """一维切分：返回 (starts, lengths, overlaps)。"""
    if size <= tile:
        return [0], [size], [0]
    stride = tile - overlap
    n = (size - overlap + stride - 1) // stride
    if (n - 1) * stride + tile < size:
        n += 1
    starts = [i * stride for i in range(n)]
    lengths = [min(tile, size - s) for s in starts]
    if min_tile > 0 and n >= 2 and size - starts[-1] < min_tile:
        new_last = size - min_tile
        if starts[-2] < new_last < starts[-2] + lengths[-2]:
            starts[-1], lengths[-1] = new_last, size - new_last
    overlaps = [0] * n
    for i in range(1, n):
        overlaps[i] = max(0, starts[i - 1] + lengths[i - 1] - starts[i])
    return starts, lengths, overlaps


def compute_spatial_grid(h: int, w: int, tile_h: int, tile_w: int,
                         ol_ratio: float = 0.25, min_tile: int = 0):
    """在 latent 全网格上切瓦片。tile/overlap 均为 latent 单位。"""
    ol_h, ol_w = int(tile_h * ol_ratio), int(tile_w * ol_ratio)
    rows, rlen, rovl = grid_1d(h, tile_h, ol_h, min_tile)
    cols, clen, covl = grid_1d(w, tile_w, ol_w, min_tile)
    return rows, cols, rlen, clen, rovl, covl


# ---------------------------------------------------------------- 蒙版
def fade_mask(tile_h: int, tile_w: int, ovh: int, ovw: int,
              done_top: bool, done_left: bool, fade: float = 0.5,
              seam_cap: float = 1.0) -> torch.Tensor:
    """1=自由重采样，0=冻结。重叠带 = 冻结段 + 渐变段(0->seam_cap)。"""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)

    def profile(n: int, ov: int) -> torch.Tensor:
        p = torch.ones(n, dtype=torch.float32)
        f = min(int(ov * fade), ov)
        frozen = ov - f
        p[:frozen] = 0.0
        if f > 0:
            p[frozen:ov] = torch.linspace(0.0, seam_cap, f)
        return p

    if done_left and ovw > 0:
        mask = torch.minimum(mask, profile(tile_w, ovw)[None, :])
    if done_top and ovh > 0:
        mask = torch.minimum(mask, profile(tile_h, ovh)[:, None])
    return mask


# ---------------------------------------------------------------- 色校
def dc_correct(new: torch.Tensor, refs: list, clamp: float = 0.05,
               min_samples: int = 256) -> torch.Tensor:
    """对 new 按重叠区参考做逐通道中值差校正（防块间色偏/闪烁）。"""
    pairs = [(a, b) for a, b in refs
             if a is not None and b is not None and a.numel() >= min_samples]
    if not pairs:
        return new
    c = new.shape[1]
    a = torch.cat([x.flatten(2).transpose(0, 1).reshape(-1, c) for x, _ in pairs])
    b = torch.cat([y.flatten(2).transpose(0, 1).reshape(-1, c) for _, y in pairs])
    dc = (a - b).median(dim=0).values.clamp(-clamp, clamp)
    return new - dc.view(1, c, 1, 1, 1).to(new.dtype)


# ---------------------------------------------------------------- 冻结带行锚
# noise_mask 的等效实现：原插件在 ComfyUI 采样器里用 mask 控制重采样范围，
# SGLang 无 mask 机制，改为在 denoise loop 每步把冻结带的行钉回/混合回
# 初始值。行索引按 packed 布局 (t, h_patch, w_patch) 字典序计算。
def tile_anchor_rows(t_tokens: int, tile_h: int, tile_w: int,
                     ovh: int, ovw: int, fade: float = 0.5,
                     seam_cap: float = 0.6):
    """计算瓦片请求的行锚（noise_mask 冻结带 + 渐变带）。

    返回 (idx LongTensor[n], alpha Tensor[n])；alpha= 初始内容保留比例：
    冻结段 1.0（每步完全钉回初始值），渐变段 1.0 -> 1-seam_cap 线性下降。
    alpha=0 的行不入表（完全自由重采样）。重叠带只取左/上（画布上已有
    已完成内容的方向），与 blend_into 的融合方向一致。
    """
    hp, wp = tile_h // 2, tile_w // 2        # patch 网格
    oh, ow = max(0, ovh // 2), max(0, ovw // 2)  # 重叠带 patch 数
    fh = max(1, int(oh * fade)) if oh else 0
    fw = max(1, int(ow * fade)) if ow else 0

    def col_alpha(i: int, o: int, f: int) -> float:
        if i >= o:
            return 0.0
        if i < o - f:                      # 冻结段
            return 1.0
        return 1.0 - seam_cap * (i - (o - f) + 1) / f  # 渐变段

    idx, alpha = [], []
    for ti in range(t_tokens):
        base_t = ti * hp * wp
        for hi in range(hp):
            base_h = base_t + hi * wp
            av = col_alpha(hi, oh, fh)
            for wi in range(wp):
                a = max(av, col_alpha(wi, ow, fw))
                if a > 1e-6:
                    idx.append(base_h + wi)
                    alpha.append(min(1.0, a))
    return (torch.tensor(idx, dtype=torch.long),
            torch.tensor(alpha, dtype=torch.float32))


def anchor_rows_from_mask(mask: torch.Tensor):
    """从 [hp, wp] 的 alpha 图生成行锚（polish 窗口用）。"""
    hp, wp = mask.shape
    idx, alpha = [], []
    for hi in range(hp):
        for wi in range(wp):
            a = float(mask[hi, wi])
            if a > 1e-6:
                idx.append(hi * wp + wi)
                alpha.append(a)
    return (torch.tensor(idx, dtype=torch.long),
            torch.tensor(alpha, dtype=torch.float32))


# ---------------------------------------------------------------- 缝检测与修补
def seam_quality(canvas: torch.Tensor, axis: str, pos: int,
                 band: int = 4, corr_thresh: float = 0.85,
                 dc_thresh: float = 0.03):
    """检测 canvas 上 position=pos 的缝（axis 'h' 横缝 / 'w' 竖缝）。

    返回 (needs_polish: bool, corr: float, dc: float)。指标照原插件：
    缝两侧 band 宽 latent 带，去均值逐通道相关系数中值、中值色差。
    """
    _, _, t, h, w = (int(x) for x in canvas.shape)
    b = min(band, pos, (h if axis == "h" else w) - pos)
    if b <= 0:
        return False, 1.0, 0.0
    if axis == "w":
        a = canvas[:, :, :, :, pos - b:pos]
        c = canvas[:, :, :, :, pos:pos + b]
    else:
        a = canvas[:, :, :, pos - b:pos, :]
        c = canvas[:, :, :, pos:pos + b, :]
    ch = a.shape[1]
    af = a.flatten(2).transpose(0, 1).reshape(ch, -1)   # [ch, n]
    cf = c.flatten(2).transpose(0, 1).reshape(ch, -1)
    corr = torch.tensor([
        torch.cosine_similarity(af[i] - af[i].mean(), cf[i] - cf[i].mean(), dim=0)
        for i in range(ch)])
    dc = float((af - cf).median().abs())
    needs = bool(corr.median() < corr_thresh or dc > dc_thresh)
    return needs, float(corr.median()), dc


def polish_window_mask(hp: int, wp: int, frozen: int = 2,
                       fade: int = 4) -> torch.Tensor:
    """polish 窗口的 alpha 图：两侧 frozen 段全冻结，向中心渐变到 0。"""
    m = torch.zeros(hp, wp, dtype=torch.float32)

    def ramp(n: int) -> torch.Tensor:
        r = torch.ones(n)
        if fade > 0:
            r[:fade] = torch.linspace(1.0, 0.0, fade + 1)[1:]
        return r

    m[:, :] = 0.0
    col = ramp(wp)
    row = ramp(hp)
    m = torch.minimum(col.view(1, wp).expand(hp, wp),
                      row.view(hp, 1).expand(hp, wp))
    m[:, :frozen] = 1.0
    m[:, wp - frozen:] = 1.0
    return m


# ---------------------------------------------------------------- 时间分块
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # H3 固有：每 5 token 17 帧


def tokens_for_frames(frames: int) -> int:
    """累计换算：frames 帧 -> token 数。"""
    n, t = 0, 0
    while n < frames:
        n += FRAME_PER_TOKEN[t % 5]
        t += 1
    return t if n == frames else -1  # -1 = 不在合法网格


def frames_for_tokens(tokens: int) -> int:
    """累计换算：tokens -> 帧数（时间块的总帧数反推用）。"""
    return sum(FRAME_PER_TOKEN[t % 5] for t in range(tokens))


def legal_frame_boundaries(total_frames: int):
    """17n+5 网格上的合法帧边界（token 组边界）。"""
    bounds = [0]
    n = 0
    while n < total_frames:
        n += FRAME_PER_TOKEN[(tokens_for_frames(n) if n else 0) % 5]
        if n <= total_frames and tokens_for_frames(n) > 0:
            bounds.append(n)
    return sorted(set(b for b in bounds if 0 <= b <= total_frames))


def time_grid(total_tokens: int, block_tokens: int = 22,
              overlap_tokens: int = 7):
    """时间分块（token 区间列表）。

    约束（推导自官方 duration 帧网格 5+17m <-> token ≡ 2 mod 5）：
    - 每块 token 数 ≡ 2 (mod 5)，保证块作为独立请求时 duration 合法
      （22 token = 73f = 5+17*4 ✓）
    - 起点 ≡ 0 (mod 5)（含 0），终点 ≡ 2 (mod 5) 或等于总长
    单块直通时返回 [(0, total_tokens)]。
    """
    if total_tokens <= block_tokens:
        return [(0, total_tokens)]
    blocks, a = [], 0
    while a < total_tokens:
        cands = list(range(a + 2, a + block_tokens + 1, 5))
        b = max(cands) if cands else a + 2
        b = min(b, total_tokens)
        # 尾块太短并入前块
        if blocks and 0 < total_tokens - b < max(7, block_tokens // 2):
            b = total_tokens
        blocks.append((a, b))
        if b >= total_tokens:
            break
        a = max(0, b - overlap_tokens)
        a -= a % 5                     # snap 起点 ≡ 0 mod 5
        if a >= b:                     # 防死循环（极短总量）
            a = max(0, b - 5)
            if a <= blocks[-1][0]:
                blocks[-1] = (blocks[-1][0], total_tokens)
                break
    return blocks


# ---------------------------------------------------------------- 融合写回
def blend_into(canvas: torch.Tensor, out: torch.Tensor, r0: int, c0: int,
               ovh: int, ovw: int) -> None:
    """把瓦片结果 out 写回 canvas 的 (r0, c0) 位置。

    重叠带（左/上）线性 cross-fade，其余区域直接覆盖。原地修改 canvas。
    """
    canvas[:, :, :, r0:r0 + out.shape[3], c0:c0 + out.shape[4]] = out
    if ovw > 0:
        wt = torch.linspace(0.0, 1.0, ovw, device=out.device,
                            dtype=out.dtype).view(1, 1, 1, 1, -1)
        region = canvas[:, :, :, r0:r0 + out.shape[3], c0:c0 + out.shape[4]]
        old = canvas[:, :, :, r0:r0 + out.shape[3],
                     max(0, c0 - ovw):c0].clone()  # 画布上重叠带旧内容
        if old.shape[4] == ovw:
            region[:, :, :, :, :ovw] = (old * (1 - wt) + out[..., :ovw] * wt)


def overlap_refs(canvas, out, r0, c0, ovh, ovw, min_samples: int = 256):
    """取 (新瓦片重叠区, 画布重叠区) 参照对，供 dc_correct 用。"""
    refs = []
    h, w = out.shape[3], out.shape[4]
    if ovw > 0 and c0 > 0:
        a = out[:, :, :, :, :ovw]
        b = canvas[:, :, :, r0:r0 + h, max(0, c0 - ovw):c0]
        if b.shape[4] == a.shape[4] and a.numel() >= min_samples:
            refs.append((a, b))
    if ovh > 0 and r0 > 0:
        a = out[:, :, :, :ovh, :]
        b = canvas[:, :, :, max(0, r0 - ovh):r0, c0:c0 + w]
        if b.shape[3] == a.shape[3] and a.numel() >= min_samples:
            refs.append((a, b))
    return refs


# ---------------------------------------------------------------- 高层
def tile_refine(latent: torch.Tensor, refine_fn, tile_hw: tuple[int, int],
                ol_ratio: float = 0.25, fade: float = 0.5,
                seam_cap: float = 1.0, color_match: bool = True) -> torch.Tensor:
    """逐瓦片调用 refine_fn 并无缝拼回。

    refine_fn(tile_latent: [1,24,t,h,w], base_tile: 同形) -> [1,24,t,h,w]
      （base_tile 为当前画布上的原瓦片，供 refine_fn 做参考/加噪起点）
    """
    _, _, t, h, w = (int(x) for x in latent.shape)
    th, tw = tile_hw
    rows, cols, rlen, clen, rovl, covl = compute_spatial_grid(
        h, w, th, tw, ol_ratio)
    canvas = latent.clone()

    for ri, r0 in enumerate(rows):
        for cj, c0 in enumerate(cols):
            tr, tc = rlen[ri], clen[cj]
            ovh, ovw = rovl[ri], covl[cj]
            base = canvas[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
            out = refine_fn(base.clone(), base)

            if color_match:
                refs = []
                if cj > 0 and ovw > 0:
                    refs.append((out[:, :, :, :, :ovw], base[:, :, :, :, :ovw]))
                if ri > 0 and ovh > 0:
                    refs.append((out[:, :, :, :ovh, :], base[:, :, :, :ovh, :]))
                out = dc_correct(out, refs)

            # 重叠带线性加权融合写回
            region = canvas[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
            if cj > 0 and ovw > 0:
                wt = torch.linspace(0.0, 1.0, ovw).view(1, 1, 1, 1, -1)
                region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1 - wt)
                                            + out[:, :, :, :, :ovw] * wt)
            if ri > 0 and ovh > 0:
                wt = torch.linspace(0.0, 1.0, ovh).view(1, 1, 1, -1, 1)
                region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1 - wt)
                                            + out[:, :, :, :ovh, :] * wt)
            band = torch.zeros(1, 1, 1, tr, tc, dtype=torch.bool)
            if cj > 0 and ovw > 0:
                band[:, :, :, :, :ovw] = True
            if ri > 0 and ovh > 0:
                band[:, :, :, :ovh, :] = True
            canvas[:, :, :, r0:r0 + tr, c0:c0 + tc] = torch.where(
                band, region, out)

    return canvas
