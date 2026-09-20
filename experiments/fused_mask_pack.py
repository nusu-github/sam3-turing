"""Experimental fused bilinear resize, FP16 sigmoid threshold and bit packing."""

import torch
import triton
import triton.language as tl


@triton.jit
def _resize_pack(
    src,
    dst,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
    DIRECT: tl.constexpr,
):
    row = tl.program_id(0)
    byte = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 8)
    p = byte[:, None] * 8 + bit[None, :]
    valid = p < OH * OW
    fy = tl.maximum((p // OW + 0.5) * (IH / OH) - 0.5, 0.0)
    fx = tl.maximum((p % OW + 0.5) * (IW / OW) - 0.5, 0.0)
    y0 = fy.to(tl.int32)
    x0 = fx.to(tl.int32)
    y1 = tl.minimum(y0 + 1, IH - 1)
    x1 = tl.minimum(x0 + 1, IW - 1)
    wy = fy - y0
    wx = fx - x0
    base = src + row * S0
    v00 = tl.load(base + y0 * S1 + x0 * S2, valid, 0).to(tl.float32)
    v01 = tl.load(base + y0 * S1 + x1 * S2, valid, 0).to(tl.float32)
    v10 = tl.load(base + y1 * S1 + x0 * S2, valid, 0).to(tl.float32)
    v11 = tl.load(base + y1 * S1 + x1 * S2, valid, 0).to(tl.float32)
    value = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
    if HALF:
        value = value.to(tl.float16).to(tl.float32)
    if DIRECT and HALF:
        yes = ((value > 0.0009765625) & valid).to(tl.uint32)
    else:
        prob = 1.0 / (1.0 + tl.exp(-value))
        if HALF:
            prob = prob.to(tl.float16).to(tl.float32)
        yes = ((prob > 0.5) & valid).to(tl.uint32)
    packed = tl.sum(yes << bit[None, :], 1).to(tl.uint8)
    row_bytes: tl.constexpr = (OH * OW + 7) // 8
    tl.store(dst + row * row_bytes + byte, packed, byte < row_bytes)


@torch.inference_mode()
def fused_resize_and_pack(logits, size, *, block=128, warps=4, direct=False):
    if (
        logits.ndim != 3
        or logits.dtype not in (torch.float16, torch.float32)
        or not logits.is_cuda
    ):
        raise ValueError("Expected FP16/FP32 CUDA logits [N,H,W]")
    if len(size) != 2 or min(size) < 1:
        raise ValueError("Expected positive height and width")
    oh, ow = size
    row_bytes = triton.cdiv(oh * ow, 8)
    out = torch.empty((len(logits), row_bytes), device=logits.device, dtype=torch.uint8)
    if len(logits):
        _resize_pack[(len(logits), triton.cdiv(row_bytes, block))](
            logits,
            out,
            *logits.shape[-2:],
            oh,
            ow,
            *logits.stride(),
            logits.dtype == torch.float16,
            block,
            direct,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return out
