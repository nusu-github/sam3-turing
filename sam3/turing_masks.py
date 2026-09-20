"""Optional packed binary mask output; probabilities are not retained."""

import math

import torch
import torch.nn.functional as F
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
    if HALF:
        # Match sigmoid's FP16 rounding at 0.5 without computing an exponential.
        yes = ((value > 0.0009765625) & valid).to(tl.uint32)
    else:
        prob = 1.0 / (1.0 + tl.exp(-value))
        yes = ((prob > 0.5) & valid).to(tl.uint32)
    packed = tl.sum(yes << bit[None, :], 1).to(tl.uint8)
    row_bytes: tl.constexpr = (OH * OW + 7) // 8
    tl.store(dst + row * row_bytes + byte, packed, byte < row_bytes)


@triton.jit
def _pack(src, dst, PIXELS: tl.constexpr, BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    groups = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.arange(0, 8)
    indices = groups[:, None] * 8 + bits[None, :]
    values = tl.load(src + row * PIXELS + indices, indices < PIXELS, 0).to(tl.uint32)
    packed = tl.sum(values << bits[None, :], 1).to(tl.uint8)
    tl.store(dst + row * BYTES + groups, packed, groups < BYTES)


@triton.jit
def _unpack(src, dst, PIXELS: tl.constexpr, BYTES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    indices = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + row * BYTES + indices // 8, indices < PIXELS, 0)
    tl.store(
        dst + row * PIXELS + indices,
        ((values >> (indices % 8)) & 1).to(tl.int1),
        indices < PIXELS,
    )


@torch.inference_mode()
def resize_and_pack_masks(logits, size, chunk_size=8, *, fused=True):
    """[N,h,w] logits -> [N,ceil(H*W/8)] uint8, little-endian bits per row.

    FP16/FP32 uses one fused kernel by default, retaining precision rounding.
    fused=False uses PyTorch interpolation and chunk_size-bounded temporaries.
    Each mask has its own byte padding. Fused interpolation can differ at a
    threshold boundary by floating-point rounding.
    """
    if logits.ndim != 3 or not logits.is_cuda or not logits.is_floating_point():
        raise ValueError("Expected floating CUDA logits shaped [N,h,w]")
    if chunk_size < 1 or len(size) != 2 or min(size) < 1 or min(logits.shape[-2:]) < 1:
        raise ValueError("Use a positive chunk_size and output height/width")
    pixels = math.prod(size)
    row_bytes = triton.cdiv(pixels, 8)
    out = torch.empty(
        (logits.shape[0], row_bytes), device=logits.device, dtype=torch.uint8
    )
    if fused and logits.dtype in (torch.float16, torch.float32):
        if len(logits):
            _resize_pack[(len(logits), triton.cdiv(row_bytes, 256))](
                logits,
                out,
                *logits.shape[-2:],
                *size,
                *logits.stride(),
                logits.dtype == torch.float16,
                256,
                enable_fp_fusion=False,
            )
        return out
    for start in range(0, logits.shape[0], chunk_size):
        masks = (
            F.interpolate(
                logits[start : start + chunk_size, None],
                size,
                mode="bilinear",
                align_corners=False,
            ).sigmoid_()
            > 0.5
        )
        _pack[(masks.shape[0], triton.cdiv(row_bytes, 128))](
            masks, out[start : start + chunk_size], pixels, row_bytes, 128
        )
    return out


@torch.inference_mode()
def unpack_masks(packed, size):
    """Decode packed rows to CUDA bool [N,1,H,W]; slice rows to decode on demand."""
    if len(size) != 2 or min(size) < 1:
        raise ValueError("Use a positive output height/width")
    pixels = math.prod(size)
    row_bytes = triton.cdiv(pixels, 8)
    if (
        packed.ndim != 2
        or packed.dtype != torch.uint8
        or not packed.is_cuda
        or packed.shape[1] != row_bytes
    ):
        raise ValueError("Packed CUDA uint8 rows do not match the requested size")
    packed = packed.contiguous()
    out = torch.empty(
        (packed.shape[0], 1, *size), device=packed.device, dtype=torch.bool
    )
    if packed.shape[0]:
        _unpack[(packed.shape[0], triton.cdiv(pixels, 1024))](
            packed, out, pixels, row_bytes, 1024
        )
    return out
