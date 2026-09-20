"""Optional packed binary mask output; probabilities are not retained."""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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
def resize_and_pack_masks(logits, size, chunk_size=8):
    """[N,h,w] logits -> [N,ceil(H*W/8)] uint8, little-endian bits per row.

    Resize, sigmoid and threshold run before packing, with the input precision.
    Chunking bounds temporary storage; each mask has its own byte padding.
    """
    if logits.ndim != 3 or not logits.is_cuda or not logits.is_floating_point():
        raise ValueError("Expected floating CUDA logits shaped [N,h,w]")
    if chunk_size < 1 or len(size) != 2 or min(size) < 1:
        raise ValueError("Use a positive chunk_size and output height/width")
    pixels = math.prod(size)
    row_bytes = triton.cdiv(pixels, 8)
    out = torch.empty(
        (logits.shape[0], row_bytes), device=logits.device, dtype=torch.uint8
    )
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
