"""Inference-only, CPU-readable graph transformations for the SAM3 audit.

These are reference implementations, not GPU kernels or an installed SAM3 API.
They preserve real-number algebra; floating-point bit equality is not promised.
"""
from __future__ import annotations

import importlib
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


def import_source_modules(repo: Path):
    """Load the actual modules without executing the GPU-oriented package builder.

    Only package containers are supplied here. No tensor operator or upstream
    module implementation is stubbed. Run this lab in its own Python process.
    """
    for name, relative in [
        ("sam3", "sam3"), ("sam3.model", "sam3/model"),
        ("sam3.sam", "sam3/sam"), ("sam3.perflib", "sam3/perflib"),
    ]:
        if name in sys.modules:
            raise RuntimeError(f"Run in a fresh process: {name} already imported")
        module = types.ModuleType(name)
        module.__path__ = [str(repo / relative)]
        module.__package__ = name
        sys.modules[name] = module
    return {
        name: importlib.import_module(path)
        for name, path in {
            "mask": "sam3.model.maskformer_segmentation",
            "misc": "sam3.model.model_misc",
            "tracker": "sam3.sam.transformer",
            "position": "sam3.model.position_encoding",
            "data": "sam3.model.data_misc",
            "vit": "sam3.model.vitdet",
        }.items()
    }


@torch.no_grad()
def online_attention(q, k, v, mask=None, q_chunk=128, k_chunk=256,
                     scale=None, stats=None):
    """4-D [batch, heads, tokens, channels], FP32/FP64, dropout=0.

    Boolean masks use SDPA semantics (True=allowed); additive masks can contain
    -inf. Fully masked rows return zero. Finite Q/K/V and finite or -inf masks
    are required. No full score matrix is allocated by this implementation.
    A caller-supplied dense mask is still the caller's memory responsibility.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected four-dimensional Q/K/V")
    if not (q.shape[:2] == k.shape[:2] == v.shape[:2]):
        raise ValueError("Batch and head dimensions must agree")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Incompatible Q/K/V shapes")
    if not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device):
        raise ValueError("Q/K/V must share dtype and device")
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError("This reference explicitly supports FP32 and FP64")
    if min(q_chunk, k_chunk, k.shape[-2]) < 1:
        raise ValueError("Positive chunk sizes and nonempty K/V required")
    if mask is not None:
        mask = torch.broadcast_to(mask, (*q.shape[:2], q.shape[-2], k.shape[-2]))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    largest = 0
    tiles = 0
    for qs in range(0, q.shape[-2], q_chunk):
        qt = q[..., qs:qs + q_chunk, :]
        m = torch.full((*qt.shape[:-1], 1), -torch.inf, dtype=q.dtype, device=q.device)
        den = torch.zeros_like(m)
        num = torch.zeros((*qt.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
        for ks in range(0, k.shape[-2], k_chunk):
            scores = (qt @ k[..., ks:ks + k_chunk, :].transpose(-1, -2)) * scale
            largest = max(largest, scores.numel())
            tiles += 1
            if mask is not None:
                tile_mask = mask[..., qs:qs + q_chunk, ks:ks + k_chunk]
                if tile_mask.dtype == torch.bool:
                    scores.masked_fill_(~tile_mask, -torch.inf)
                else:
                    scores.add_(tile_mask)
            new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
            safe_m = torch.where(torch.isfinite(new_m), new_m, 0.0)
            alpha = torch.where(torch.isfinite(m), (m - safe_m).exp(), 0.0)
            p = (scores - safe_m).exp()
            num = alpha * num + p @ v[..., ks:ks + k_chunk, :]
            den = alpha * den + p.sum(-1, keepdim=True)
            m = new_m
        out[..., qs:qs + q_chunk, :] = num / torch.where(den > 0, den, 1.0)
    if stats is not None:
        stats.update(max_score_tile_elements=largest,
                     max_score_tile_bytes=largest * q.element_size(),
                     tile_count=tiles)
    return out


def reassociated_masks(predictor, projection, queries, pixels):
    """E(AF+b) -> (EA)F+Eb; do not cross the mask MLP's nonlinearities."""
    if (projection.kernel_size != (1, 1) or projection.groups != 1
            or projection.stride != (1, 1) or projection.padding != (0, 0)):
        raise ValueError("Requires an ungrouped, stride-1, unpadded 1x1 convolution")
    embedding = predictor.mask_embed(queries)
    folded = embedding @ projection.weight[:, :, 0, 0]
    if queries.ndim == 3:
        formula = "bqc,bchw->bqhw" if pixels.ndim == 4 else "bqc,chw->bqhw"
    elif queries.ndim == 4:
        formula = "lbqc,bchw->lbqhw" if pixels.ndim == 4 else "lbqc,chw->lbqhw"
    else:
        raise ValueError("Expected queries with 3 or 4 dimensions")
    masks = torch.einsum(formula, folded, pixels)
    if projection.bias is not None:
        masks = masks + (embedding @ projection.bias)[..., None, None]
    return masks


@contextmanager
def reassociate_head(head):
    """Temporarily substitute only the two affine operations in the actual head.

    The upstream forward, prompt cross-attention, pixel decoder, GroupNorm,
    semantic head, and query MLP all execute unchanged. Never use concurrently.
    """
    if head.training or head.no_dec or torch.is_grad_enabled():
        raise ValueError("Inference with a query mask predictor is required")
    def masks(queries, pixels):
        return reassociated_masks(head.mask_predictor, head.instance_seg_head,
                                  queries, pixels)
    with patch.object(head.instance_seg_head, "forward", lambda x: x), \
         patch.object(head.mask_predictor, "forward", masks):
        yield


@contextmanager
def fold_tracker_value(attention):
    """Compose Wo @ Wv for the actual one-head, eval-mode RoPEAttention.

    This upstream API has no attention mask and dropout is zero in eval mode.
    All softmax rows have sum one. A masked extension must carry the row sum
    and cannot unconditionally add the same folded V bias to fully masked rows.
    """
    if attention.training or attention.num_heads != 1 or torch.is_grad_enabled():
        raise ValueError("One-head inference is required")
    if attention.use_fa3:
        raise ValueError("This CPU reference uses SDPA")
    wo, wv = attention.out_proj.weight, attention.v_proj.weight
    fused_weight = wo @ wv
    fused_bias = F.linear(attention.v_proj.bias, wo, attention.out_proj.bias)
    with patch.object(attention.v_proj, "forward", lambda x: x), \
         patch.object(attention.out_proj, "forward",
                      lambda x: F.linear(x, fused_weight, fused_bias)):
        yield


def dense_attention(q, k, v):
    """Explicit dense baseline, not a claim about SAM3's dispatched SDPA backend."""
    scores = q @ k.transpose(-1, -2)
    scores.mul_(1.0 / math.sqrt(q.shape[-1]))
    probs = scores.softmax(-1)
    return probs @ v
