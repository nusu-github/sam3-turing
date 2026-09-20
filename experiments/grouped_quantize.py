"""Quantize several token rows per CTA to amortize per-block overhead."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import DynamicInt8Linear


@triton.jit
def _quantize_rows(
    src,
    dst,
    scales,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    x = tl.load(
        src + row[:, None] * K + col[None, :],
        (row[:, None] < M) & (col[None, :] < K),
        0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-8) / 127.0
    q = libdevice.nearbyint(x / scale[:, None]).to(tl.int8)
    tl.store(
        dst + row[:, None] * K + col[None, :],
        q,
        (row[:, None] < M) & (col[None, :] < K),
    )
    tl.store(scales + row, scale, row < M)


@triton.jit
def _gelu_rows(
    mm,
    row_scales,
    weight_scales,
    bias,
    q,
    scales,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    valid = (row[:, None] < M) & (col[None, :] < K)
    x = tl.load(mm + row[:, None] * K + col[None, :], valid, 0).to(tl.float32)
    x *= tl.load(row_scales + row, row < M, 0)[:, None]
    x = (
        x * tl.load(weight_scales + col, col < K, 0)[None, :]
        + tl.load(bias + col, col < K, 0)[None, :]
    )
    x = x.to(tl.float16).to(tl.float32)
    y = (
        (0.5 * x * (1 + libdevice.erf(x * 0.7071067811865476)))
        .to(tl.float16)
        .to(tl.float32)
    )
    scale = tl.maximum(tl.max(tl.abs(y), 1), 1.0e-8) / 127.0
    quantized = libdevice.nearbyint(y / scale[:, None]).to(tl.int8)
    tl.store(q + row[:, None] * K + col[None, :], quantized, valid)
    tl.store(scales + row, scale, row < M)


def quantize(x, group, warps):
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(m, device=x.device, dtype=torch.float32)
    _quantize_rows[(triton.cdiv(m, group),)](
        x, q, scales, m, k, triton.next_power_of_2(k), group, num_warps=warps
    )
    return q, scales


def linear_forward(x, linear, group, warps):
    shape = x.shape
    x = x.reshape(-1, linear.in_features).contiguous()
    rows = x.shape[0]
    if not rows:
        return x.new_empty((*shape[:-1], linear.out_features), dtype=torch.float16)
    padded = max(32, triton.cdiv(rows, 16) * 16)
    if padded != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - rows))
    q, scales = quantize(x, group, warps)
    mm = torch._int_mm(q, linear.weight_int8.T)
    y = (mm.float() * scales[:, None]) * linear.weight_scale[None, :] + linear.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], linear.out_features)


def fused_forward(x, fc1, fc2, group, warps, gelu_group, gelu_warps):
    shape = x.shape
    x = x.reshape(-1, fc1.in_features).contiguous()
    rows = x.shape[0]
    if not rows:
        return x.new_empty((*shape[:-1], fc2.out_features), dtype=torch.float16)
    padded = max(32, triton.cdiv(rows, 16) * 16)
    if padded != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - rows))
    q1, scale1 = quantize(x, group, warps)
    mm1 = torch._int_mm(q1, fc1.weight_int8.T)
    q2 = torch.empty_like(mm1, dtype=torch.int8)
    scale2 = torch.empty_like(scale1)
    _gelu_rows[(triton.cdiv(padded, gelu_group),)](
        mm1,
        scale1,
        fc1.weight_scale,
        fc1.bias,
        q2,
        scale2,
        padded,
        fc2.in_features,
        triton.next_power_of_2(fc2.in_features),
        gelu_group,
        num_warps=gelu_warps,
    )
    mm2 = torch._int_mm(q2, fc2.weight_int8.T)
    y = (mm2.float() * scale2[:, None]) * fc2.weight_scale[None, :] + fc2.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], fc2.out_features)


def apply_grouped_quantize(model, stack, group=4, warps=4, gelu_group=1, gelu_warps=8):
    for module in model.modules():
        if type(module) is DynamicInt8Linear:

            def forward(x, module=module):
                return linear_forward(x, module, group, warps)

            stack.enter_context(patch.object(module, "forward", forward))
    for block in model.backbone.vision_backbone.trunk.blocks:
        mlp = block.mlp

        def forward(x, mlp=mlp):
            return fused_forward(
                x, mlp.fc1, mlp.fc2, group, warps, gelu_group, gelu_warps
            )

        stack.enter_context(patch.object(mlp, "forward", forward))
