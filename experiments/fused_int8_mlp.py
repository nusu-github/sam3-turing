"""Fuse INT8 dequantization, FP16 GELU and the next activation quantization."""

from unittest.mock import patch

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import DynamicInt8Linear, _quantize


@triton.jit
def _gelu_quantize(
    mm,
    row_scales,
    weight_scales,
    bias,
    q,
    scales,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    TANH: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(mm + row * K + i, i < K, 0).to(tl.float32)
    x *= tl.load(row_scales + row)
    x = x * tl.load(weight_scales + i, i < K, 0) + tl.load(bias + i, i < K, 0)
    x = x.to(tl.float16).to(tl.float32)
    if TANH:
        y = (
            0.5
            * x
            * (1 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        )
    else:
        y = 0.5 * x * (1 + libdevice.erf(x * 0.7071067811865476))
    y = y.to(tl.float16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), 0), 1.0e-8) / 127.0
    quantized = libdevice.nearbyint(y / scale).to(tl.int8)
    tl.store(q + row * K + i, quantized, i < K)
    tl.store(scales + row, scale)


def fused_mlp(x, fc1, fc2, warps, tanh):
    shape = x.shape
    x = x.reshape(-1, fc1.in_features).contiguous()
    rows = x.shape[0]
    if not rows:
        return x.new_empty((*shape[:-1], fc2.out_features), dtype=torch.float16)
    padded_rows = max(32, triton.cdiv(rows, 16) * 16)
    if padded_rows != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_rows - rows))
    q1 = torch.empty_like(x, dtype=torch.int8)
    scale1 = torch.empty(padded_rows, device=x.device, dtype=torch.float32)
    _quantize[(padded_rows,)](
        x, q1, scale1, fc1.in_features, triton.next_power_of_2(fc1.in_features)
    )
    mm1 = torch._int_mm(q1, fc1.weight_int8.T)
    q2 = torch.empty_like(mm1, dtype=torch.int8)
    scale2 = torch.empty_like(scale1)
    _gelu_quantize[(padded_rows,)](
        mm1,
        scale1,
        fc1.weight_scale,
        fc1.bias,
        q2,
        scale2,
        fc2.in_features,
        triton.next_power_of_2(fc2.in_features),
        tanh,
        num_warps=warps,
    )
    mm2 = torch._int_mm(q2, fc2.weight_int8.T)
    y = (mm2.float() * scale2[:, None]) * fc2.weight_scale[None, :] + fc2.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], fc2.out_features)


def apply_fused_int8_mlp(model, stack, warps=4, tanh=False):
    for block in model.backbone.vision_backbone.trunk.blocks:
        mlp = block.mlp
        assert isinstance(mlp.fc1, DynamicInt8Linear)
        assert isinstance(mlp.fc2, DynamicInt8Linear)
        assert isinstance(mlp.norm, nn.Identity) and isinstance(mlp.act, nn.GELU)

        def forward(x, mlp=mlp):
            return fused_mlp(x, mlp.fc1, mlp.fc2, warps, tanh)

        stack.enter_context(patch.object(mlp, "forward", forward))
