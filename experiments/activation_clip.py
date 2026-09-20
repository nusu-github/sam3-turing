"""Trade rare activation tails for a finer dynamic INT8 quantization step."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import DynamicInt8Linear


@triton.jit
def _quantize_clip(
    src, dst, scales, K: tl.constexpr, BLOCK: tl.constexpr, CLIP: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(src + row * K + i, i < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) * CLIP, 1.0e-8) / 127.0
    q = tl.minimum(tl.maximum(libdevice.nearbyint(x / scale), -127.0), 127.0).to(
        tl.int8
    )
    tl.store(dst + row * K + i, q, i < K)
    tl.store(scales + row, scale)


@triton.jit
def _gelu_clip(
    mm,
    row_scales,
    weight_scales,
    bias,
    q,
    scales,
    zeros,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    CLIP: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(mm + row * K + i, i < K, 0).to(tl.float32)
    x *= tl.load(row_scales + row)
    x = x * tl.load(weight_scales + i, i < K, 0) + tl.load(bias + i, i < K, 0)
    x = x.to(tl.float16).to(tl.float32)
    y = (
        (0.5 * x * (1 + libdevice.erf(x * 0.7071067811865476)))
        .to(tl.float16)
        .to(tl.float32)
    )
    lo = tl.minimum(tl.min(tl.where(i < K, y, float("inf")), 0), 0.0)
    hi = tl.maximum(tl.max(tl.where(i < K, y, -float("inf")), 0), 0.0) * CLIP
    scale = tl.maximum(hi - lo, 1.0e-8) / 255.0
    zero = tl.minimum(
        tl.maximum(libdevice.nearbyint(-lo / scale) - 128.0, -128.0), 127.0
    )
    quantized = tl.minimum(
        tl.maximum(libdevice.nearbyint(y / scale) + zero, -128.0), 127.0
    ).to(tl.int8)
    tl.store(q + row * K + i, quantized, i < K)
    tl.store(scales + row, scale)
    tl.store(zeros + row, zero.to(tl.int32))


def quantize(x, ratio):
    rows, width = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    _quantize_clip[(rows,)](x, q, scale, width, triton.next_power_of_2(width), ratio)
    return q, scale


def _input(x, width):
    x = x.reshape(-1, width).contiguous()
    rows = x.shape[0]
    padded = max(32, triton.cdiv(rows, 16) * 16)
    if rows and padded != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - rows))
    return x, rows


def linear_forward(x, linear, ratio):
    shape = x.shape
    x, rows = _input(x, linear.in_features)
    if not rows:
        return x.new_empty((*shape[:-1], linear.out_features), dtype=torch.float16)
    q, scale = quantize(x, ratio)
    mm = torch._int_mm(q, linear.weight_int8.T)
    y = (mm.float() * scale[:, None]) * linear.weight_scale[None, :] + linear.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], linear.out_features)


def fused_forward(x, fc1, fc2, input_ratio, gelu_ratio):
    shape = x.shape
    x, rows = _input(x, fc1.in_features)
    if not rows:
        return x.new_empty((*shape[:-1], fc2.out_features), dtype=torch.float16)
    q1, scale1 = quantize(x, input_ratio)
    mm1 = torch._int_mm(q1, fc1.weight_int8.T)
    q2 = torch.empty_like(mm1, dtype=torch.int8)
    scale2 = torch.empty_like(scale1)
    zero2 = torch.empty(x.shape[0], device=x.device, dtype=torch.int32)
    _gelu_clip[(x.shape[0],)](
        mm1,
        scale1,
        fc1.weight_scale,
        fc1.bias,
        q2,
        scale2,
        zero2,
        fc2.in_features,
        triton.next_power_of_2(fc2.in_features),
        gelu_ratio,
        num_warps=4,
    )
    mm2 = torch._int_mm(q2, fc2.weight_int8.T)
    mm2 = mm2 - zero2[:, None] * fc2.weight_sum[None, :]
    y = (mm2.float() * scale2[:, None]) * fc2.weight_scale[None, :] + fc2.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], fc2.out_features)


def apply_activation_clip(model, stack, input_ratio=1.0, gelu_ratio=1.0):
    if not (0 < input_ratio <= 1 and 0 < gelu_ratio <= 1):
        raise ValueError("Clipping ratios must be in (0, 1]")
    for block in model.backbone.vision_backbone.trunk.blocks:
        mlp = block.mlp
        if not isinstance(mlp.fc1, DynamicInt8Linear) or not hasattr(
            mlp.fc2, "weight_sum"
        ):
            raise ValueError("This candidate requires asymmetric fused INT8 MLPs")
        if input_ratio != 1:
            for name in ("qkv", "proj"):
                linear = getattr(block.attn, name)
                if not isinstance(linear, DynamicInt8Linear):
                    raise ValueError(
                        "This candidate requires INT8 attention projections"
                    )

                def forward(x, linear=linear):
                    return linear_forward(x, linear, input_ratio)

                stack.enter_context(patch.object(linear, "forward", forward))

        def forward(x, mlp=mlp):
            return fused_forward(x, mlp.fc1, mlp.fc2, input_ratio, gelu_ratio)

        stack.enter_context(patch.object(mlp, "forward", forward))
