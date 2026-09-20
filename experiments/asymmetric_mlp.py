"""Use signed INT8 with a per-token zero point for the positive-heavy GELU."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import _quantize


@triton.jit
def _gelu_asymmetric(
    mm,
    row_scales,
    weight_scales,
    bias,
    q,
    scales,
    zeros,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
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
    hi = tl.maximum(tl.max(tl.where(i < K, y, -float("inf")), 0), 0.0)
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


def fused_asymmetric(x, fc1, fc2, warps):
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
    zero2 = torch.empty(padded_rows, device=x.device, dtype=torch.int32)
    _gelu_asymmetric[(padded_rows,)](
        mm1,
        scale1,
        fc1.weight_scale,
        fc1.bias,
        q2,
        scale2,
        zero2,
        fc2.in_features,
        triton.next_power_of_2(fc2.in_features),
        num_warps=warps,
    )
    mm2 = torch._int_mm(q2, fc2.weight_int8.T)
    corrected = mm2 - zero2[:, None] * fc2.weight_sum[None, :]
    y = (corrected.float() * scale2[:, None]) * fc2.weight_scale[None, :] + fc2.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], fc2.out_features)


def apply_asymmetric_mlp(model, stack, warps=8):
    for block in model.backbone.vision_backbone.trunk.blocks:
        mlp = block.mlp
        mlp.fc2.register_buffer(
            "weight_sum", mlp.fc2.weight_int8.sum(1, dtype=torch.int32)
        )

        def forward(x, mlp=mlp):
            return fused_asymmetric(x, mlp.fc1, mlp.fc2, warps)

        stack.enter_context(patch.object(mlp, "forward", forward))
