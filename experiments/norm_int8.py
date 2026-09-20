"""Fuse the ViT MLP input LayerNorm with its dynamic INT8 quantizer."""

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import DynamicInt8Linear


@triton.jit
def _norm_quantize(
    src,
    weight,
    bias,
    q,
    scales,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(src + row * K + i, i < K, 0).to(tl.float32)
    mean = tl.sum(x, 0) / K
    centered = tl.where(i < K, x - mean, 0)
    variance = tl.sum(centered * centered, 0) / K
    y = centered * tl.rsqrt(variance + EPS)
    y = y * tl.load(weight + i, i < K, 0) + tl.load(bias + i, i < K, 0)
    y = y.to(src.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), 0), 1.0e-8) / 127.0
    quantized = libdevice.nearbyint(y / scale).to(tl.int8)
    tl.store(q + row * K + i, quantized, i < K)
    tl.store(scales + row, scale)


class NormInt8Linear(DynamicInt8Linear):
    def __init__(self, linear, norm, warps):
        nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.eps = norm.eps
        self.warps = warps
        for name in ("weight_int8", "weight_scale", "bias"):
            self.register_buffer(name, getattr(linear, name))
        self.register_buffer("norm_weight", norm.weight.detach())
        self.register_buffer("norm_bias", norm.bias.detach())

    def quantize_input(self, x):
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        _norm_quantize[(x.shape[0],)](
            x,
            self.norm_weight,
            self.norm_bias,
            q,
            scales,
            self.in_features,
            self.eps,
            triton.next_power_of_2(self.in_features),
            num_warps=self.warps,
        )
        return q, scales

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        rows = x.shape[0]
        if not rows:
            return x.new_empty((*shape[:-1], self.out_features), dtype=torch.float16)
        padded_rows = max(32, triton.cdiv(rows, 16) * 16)
        if padded_rows != rows:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded_rows - rows))
        q, scales = self.quantize_input(x)
        mm = torch._int_mm(q, self.weight_int8.T)
        y = (mm.float() * scales[:, None]) * self.weight_scale[None, :] + self.bias
        return y[:rows].to(torch.float16).reshape(*shape[:-1], self.out_features)


def apply_norm_int8(model, warps):
    for block in model.backbone.vision_backbone.trunk.blocks:
        assert isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.mlp.fc1, DynamicInt8Linear)
        block.mlp.fc1 = NormInt8Linear(block.mlp.fc1, block.norm2, warps)
        block.norm2 = nn.Identity()
