"""Fold LayerNorm's affine parameters into the first INT8 MLP projection."""

from types import SimpleNamespace

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from norm_int8 import NormInt8Linear
from sam3.turing_int8 import DynamicInt8Linear


@triton.jit
def _center_quantize(
    src, q, scales, K: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(src + row * K + i, i < K, 0).to(tl.float32)
    mean = tl.sum(x, 0) / K
    centered = tl.where(i < K, x - mean, 0)
    variance = tl.sum(centered * centered, 0) / K
    scale = tl.maximum(tl.max(tl.abs(centered), 0), 1.0e-8) / 127.0
    quantized = libdevice.nearbyint(centered / scale).to(tl.int8)
    tl.store(q + row * K + i, quantized, i < K)
    tl.store(scales + row, scale * tl.rsqrt(variance + EPS))


class FoldedNormInt8Linear(NormInt8Linear):
    def __init__(self, linear, norm, warps):
        # W * (gamma * (x - mean) / std + beta) + b
        # = (W * gamma) * (x - mean) / std + (W * beta + b).
        w = linear.weight.detach().float()
        bias = w @ norm.bias.detach().float()
        if linear.bias is not None:
            bias += linear.bias.detach().float()
        DynamicInt8Linear.__init__(
            self,
            SimpleNamespace(
                in_features=linear.in_features,
                out_features=linear.out_features,
                weight=w * norm.weight.detach().float()[None, :],
                bias=bias,
            ),
        )
        self.eps = norm.eps
        self.warps = warps

    def quantize_input(self, x):
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        _center_quantize[(x.shape[0],)](
            x,
            q,
            scales,
            self.in_features,
            self.eps,
            triton.next_power_of_2(self.in_features),
            num_warps=self.warps,
        )
        return q, scales


def apply_folded_norm_int8(model, warps):
    for block in model.backbone.vision_backbone.trunk.blocks:
        assert isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.mlp.fc1, nn.Linear)
        block.mlp.fc1 = FoldedNormInt8Linear(block.mlp.fc1, block.norm2, warps)
        block.mlp.fc2 = DynamicInt8Linear(block.mlp.fc2)
        block.norm2 = nn.Identity()
