"""Weight-statistic channel balancing before dynamic INT8 MLP quantization."""

from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sam3.turing_int8 import DynamicInt8Linear


@triton.jit
def _quantize_balanced(
    src, dst, scales, channels, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(src + row * K + i, i < K, 0).to(tl.float32)
    x *= tl.load(channels + i, i < K, 1)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-8) / 127.0
    q = libdevice.nearbyint(x / scale).to(tl.int8)
    tl.store(dst + row * K + i, q, i < K)
    tl.store(scales + row, scale)


class BalancedInt8Linear(DynamicInt8Linear):
    def __init__(self, linear, alpha):
        w = linear.weight.detach().float()
        channel = w.abs().amax(0).clamp_min_(1.0e-6).pow_(-alpha)
        channel /= channel.log().mean().exp()
        channel.clamp_(0.125, 8)
        super().__init__(
            SimpleNamespace(
                in_features=linear.in_features,
                out_features=linear.out_features,
                weight=w * channel[None, :],
                bias=linear.bias,
            )
        )
        self.register_buffer("input_scale", channel.reciprocal())

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        rows = x.shape[0]
        if not rows:
            return x.new_empty((*shape[:-1], self.out_features), dtype=torch.float16)
        padded_rows = max(32, triton.cdiv(rows, 16) * 16)
        if padded_rows != rows:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded_rows - rows))
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        _quantize_balanced[(x.shape[0],)](
            x,
            q,
            scales,
            self.input_scale,
            self.in_features,
            triton.next_power_of_2(self.in_features),
        )
        mm = torch._int_mm(q, self.weight_int8.T)
        result = (mm.float() * scales[:, None]) * self.weight_scale[None, :] + self.bias
        return result[:rows].to(torch.float16).reshape(*shape[:-1], self.out_features)


def apply_balanced_int8(model, alpha, scope="both"):
    for block in model.backbone.vision_backbone.trunk.blocks:
        for name in ("fc1", "fc2"):
            original = getattr(block.mlp, name)
            replacement = (
                BalancedInt8Linear(original, alpha)
                if scope in ("both", name)
                else DynamicInt8Linear(original)
            )
            setattr(block.mlp, name, replacement)
