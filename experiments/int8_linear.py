"""Calibration-free per-token / per-output-channel INT8 GEMM candidate."""

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quantize(src, dst, scales, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(src + row * K + i, i < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-8) / 127.0
    q = libdevice.nearbyint(x / scale).to(tl.int8)
    tl.store(dst + row * K + i, q, i < K)
    tl.store(scales + row, scale)


class DynamicInt8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach().float()
        scales = w.abs().amax(1).clamp_min_(1.0e-8) / 127
        self.register_buffer(
            "weight_int8",
            (w / scales[:, None]).round().clamp_(-127, 127).to(torch.int8),
        )
        self.register_buffer("weight_scale", scales)
        self.register_buffer(
            "bias",
            (
                linear.bias.detach()
                if linear.bias is not None
                else torch.zeros(
                    self.out_features, device=w.device, dtype=torch.float16
                )
            ),
        )

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        _quantize[(x.shape[0],)](
            x, q, scales, self.in_features, triton.next_power_of_2(self.in_features)
        )
        mm = torch._int_mm(q, self.weight_int8.T)
        result = (mm.float() * scales[:, None]) * self.weight_scale[None, :] + self.bias
        return result.to(torch.float16).reshape(*shape[:-1], self.out_features)
