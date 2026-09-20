"""Packed 4-bit groupwise weights, decoded for ordinary FP16 linear operations."""

import torch
from torch import nn
import torch.nn.functional as F


class WeightOnlyInt4Linear(nn.Module):
    def __init__(
        self, linear, group_size=32, asymmetric=False, gaussian=False, compact=False
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.compact = compact
        if self.in_features % group_size or group_size % 2:
            raise ValueError("Group size must be even and divide the input width")
        w = linear.weight.detach().float().reshape(self.out_features, -1, group_size)
        self.register_buffer("weight_levels", None)
        if gaussian:
            if asymmetric:
                raise ValueError("Choose either Gaussian levels or asymmetric uniform")
            probabilities = (torch.arange(15, device=w.device).float() + 0.5) / 15
            levels = (2 * probabilities - 1).erfinv()
            levels = levels / levels.abs().max()
            self.weight_levels = levels
            scale = w.abs().amax(2, keepdim=True).clamp_min(1e-8)
            offset = torch.zeros_like(scale)
            boundaries = (levels[:-1] + levels[1:]) * 0.5
            q = torch.bucketize((w / scale).contiguous(), boundaries).to(torch.uint8)
        elif asymmetric:
            lo, hi = w.amin(2, keepdim=True), w.amax(2, keepdim=True)
            scale = (hi - lo).clamp_min(1e-8) / 15
            offset = lo
            q = ((w - lo) / scale).round().clamp(0, 15).to(torch.uint8)
        else:
            scale = w.abs().amax(2, keepdim=True).clamp_min(1e-8) / 7
            offset = -8 * scale
            q = ((w / scale).round().clamp(-7, 7) + 8).to(torch.uint8)
        self.register_buffer("weight_packed", q[..., 0::2] | (q[..., 1::2] << 4))
        self.register_buffer("weight_scale", scale.half() if compact else scale)
        self.register_buffer(
            "weight_offset",
            (offset.half() if asymmetric else None) if compact else offset,
        )
        self.register_buffer(
            "bias", linear.bias.detach() if linear.bias is not None else None
        )

    def forward(self, x):
        low = self.weight_packed & 15
        high = self.weight_packed >> 4
        q = torch.stack((low, high), dim=-1).flatten(-2)
        if self.weight_levels is not None:
            weight = self.weight_levels[q.long()] * self.weight_scale.float()
        elif self.weight_offset is None:
            weight = (q.float() - 8) * self.weight_scale.float()
        else:
            weight = q.float() * self.weight_scale.float() + self.weight_offset.float()
        weight = weight.reshape(self.out_features, self.in_features).half()
        return F.linear(x, weight, self.bias)


def apply_weight_only_int4(
    model,
    group_size=32,
    attention=False,
    asymmetric=False,
    gaussian=False,
    compact=False,
):
    for block in model.backbone.vision_backbone.trunk.blocks:
        for name in ("fc1", "fc2"):
            setattr(
                block.mlp,
                name,
                WeightOnlyInt4Linear(
                    getattr(block.mlp, name), group_size, asymmetric, gaussian, compact
                ),
            )
        if attention:
            for name in ("qkv", "proj"):
                setattr(
                    block.attn,
                    name,
                    WeightOnlyInt4Linear(
                        getattr(block.attn, name),
                        group_size,
                        asymmetric,
                        gaussian,
                        compact,
                    ),
                )
