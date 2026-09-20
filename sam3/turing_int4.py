"""Optional compact 4-bit vision weights, decoded for ordinary FP16 Linear."""

import torch
import torch.nn.functional as F
from torch import nn


class WeightOnlyInt4Linear(nn.Module):
    """Gaussian-quantile or asymmetric uniform levels, with packed nibbles."""

    def __init__(self, linear, group_size=32, asymmetric=False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        w = linear.weight.detach().float().reshape(self.out_features, -1, group_size)
        levels, offset = None, None
        if asymmetric:
            lo, hi = w.amin(2, keepdim=True), w.amax(2, keepdim=True)
            scale = (hi - lo).clamp_min(1e-8) / 15
            offset = lo.half()
            q = ((w - lo) / scale).round().clamp(0, 15).to(torch.uint8)
        else:
            probabilities = (torch.arange(15, device=w.device).float() + 0.5) / 15
            levels = (2 * probabilities - 1).erfinv()
            levels = levels / levels.abs().max()
            scale = w.abs().amax(2, keepdim=True).clamp_min(1e-8)
            boundaries = (levels[:-1] + levels[1:]) * 0.5
            q = torch.bucketize((w / scale).contiguous(), boundaries).to(torch.uint8)
        self.register_buffer("weight_levels", levels)
        self.register_buffer("weight_offset", offset)
        self.register_buffer("weight_packed", q[..., 0::2] | (q[..., 1::2] << 4))
        self.register_buffer("weight_scale", scale.half())
        self.register_buffer(
            "bias", linear.bias.detach() if linear.bias is not None else None
        )

    def forward(self, x):
        q = torch.stack(
            (self.weight_packed & 15, self.weight_packed >> 4), dim=-1
        ).flatten(-2)
        if self.weight_levels is not None:
            weight = self.weight_levels[q.long()] * self.weight_scale.float()
        else:
            weight = q.float() * self.weight_scale.float() + self.weight_offset.float()
        weight = weight.reshape(self.out_features, self.in_features).half()
        return F.linear(x, weight, self.bias)


@torch.no_grad()
def apply_int4_patch(
    processor, *, group_size=32, attention_projections=True, asymmetric=False
):
    """Trade output differences for lower vision-weight memory.

    Apply after apply_turing_patch and before inference. Activations and matrix
    multiplication remain FP16; this is not INT4 arithmetic or NF4. Uses compact
    FP16 group scales with 15 symmetric Gaussian-quantile levels by default;
    asymmetric=True uses 16 uniform levels and an additional FP16 offset.
    Group sizes 16 and 32 are supported. Measured on RTX3090, not physical Turing.
    CPU text, packed masks and image refinements can be applied separately.
    Selected layers must still be ordinary FP16 CUDA Linear modules.
    """
    if not getattr(processor, "_turing_patched", False):
        raise ValueError("Apply the Turing image patch first")
    if processor.model.training:
        raise ValueError("INT4 weights are for inference only")
    if group_size not in (16, 32):
        raise ValueError("group_size must be 16 or 32")
    targets = []
    for block in processor.model.backbone.vision_backbone.trunk.blocks:
        targets.extend((block.mlp, name) for name in ("fc1", "fc2"))
        if attention_projections:
            targets.extend((block.attn, name) for name in ("qkv", "proj"))
    for parent, name in targets:
        layer = getattr(parent, name)
        if not isinstance(layer, nn.Linear):
            raise ValueError(
                "Expected unquantized Linear layers; apply each selection once"
            )
        if layer.weight.device.type != "cuda" or layer.weight.dtype != torch.float16:
            raise ValueError("INT4 conversion requires FP16 CUDA Linear weights")
        if layer.in_features % group_size:
            raise ValueError("group_size must divide every selected input width")
    for parent, name in targets:
        setattr(
            parent,
            name,
            WeightOnlyInt4Linear(getattr(parent, name), group_size, asymmetric),
        )
    return processor
