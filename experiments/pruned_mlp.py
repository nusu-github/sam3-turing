"""Drop low weight-norm hidden channels without calibration or retraining."""

import torch
from torch import nn

from sam3.turing_int8 import DynamicInt8Linear


def _weight(linear):
    if isinstance(linear, DynamicInt8Linear):
        return linear.weight_int8.float() * linear.weight_scale[:, None]
    return linear.weight.detach().float()


def apply_pruned_mlp(model, keep, layers=None, compensate=False):
    if not 0 < keep < 1:
        raise ValueError("keep must be between zero and one")
    blocks = model.backbone.vision_backbone.trunk.blocks
    selected = set(range(len(blocks)) if layers is None else layers)
    for i, block in enumerate(blocks):
        if i not in selected:
            continue
        fc1, fc2 = block.mlp.fc1, block.mlp.fc2
        w1, w2 = _weight(fc1), _weight(fc2)
        # A cheap channel ranking, not an estimate of segmentation quality.
        gamma, beta = block.norm2.weight.float(), block.norm2.bias.float()
        mean = w1 @ beta
        if fc1.bias is not None:
            mean = mean + fc1.bias.float()
        variance = (w1 * gamma[None, :]).square().sum(1)
        energy = variance + mean.square()
        score = energy.sqrt() * w2.square().mean(0).sqrt()
        count = max(64, int(fc1.out_features * keep) // 64 * 64)
        indices = score.topk(count).indices.sort().values
        if compensate:
            # Approximate E[GELU(z)] under independent standard-normal LN inputs.
            # This replaces discarded channels with a constant; no image fitting.
            denominator = (1 + variance).sqrt()
            t = mean / denominator
            expected = 0.5 * mean * (1 + torch.erf(t / 2**0.5))
            expected += (
                variance
                / denominator
                * torch.exp(-0.5 * t.square())
                / (2 * torch.pi) ** 0.5
            )
            expected[indices] = 0
            corrected = fc2.bias.float() + w2 @ expected
            if isinstance(fc2, DynamicInt8Linear):
                fc2.bias = corrected.to(fc2.bias.dtype)
            else:
                fc2.bias = nn.Parameter(
                    corrected.to(fc2.bias.dtype), requires_grad=False
                )
        if isinstance(fc1, DynamicInt8Linear):
            fc1.weight_int8 = fc1.weight_int8[indices].contiguous()
            fc1.weight_scale = fc1.weight_scale[indices].contiguous()
            fc1.bias = fc1.bias[indices].contiguous()
            fc2.weight_int8 = fc2.weight_int8[:, indices].contiguous()
        else:
            fc1.weight = nn.Parameter(
                fc1.weight[indices].contiguous(), requires_grad=False
            )
            if fc1.bias is not None:
                fc1.bias = nn.Parameter(
                    fc1.bias[indices].contiguous(), requires_grad=False
                )
            fc2.weight = nn.Parameter(
                fc2.weight[:, indices].contiguous(), requires_grad=False
            )
        fc1.out_features = count
        fc2.in_features = count
