"""Prune whole ViT attention heads using a weight-only output-energy heuristic."""

import torch

from pruned_mlp import _weight
from sam3.turing_int8 import DynamicInt8Linear


def apply_pruned_heads(model, keep, scope="global", compensate=False, calibration=None):
    selected_heads = {}
    for index, block in enumerate(model.backbone.vision_backbone.trunk.blocks):
        if scope == "global" and block.window_size:
            continue
        if scope == "window" and not block.window_size:
            continue
        attn = block.attn
        qkv, proj = attn.qkv, attn.proj
        if not isinstance(qkv, DynamicInt8Linear) or not isinstance(
            proj, DynamicInt8Linear
        ):
            raise ValueError("This candidate requires INT8 attention projections")
        heads = attn.num_heads
        if not 0 < keep < heads:
            raise ValueError("keep must be smaller than the original head count")
        width = qkv.out_features // 3
        dim = width // heads
        v_weight = _weight(qkv)[2 * width :]
        out_weight = _weight(proj)
        gamma, beta = block.norm1.weight.float(), block.norm1.bias.float()
        mean = v_weight @ beta + qkv.bias[2 * width :].float()
        variance = (v_weight * gamma[None, :]).square().sum(1)
        energy = variance if compensate else variance + mean.square()
        score = (energy * out_weight.square().sum(0)).reshape(heads, dim).sum(1)
        if calibration is not None:
            mean = calibration[index]["mean"]
            score = calibration[index]["variance" if compensate else "energy"]
        selected = score.topk(keep).indices.sort().values
        selected_heads[str(index)] = selected.tolist()
        channels = (
            selected[:, None] * dim + torch.arange(dim, device=selected.device)
        ).flatten()
        if compensate:
            removed_mean = mean.clone()
            removed_mean[channels] = 0
            proj.bias = (proj.bias.float() + out_weight @ removed_mean).to(
                proj.bias.dtype
            )
        qkv_rows = torch.cat([channels + i * width for i in range(3)])
        qkv.weight_int8 = qkv.weight_int8[qkv_rows].contiguous()
        qkv.weight_scale = qkv.weight_scale[qkv_rows].contiguous()
        qkv.bias = qkv.bias[qkv_rows].contiguous()
        qkv.out_features = 3 * keep * dim
        proj.weight_int8 = proj.weight_int8[:, channels].contiguous()
        proj.in_features = keep * dim
        attn.num_heads = keep
    return selected_heads
