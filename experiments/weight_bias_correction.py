"""Weight-error bias correction from LayerNorm parameters, without image fitting."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from sam3.turing_int8 import DynamicInt8Linear


def input_means(model, scope):
    means = {}
    for block in model.backbone.vision_backbone.trunk.blocks:
        beta1, beta2 = block.norm1.bias.float(), block.norm2.bias.float()
        means[id(block.attn.qkv)] = beta1
        means[id(block.mlp.fc1)] = beta2
        if scope in ("mlp", "all"):
            fc1 = block.mlp.fc1
            w = fc1.weight.detach().float()
            mu = w @ beta2 + fc1.bias.float()
            variance = (w * block.norm2.weight.float()[None, :]).square().sum(1)
            denom = (1 + variance).sqrt()
            z = mu / denom
            means[id(block.mlp.fc2)] = (
                0.5 * mu * (1 + torch.erf(z / 2**0.5))
                + variance
                / denom
                * torch.exp(-0.5 * z.square())
                / (2 * torch.pi) ** 0.5
            )
        if scope == "all":
            qkv = block.attn.qkv
            width = qkv.out_features // 3
            means[id(block.attn.proj)] = (
                qkv.weight.detach().float()[2 * width :] @ beta1
                + qkv.bias[2 * width :].float()
            )
    return means


@contextmanager
def weight_bias_correction(model, scope="norm"):
    if scope not in ("norm", "mlp", "all"):
        raise ValueError("Expected norm, mlp or all scope")
    with torch.no_grad():
        means = input_means(model, scope)
    original = DynamicInt8Linear.__init__
    stats = {"layers": 0, "max_abs_correction": 0.0}

    def initialize(self, linear):
        original(self, linear)
        mean = means.get(id(linear))
        if mean is None:
            return
        with torch.no_grad():
            error = (
                linear.weight.detach().float()
                - self.weight_int8.float() * self.weight_scale[:, None]
            )
            correction = error @ mean
            self.bias = (self.bias.float() + correction).to(self.bias.dtype)
            stats["layers"] += 1
            stats["max_abs_correction"] = max(
                stats["max_abs_correction"], correction.abs().max().item()
            )

    with patch.object(DynamicInt8Linear, "__init__", initialize):
        yield stats
