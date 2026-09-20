"""INT8 storage with ordinary FP16 linear kernels and transient decoded weights."""

import torch
import torch.nn.functional as F

from sam3.turing_int8 import DynamicInt8Linear


class DequantizedWeightLinear(DynamicInt8Linear):
    def forward(self, x):
        weight = (self.weight_int8.float() * self.weight_scale[:, None]).half()
        return F.linear(x, weight, self.bias)


def apply_dequant_weight(model, attention=False):
    for block in model.backbone.vision_backbone.trunk.blocks:
        block.mlp.fc1 = DequantizedWeightLinear(block.mlp.fc1)
        block.mlp.fc2 = DequantizedWeightLinear(block.mlp.fc2)
        if attention:
            block.attn.qkv = DequantizedWeightLinear(block.attn.qkv)
            block.attn.proj = DequantizedWeightLinear(block.attn.proj)
