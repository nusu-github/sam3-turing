"""Dynamic CPU INT8 MLPs for the already offloaded text encoder."""

import torch
from torch import nn


def apply_cpu_text_quantize(processor, per_channel=False):
    if not getattr(processor, "_turing_cpu_text", False):
        raise ValueError("Offload text to CPU before applying this candidate")
    from torch.ao.quantization import (
        default_dynamic_qconfig,
        per_channel_dynamic_qconfig,
        quantize_dynamic,
    )

    text = processor.model.backbone.language_backbone
    qconfig = per_channel_dynamic_qconfig if per_channel else default_dynamic_qconfig
    for block in text.encoder.transformer.resblocks:
        quantize_dynamic(block.mlp, {nn.Linear: qconfig}, inplace=True)
    processor._turing_text_cache.clear()
    quantized = list(
        m for m in text.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
    )
    stored = sum(p.numel() * p.element_size() for p in text.parameters())
    for module in quantized:
        weight = module.weight()
        bias = module.bias()
        stored += weight.numel() * weight.element_size()
        if bias is not None:
            stored += bias.numel() * bias.element_size()
    processor._cpu_text_quantization = {
        "engine": torch.backends.quantized.engine,
        "per_channel": per_channel,
        "linear_layers": len(quantized),
        "parameter_and_quantized_weight_bytes": stored,
        "note": "Includes ordinary parameters and decoded packed INT8 weights/biases, not process RSS or packing overhead.",
    }
