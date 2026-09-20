"""Keep selected QKV input channels out of activation quantization."""

import contextlib
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

from sam3.turing_int8 import DynamicInt8Linear


@torch.inference_mode()
def apply_calibrated_input_split(processor, eager_forward, channels=16, scope="all"):
    if scope not in ("all", "global") or channels not in (8, 16, 32):
        raise ValueError("Expected all/global QKV scope and 8/16/32 channels")
    trunk = processor.model.backbone.vision_backbone.trunk
    selected = {
        i: block.attn.qkv
        for i, block in enumerate(trunk.blocks)
        if scope == "all" or not block.window_size
    }
    if not all(isinstance(layer, DynamicInt8Linear) for layer in selected.values()):
        raise ValueError("Input splitting requires INT8 QKV projections")
    maxima = {}
    start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        for index, layer in selected.items():

            def observe(module, inputs, index=index):
                x = inputs[0].reshape(-1, module.in_features)
                stride = max(1, x.shape[0] // 512)
                maxima[index] = x[::stride][:512].float().abs().amax(0)

            handle = layer.register_forward_pre_hook(observe)
            stack.callback(handle.remove)
        stack.enter_context(patch.object(trunk, "forward", eager_forward))
        source = Path(__file__).resolve().parents[1] / "assets/videos/0001/0.jpg"
        state = processor.set_image(Image.open(source).convert("RGB"))
        del state
    metadata = {}
    for index, layer in selected.items():
        indices = maxima[index].topk(channels).indices.sort().values
        mask = torch.ones(layer.in_features, device=indices.device, dtype=torch.float16)
        mask[indices] = 0
        # Decode the already quantized weights, so this experiment isolates
        # activation splitting rather than also restoring original FP16 weights.
        weight = (
            layer.weight_int8[:, indices].float() * layer.weight_scale[:, None]
        ).half()
        layer.register_buffer("_split_indices", indices, persistent=False)
        layer.register_buffer("_split_mask", mask, persistent=False)
        layer.register_buffer("_split_weight", weight, persistent=False)
        original = layer.forward

        def forward(x, layer=layer, original=original):
            retained = F.linear(
                x.index_select(-1, layer._split_indices), layer._split_weight
            )
            bulk = original(x * layer._split_mask)
            return bulk + retained

        layer.forward = forward
        metadata[index] = indices.tolist()
    torch.cuda.synchronize()
    return {
        "image": "assets/videos/0001/0.jpg",
        "max_token_samples_per_layer": 512,
        "channels": channels,
        "scope": scope,
        "seconds": time.perf_counter() - start,
        "selected_indices": metadata,
        "note": "One separate image chooses fixed large-activation channels. Those activations use FP16 with decoded INT8 weights; other activations use the existing dynamic INT8 path. Inspired by mixed precision decomposition, not an LLM.int8 reproduction.",
    }
