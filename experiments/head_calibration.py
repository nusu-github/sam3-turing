"""Rank attention heads with one separate image, without gradients or fitting."""

import contextlib
import time
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from pruned_mlp import _weight


@torch.inference_mode()
def collect_head_stats(processor, eager_forward, samples=128):
    trunk = processor.model.backbone.vision_backbone.trunk
    stats = {}
    start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        for index, block in enumerate(trunk.blocks):
            heads = block.attn.num_heads

            def observe(projection, inputs, index=index, heads=heads):
                x = inputs[0].reshape(-1, projection.in_features).float()
                stride = max(1, x.shape[0] // samples)
                x = x[::stride][:samples]
                width = x.shape[-1]
                with torch.autocast("cuda", enabled=False):
                    by_head = x.reshape(-1, heads, width // heads).transpose(0, 1)
                    w = _weight(projection).reshape(-1, heads, width // heads)
                    contribution = torch.bmm(by_head, w.permute(1, 2, 0))
                    stats[index] = {
                        "energy": contribution.square().mean((1, 2)),
                        "variance": (contribution - contribution.mean(1, keepdim=True))
                        .square()
                        .mean((1, 2)),
                        "mean": x.mean(0),
                    }

            handle = block.attn.proj.register_forward_pre_hook(observe)
            stack.callback(handle.remove)
        stack.enter_context(patch.object(trunk, "forward", eager_forward))
        stack.enter_context(
            torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
        )
        source = Path(__file__).resolve().parents[1] / "assets/videos/0001/0.jpg"
        state = processor.set_image(Image.open(source).convert("RGB"))
        del state
    torch.cuda.synchronize()
    return stats, {
        "image": "assets/videos/0001/0.jpg",
        "max_token_samples_per_block": samples,
        "seconds": time.perf_counter() - start,
        "note": "Separate from the 3 evaluation images; one-image head contribution heuristic, not Hessian pruning or training.",
    }
