"""Move the text encoder to CPU, retaining the existing GPU feature cache."""

from unittest.mock import patch

import torch


def apply_cpu_text(model, stack):
    backbone = model.backbone
    backbone.language_backbone.to(device="cpu", dtype=torch.float32)
    original = backbone._forward_text_no_ack_ckpt

    def forward(captions, input_boxes=None, additional_text=None, device="cuda"):
        with torch.autocast("cuda", enabled=False), torch.autocast(
            "cpu", enabled=False
        ):
            out = original(captions, input_boxes, additional_text, device="cpu")
        return {
            name: value.to(
                device=device,
                dtype=torch.float16 if value.is_floating_point() else value.dtype,
            )
            for name, value in out.items()
        }

    stack.enter_context(patch.object(backbone, "_forward_text_no_ack_ckpt", forward))
