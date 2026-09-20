"""Optional dynamic INT8 MLPs; trades a small output difference for speed/memory."""

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quantize(src, dst, scales, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(src + row * K + i, i < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-8) / 127.0
    q = libdevice.nearbyint(x / scale).to(tl.int8)
    tl.store(dst + row * K + i, q, i < K)
    tl.store(scales + row, scale)


class DynamicInt8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach().float()
        scales = w.abs().amax(1).clamp_min_(1.0e-8) / 127
        self.register_buffer(
            "weight_int8",
            (w / scales[:, None]).round().clamp_(-127, 127).to(torch.int8),
        )
        self.register_buffer("weight_scale", scales)
        self.register_buffer(
            "bias",
            (
                linear.bias.detach()
                if linear.bias is not None
                else torch.zeros(
                    self.out_features, device=w.device, dtype=torch.float16
                )
            ),
        )

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        rows = x.shape[0]
        if not rows:
            return x.new_empty((*shape[:-1], self.out_features), dtype=torch.float16)
        # Keep small/nonstandard resolutions compatible with INT8 GEMM tiles.
        padded_rows = max(32, triton.cdiv(rows, 16) * 16)
        if padded_rows != rows:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded_rows - rows))
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        _quantize[(x.shape[0],)](
            x, q, scales, self.in_features, triton.next_power_of_2(self.in_features)
        )
        mm = torch._int_mm(q, self.weight_int8.T)
        result = (mm.float() * scales[:, None]) * self.weight_scale[None, :] + self.bias
        return result[:rows].to(torch.float16).reshape(*shape[:-1], self.out_features)


def apply_int8_mlp_patch(processor, *, vision=True, text=False):
    """Quantize selected MLPs after apply_turing_patch, before first inference.

    Per-token activations and per-output-channel weights use signed INT8;
    GEMM accumulates in INT32 and returns FP16. No calibration or retraining.
    The original weights are released. Rebuild the model to undo the patch.
    vision=True targets the ViT MLPs; text=True additionally targets the text
    MLPs, reducing memory but potentially slowing new-prompt encoding.
    """
    if not getattr(processor, "_turing_patched", False):
        raise ValueError("Apply the Turing image patch first")
    if not vision and not text:
        raise ValueError("Select vision and/or text MLPs")
    model = processor.model
    if model.training:
        raise ValueError("INT8 MLPs are for inference only")
    targets = []
    if vision:
        targets.extend(
            (block.mlp, name)
            for block in model.backbone.vision_backbone.trunk.blocks
            for name in ("fc1", "fc2")
        )
    if text:
        if model.backbone.language_backbone is None:
            raise ValueError("The text encoder has already been released")
        targets.extend(
            (block.mlp, name)
            for block in model.backbone.language_backbone.encoder.transformer.resblocks
            for name in ("c_fc", "c_proj")
        )
    if any(
        not isinstance(getattr(parent, name), nn.Linear) for parent, name in targets
    ):
        raise ValueError("Expected unquantized Linear MLPs; apply each selection once")
    for parent, name in targets:
        setattr(parent, name, DynamicInt8Linear(getattr(parent, name)))
    return processor
