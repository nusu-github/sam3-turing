"""Optional INT8 weight storage and projections for image inference."""

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


@triton.jit
def _gelu_quantize(
    mm, row_scales, weight_scales, bias, q, scales, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(mm + row * K + i, i < K, 0).to(tl.float32)
    x *= tl.load(row_scales + row)
    x = x * tl.load(weight_scales + i, i < K, 0) + tl.load(bias + i, i < K, 0)
    x = x.to(tl.float16).to(tl.float32)
    y = 0.5 * x * (1 + libdevice.erf(x * 0.7071067811865476))
    y = y.to(tl.float16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), 0), 1.0e-8) / 127.0
    quantized = libdevice.nearbyint(y / scale).to(tl.int8)
    tl.store(q + row * K + i, quantized, i < K)
    tl.store(scales + row, scale)


@triton.jit
def _gelu_asymmetric(
    mm,
    row_scales,
    weight_scales,
    bias,
    q,
    scales,
    zeros,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    x = tl.load(mm + row * K + i, i < K, 0).to(tl.float32)
    x *= tl.load(row_scales + row)
    x = x * tl.load(weight_scales + i, i < K, 0) + tl.load(bias + i, i < K, 0)
    x = x.to(tl.float16).to(tl.float32)
    y = (
        (0.5 * x * (1 + libdevice.erf(x * 0.7071067811865476)))
        .to(tl.float16)
        .to(tl.float32)
    )
    lo = tl.minimum(tl.min(tl.where(i < K, y, float("inf")), 0), 0.0)
    hi = tl.maximum(tl.max(tl.where(i < K, y, -float("inf")), 0), 0.0)
    scale = tl.maximum(hi - lo, 1.0e-8) / 255.0
    zero = tl.minimum(
        tl.maximum(libdevice.nearbyint(-lo / scale) - 128.0, -128.0), 127.0
    )
    quantized = tl.minimum(
        tl.maximum(libdevice.nearbyint(y / scale) + zero, -128.0), 127.0
    ).to(tl.int8)
    tl.store(q + row * K + i, quantized, i < K)
    tl.store(scales + row, scale)
    tl.store(zeros + row, zero.to(tl.int32))


def _fused_mlp(x, fc1, fc2, asymmetric=False):
    shape = x.shape
    x = x.reshape(-1, fc1.in_features).contiguous()
    rows = x.shape[0]
    if not rows:
        return x.new_empty((*shape[:-1], fc2.out_features), dtype=torch.float16)
    padded_rows = max(32, triton.cdiv(rows, 16) * 16)
    if padded_rows != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded_rows - rows))
    q1 = torch.empty_like(x, dtype=torch.int8)
    scale1 = torch.empty(padded_rows, device=x.device, dtype=torch.float32)
    _quantize[(padded_rows,)](
        x, q1, scale1, fc1.in_features, triton.next_power_of_2(fc1.in_features)
    )
    mm1 = torch._int_mm(q1, fc1.weight_int8.T)
    q2 = torch.empty_like(mm1, dtype=torch.int8)
    scale2 = torch.empty_like(scale1)
    if asymmetric:
        zero2 = torch.empty(padded_rows, device=x.device, dtype=torch.int32)
        _gelu_asymmetric[(padded_rows,)](
            mm1,
            scale1,
            fc1.weight_scale,
            fc1.bias,
            q2,
            scale2,
            zero2,
            fc2.in_features,
            triton.next_power_of_2(fc2.in_features),
            num_warps=4,
        )
    else:
        _gelu_quantize[(padded_rows,)](
            mm1,
            scale1,
            fc1.weight_scale,
            fc1.bias,
            q2,
            scale2,
            fc2.in_features,
            triton.next_power_of_2(fc2.in_features),
            num_warps=8,
        )
    mm2 = torch._int_mm(q2, fc2.weight_int8.T)
    if asymmetric:
        mm2 = mm2 - zero2[:, None] * fc2.weight_sum[None, :]
    y = (mm2.float() * scale2[:, None]) * fc2.weight_scale[None, :] + fc2.bias
    return y[:rows].to(torch.float16).reshape(*shape[:-1], fc2.out_features)


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


class WeightOnlyInt8Linear(DynamicInt8Linear):
    """Decode weights transiently; keep activations and the linear operation FP16."""

    def forward(self, x):
        weight = (self.weight_int8.float() * self.weight_scale[:, None]).half()
        return torch.nn.functional.linear(x, weight, self.bias)


@torch.no_grad()
def _optimize_weight_scales(quantized, original_weight):
    """Minimize each weight row's reconstruction error before inference."""
    w = original_weight.detach().float()
    initial = quantized.weight_scale.clone()
    best_scale = initial.clone()
    best_error = (w - quantized.weight_int8.float() * initial[:, None]).square().sum(1)
    for ratio in (1.0, 0.995, 0.99, 0.985, 0.975, 0.95, 0.925, 0.9, 0.85, 0.8):
        scale = initial * ratio
        q = (w / scale[:, None]).round().clamp(-127, 127)
        for _ in range(2):
            scale = ((w * q).sum(1) / q.square().sum(1).clamp_min(1)).clamp_min(1e-10)
            q = (w / scale[:, None]).round().clamp(-127, 127)
        error = (w - q * scale[:, None]).square().sum(1)
        better = error < best_error
        best_scale = torch.where(better, scale, best_scale)
        best_error = torch.minimum(best_error, error)
    quantized.weight_scale = best_scale
    quantized.weight_int8 = (
        (w / best_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    )


def apply_int8_patch(
    processor,
    *,
    vision=True,
    text=False,
    attention_projections=False,
    fused_mlp=False,
    asymmetric_gelu=False,
    weight_only=False,
    optimize_weight_scales=False,
):
    """Quantize selected MLPs after apply_turing_patch, before first inference.

    Per-token activations and per-output-channel weights use signed INT8;
    GEMM accumulates in INT32 and returns FP16. No calibration or retraining.
    The original weights are released. Rebuild the model to undo the patch.
    vision=True targets the ViT MLPs; text=True additionally targets the text
    MLPs, reducing memory but potentially slowing new-prompt encoding.
    attention_projections=True also targets the ViT QKV and output projections.
    The attention operation itself remains FP16.
    fused_mlp=True fuses the vision MLP's dequantization, GELU and requantization.
    asymmetric_gelu=True uses a per-token zero point for GELU activations. It
    requires fused_mlp=True and offers a different output-error tradeoff.
    weight_only=True instead decodes each layer's INT8 weights for a standard
    FP16 linear operation. Activations are not quantized. It cannot be combined
    with fused_mlp or asymmetric_gelu, which require dynamic INT8 activations.
    optimize_weight_scales=True searches per-row scales once using weight error
    only. It changes no inference operations and needs no calibration images.
    Output differences can improve or worsen depending on the selected path.
    """
    if not getattr(processor, "_turing_patched", False):
        raise ValueError("Apply the Turing image patch first")
    if not vision and not text and not attention_projections:
        raise ValueError("Select vision MLPs, text MLPs and/or attention projections")
    if text and getattr(processor, "_turing_cpu_text", False):
        raise ValueError("CPU text offload cannot use INT8 text layers")
    model = processor.model
    if model.training:
        raise ValueError("INT8 MLPs are for inference only")
    if asymmetric_gelu and not fused_mlp:
        raise ValueError("asymmetric_gelu requires fused_mlp=True")
    if weight_only and fused_mlp:
        raise ValueError("weight_only cannot be combined with fused_mlp")
    if fused_mlp:
        if not vision:
            raise ValueError("fused_mlp requires vision MLP quantization")
        for block in model.backbone.vision_backbone.trunk.blocks:
            if (
                not isinstance(block.mlp.norm, nn.Identity)
                or not isinstance(block.mlp.act, nn.GELU)
                or block.mlp.act.approximate != "none"
            ):
                raise ValueError("Fused MLPs require GELU without an intermediate norm")
    targets = []
    if vision:
        targets.extend(
            (block.mlp, name)
            for block in model.backbone.vision_backbone.trunk.blocks
            for name in ("fc1", "fc2")
        )
    if attention_projections:
        targets.extend(
            (block.attn, name)
            for block in model.backbone.vision_backbone.trunk.blocks
            for name in ("qkv", "proj")
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
        raise ValueError(
            "Expected unquantized Linear layers; apply each selection once"
        )
    linear_type = WeightOnlyInt8Linear if weight_only else DynamicInt8Linear
    for parent, name in targets:
        linear = getattr(parent, name)
        quantized = linear_type(linear)
        if optimize_weight_scales:
            _optimize_weight_scales(quantized, linear.weight)
        setattr(parent, name, quantized)
    if fused_mlp:
        for block in model.backbone.vision_backbone.trunk.blocks:
            mlp = block.mlp
            if asymmetric_gelu:
                mlp.fc2.register_buffer(
                    "weight_sum", mlp.fc2.weight_int8.sum(1, dtype=torch.int32)
                )

            def forward(x, mlp=mlp):
                return _fused_mlp(x, mlp.fc1, mlp.fc2, asymmetric_gelu)

            mlp.forward = forward
    return processor


def apply_int8_mlp_patch(processor, *, vision=True, text=False):
    """Compatibility entry point for quantizing only the selected MLPs."""
    return apply_int8_patch(processor, vision=vision, text=text)
