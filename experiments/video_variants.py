"""Experimental SAM 3.1 FP16 monkeypatches; no image-only neck pruning."""

import ast
import importlib
import inspect
import textwrap
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sam3.model.decoder import TransformerDecoderLayer
from sam3.model.model_misc import MultiheadAttention


class _HalfDtype(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and node.attr == "bfloat16"
        ):
            node.attr = "float16"
        return node


def _replace_explicit_bf16():
    # Prototype only: replace literal BF16 casts/decorators in a few upstream
    # video methods, preserving the rest of their current implementation.
    changed = []
    for name in (
        "sam3_base_predictor",
        "sam3_multiplex_tracking",
        "sam3_multiplex_detector",
        "video_tracking_multiplex_demo",
    ):
        module = importlib.import_module("sam3.model." + name)
        for cls in vars(module).values():
            if not inspect.isclass(cls) or cls.__module__ != module.__name__:
                continue
            for method, function in list(vars(cls).items()):
                if not inspect.isfunction(function) or method in (
                    "__init__",
                    "warm_up_compilation",
                ):
                    continue
                source = textwrap.dedent(inspect.getsource(function))
                if "torch.bfloat16" not in source:
                    continue
                if inspect.unwrap(function).__code__.co_freevars:
                    raise ValueError(
                        f"Cannot source-patch closure {cls.__name__}.{method}"
                    )
                tree = ast.fix_missing_locations(_HalfDtype().visit(ast.parse(source)))
                local = {}
                exec(
                    compile(tree, inspect.getfile(function), "exec"),
                    vars(module),
                    local,
                )
                setattr(cls, method, local[method])
                changed.append(f"{cls.__name__}.{method}")
    return changed


@torch.no_grad()
def apply_video_variant(predictor, variant):
    if variant not in (
        "fp16",
        "fp16_efficient",
        "fp16_int8",
        "fp16_int8_compile",
        "fp16_int8_cpu_compile",
        "fp16_int8_cpu_compile_efficient",
        "fp16_int8_cpu_trim_compile",
        "fp16_int8_cpu_trim_compile_efficient",
    ):
        raise ValueError(variant)
    model = predictor.model
    predictor.bf16_context.__exit__(None, None, None)
    model.tracker.bf16_context.__exit__(None, None, None)
    changed = _replace_explicit_bf16()
    protected = {
        id(linear)
        for module in model.modules()
        if isinstance(module, TransformerDecoderLayer)
        for linear in (module.linear1, module.linear2)
    }
    model.requires_grad_(False)
    for module in model.modules():
        if (
            isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))
            and id(module) not in protected
        ):
            module.half()
        if isinstance(module, (nn.MultiheadAttention, MultiheadAttention)):
            module.half()
        if isinstance(module, nn.MultiheadAttention):
            original = module.forward

            def attention(*args, original=original, **kwargs):
                if len(args) < 5:
                    kwargs.setdefault("need_weights", False)
                return original(*args, **kwargs)

            module.forward = attention
    backbone = model.detector.backbone
    backbone.language_backbone.encoder.token_embedding.half()
    for block in backbone.vision_backbone.trunk.blocks:
        mlp = block.mlp

        def forward(x, mlp=mlp):
            return mlp.drop2(mlp.fc2(mlp.norm(mlp.drop1(mlp.act(mlp.fc1(x))))))

        mlp.forward = forward
    for owner in (model.tracker, predictor):
        owner.bf16_context = torch.autocast(
            "cuda", dtype=torch.float16, cache_enabled=False
        )
        owner.bf16_context.__enter__()
    if "int8" in variant:
        from sam3.turing_int8 import (
            DynamicInt8Linear,
            _fused_mlp,
            _optimize_weight_scales,
        )

        for block in backbone.vision_backbone.trunk.blocks:
            for parent, name in (
                (block.attn, "qkv"),
                (block.attn, "proj"),
                (block.mlp, "fc1"),
                (block.mlp, "fc2"),
            ):
                linear = getattr(parent, name)
                quantized = DynamicInt8Linear(linear)
                _optimize_weight_scales(quantized, linear.weight)
                setattr(parent, name, quantized)
            block.mlp.fc2.register_buffer(
                "weight_sum", block.mlp.fc2.weight_int8.sum(1, dtype=torch.int32)
            )

            def int8_forward(x, mlp=block.mlp):
                return _fused_mlp(x, mlp.fc1, mlp.fc2, asymmetric=True)

            block.mlp.forward = int8_forward
    if "cpu" in variant:
        from torch.utils._pytree import tree_map

        device = next(backbone.vision_backbone.parameters()).device
        backbone.language_backbone.cpu().float()
        if "trim" in variant:
            encoder = backbone.language_backbone.encoder
            original_encoder = encoder.forward

            def encode(tokens):
                length = int(tokens.argmax(1).max()) + 1
                pooled, memory = original_encoder(tokens[:, :length])
                padding = tokens.shape[1] - length
                if padding:
                    memory = F.pad(memory, (0, 0, 0, padding))
                    if pooled.ndim == 3:
                        pooled = F.pad(pooled, (0, 0, 0, padding))
                return pooled, memory

            encoder.forward = encode
        original_text = backbone._forward_text_no_ack_ckpt
        predictor._cpu_text_calls = []

        def cpu_text(captions, input_boxes=None, additional_text=None, device=device):
            start = time.perf_counter()
            with torch.autocast("cuda", enabled=False), torch.autocast(
                "cpu", enabled=False
            ):
                outputs = original_text(
                    captions, input_boxes, additional_text, device="cpu"
                )
            predictor._cpu_text_calls.append(
                {
                    "captions": len(captions),
                    "additional": len(additional_text or []),
                    "cpu_seconds": time.perf_counter() - start,
                }
            )
            return tree_map(
                lambda x: (
                    x.to(
                        device=device,
                        dtype=torch.float16 if x.is_floating_point() else x.dtype,
                    )
                    if isinstance(x, torch.Tensor)
                    else x
                ),
                outputs,
            )

        backbone._forward_text_no_ack_ckpt = cpu_text
    if "compile" in variant:
        from sam3.turing import _compile_with_owned_output

        trunk = backbone.vision_backbone.trunk
        trunk.forward = _compile_with_owned_output(trunk.forward, "reduce-overhead")
    if "efficient" in variant:
        import sam3.model.decoder as decoder

        original = F.scaled_dot_product_attention

        def efficient(query, *args, **kwargs):
            if query.device.type != "cuda":
                return original(query, *args, **kwargs)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return original(query, *args, **kwargs)

        F.scaled_dot_product_attention = efficient
        decoder.sdpa_kernel = lambda *args, **kwargs: sdpa_kernel(
            SDPBackend.EFFICIENT_ATTENTION
        )
    print("FP16 source substitutions:", ", ".join(changed), flush=True)
