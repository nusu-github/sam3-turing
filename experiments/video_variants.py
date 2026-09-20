"""Experimental SAM 3.1 FP16 monkeypatches; no image-only neck pruning."""

import ast
import importlib
import inspect
import textwrap

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
    if variant not in ("fp16", "fp16_efficient"):
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
    if variant == "fp16_efficient":
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
