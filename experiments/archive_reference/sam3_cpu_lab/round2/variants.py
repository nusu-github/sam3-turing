"""Round 2 CPU research variants, restricted to SAM3 image-head inference.

No checkpoint, training, approximation of spatial neighborhoods, or public API
installation. Real algebra is preserved; floating-point order can change.
"""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_reference import import_source_modules, reassociate_head


def build_head(repo, dtype=torch.float32):
    modules = import_source_modules(repo)
    path = repo / "sam3/model_builder.py"
    tree = ast.parse(path.read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_create_segmentation_head")
    scope = {"PixelDecoder": modules["mask"].PixelDecoder,
             "UniversalSegmentationHead": modules["mask"].UniversalSegmentationHead,
             "MultiheadAttention": modules["misc"].MultiheadAttentionWrapper}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(path), "exec"), scope)
    return scope["_create_segmentation_head"]().eval().to(dtype), modules


def make_inputs(size=288, dtype=torch.float32, channels=256, queries=200):
    if size % 4:
        raise ValueError("Three levels with exact 2x nearest upsampling required")
    return dict(backbone_feats=[torch.randn(1, channels, size // f, size // f, dtype=dtype)
                                for f in (1, 2, 4)],
                obj_queries=torch.randn(6, 1, queries, channels, dtype=dtype),
                image_ids=torch.zeros(1, dtype=torch.long),
                encoder_hidden_states=torch.randn((size // 4) ** 2, 1, channels, dtype=dtype),
                prompt=torch.randn(9, 1, channels, dtype=dtype),
                prompt_mask=torch.zeros(1, 9, dtype=torch.bool))


@contextmanager
def no_fpn_clone(head):
    """Remove exactly the one clone expression from upstream _embed_pixels."""
    if head.training or torch.is_grad_enabled():
        raise ValueError("Inference only")
    original = head._embed_pixels.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    class RemoveClone(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "clone":
                self.count += 1
                if node.args or node.keywords:
                    raise ValueError("Unexpected clone options")
                return ast.copy_location(node.func.value, node)
            return node
    transform = RemoveClone()
    tree = ast.fix_missing_locations(transform.visit(tree))
    if transform.count != 1:
        raise ValueError("Upstream clone pattern changed")
    scope = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original), "exec"), scope)
    fn = types.MethodType(scope[original.__name__], head)
    with patch.object(head, "_embed_pixels", fn):
        yield


def mask_inplace(head, queries, pixels):
    e = head.mask_predictor.mask_embed(queries)
    ea = e @ head.instance_seg_head.weight[:, :, 0, 0]
    masks = torch.einsum("bqc,bchw->bqhw", ea, pixels)
    if head.instance_seg_head.bias is not None:
        masks.add_((e @ head.instance_seg_head.bias)[..., None, None])
    return masks


@contextmanager
def inplace_reassociation(head):
    if head.training or head.no_dec or head.aux_masks or torch.is_grad_enabled():
        raise ValueError("Inference with final query masks required")
    with patch.object(head.instance_seg_head, "forward", lambda x: x), \
         patch.object(head.mask_predictor, "forward", lambda q, p: mask_inplace(head, q, p)):
        yield


def folded_query_parameters(head, high_precision=True):
    """Fold the final query MLP affine layer into both EA and Eb, 257 outputs."""
    mlp = head.mask_predictor.mask_embed
    if mlp.residual or not isinstance(mlp.out_norm, torch.nn.Identity):
        raise ValueError("Cannot move across residual or normalization")
    final = mlp.layers[-1]
    dtype = torch.float64 if high_precision else final.weight.dtype
    a = head.instance_seg_head.weight[:, :, 0, 0].to(dtype)
    b = head.instance_seg_head.bias.to(dtype)
    w, bias = final.weight.to(dtype), final.bias.to(dtype)
    weight = torch.cat((a.T @ w, (b @ w)[None]), dim=0).to(final.weight.dtype)
    offset = torch.cat((a.T @ bias, (b @ bias).reshape(1))).to(final.weight.dtype)
    return weight, offset


@contextmanager
def static_reassociation(head):
    if head.training or head.no_dec or head.aux_masks or torch.is_grad_enabled():
        raise ValueError("Inference with final query masks required")
    weight, bias = folded_query_parameters(head)
    def masks(queries, pixels):
        mlp = head.mask_predictor.mask_embed
        hidden = queries
        for layer in mlp.layers[:-1]:
            hidden = mlp.drop(F.relu(layer(hidden)))
        embedding = F.linear(hidden, weight, bias)
        out = torch.einsum("bqc,bchw->bqhw", embedding[..., :-1], pixels)
        return out.add_(embedding[..., -1, None, None])
    with patch.object(head.instance_seg_head, "forward", lambda x: x), \
         patch.object(head.mask_predictor, "forward", masks):
        yield


def conv_row_tile(conv, current, previous, start, end):
    """Exact 2x nearest upsampling and the full 3x3 neighborhood (one-row halo)."""
    height = current.shape[-2]
    lo, hi = max(0, start - 1), min(height, end + 1)
    rows = torch.arange(lo, hi, device=previous.device) // 2
    upsampled = previous.index_select(-2, rows).repeat_interleave(2, dim=-1)
    merged = current[..., lo:hi, :] + upsampled
    convolved = conv(merged)
    return convolved[..., start - lo:end - lo, :]


def merge_moments(count, mean, m2, chunk, groups, stats_dtype=torch.float64):
    """Pairwise centered moments, population variance; no independent tile GN."""
    x = chunk.to(stats_dtype).reshape(chunk.shape[0], groups, -1)
    variance, local_mean = torch.var_mean(x, dim=-1, correction=0)
    n = x.shape[-1]
    if count == 0:
        return n, local_mean, variance * n
    delta = local_mean - mean
    total = count + n
    m2 = m2 + variance * n + delta.square() * (count * n / total)
    return total, mean + delta * (n / total), m2


def stream_head(head, kwargs, tile_rows=32, store_conv=False, stats_dtype=torch.float64):
    """Two-phase final GroupNorm and mask/semantic emission using real layers.

    store_conv=False recomputes the final convolution after global statistics.
    store_conv=True retains its full result to save that extra convolution.
    Both return all low-resolution masks and semantics, like the stock head.
    Restricted: batch one, 3 FPN levels, two 2x upsamplings, no aux/no_dec/presence.
    Moment accumulation precision is explicit; its rounding order differs from stock GN.
    """
    if head.training or torch.is_grad_enabled() or head.no_dec or head.aux_masks:
        raise ValueError("Inference with final query masks required")
    if head.presence_head is not None or head.pixel_decoder.shared_conv:
        raise ValueError("Unsupported head configuration")
    feats = kwargs["backbone_feats"]
    if len(feats) != 3 or feats[0].shape[0] != 1 or kwargs["image_ids"].tolist() != [0]:
        raise ValueError("One image with three FPN levels required")
    if tile_rows < 1 or any(feats[i].shape[-2] != 2 * feats[i + 1].shape[-2]
                           or feats[i].shape[-1] != 2 * feats[i + 1].shape[-1] for i in (0, 1)):
        raise ValueError("Positive tiles and exact 2x pyramid required")
    enc = kwargs["encoder_hidden_states"]
    if head.cross_attend_prompt is not None:
        normalized = head.cross_attn_norm(enc)
        enc = head.cross_attend_prompt(query=normalized, key=kwargs["prompt"],
                                       value=kwargs["prompt"],
                                       key_padding_mask=kwargs["prompt_mask"])[0] + enc
    encoder_map = enc.permute(1, 2, 0).reshape_as(feats[-1])
    decoder = head.pixel_decoder
    merged = feats[1] + F.interpolate(encoder_map, size=feats[1].shape[-2:], mode="nearest")
    previous = F.relu(decoder.norms[0](decoder.conv_layers[0](merged)))
    del merged, encoder_map, enc
    current = feats[0]
    conv, norm = decoder.conv_layers[1], decoder.norms[1]
    if conv.kernel_size != (3, 3) or conv.padding != (1, 1) or conv.stride != (1, 1):
        raise ValueError("Expected the stock final 3x3 convolution")
    height, width = current.shape[-2:]
    buffer = torch.empty_like(current) if store_conv else None
    count, mean, m2 = 0, None, None
    for start in range(0, height, tile_rows):
        end = min(height, start + tile_rows)
        chunk = conv_row_tile(conv, current, previous, start, end)
        count, mean, m2 = merge_moments(count, mean, m2, chunk, norm.num_groups, stats_dtype)
        if buffer is not None:
            buffer[..., start:end, :].copy_(chunk)
        del chunk
    channels = current.shape[1]
    repeats = channels // norm.num_groups
    mu = mean.repeat_interleave(repeats, dim=1)[..., None, None].to(current.dtype)
    invstd = torch.rsqrt(m2 / count + norm.eps).repeat_interleave(repeats, dim=1)[..., None, None].to(current.dtype)
    e = head.mask_predictor.mask_embed(kwargs["obj_queries"][-1])
    ea = e @ head.instance_seg_head.weight[:, :, 0, 0]
    eb = e @ head.instance_seg_head.bias
    masks = torch.empty(1, e.shape[1], height, width, dtype=current.dtype)
    semantic = torch.empty(1, 1, height, width, dtype=current.dtype)
    for start in range(0, height, tile_rows):
        end = min(height, start + tile_rows)
        chunk = buffer[..., start:end, :] if buffer is not None else conv_row_tile(conv, current, previous, start, end)
        normalized = (chunk - mu) * invstd
        normalized.mul_(norm.weight[None, :, None, None]).add_(norm.bias[None, :, None, None])
        F.relu_(normalized)
        tile_masks = torch.einsum("bqc,bchw->bqhw", ea, normalized)
        tile_masks.add_(eb[..., None, None])
        masks[..., start:end, :].copy_(tile_masks)
        semantic[..., start:end, :].copy_(head.semantic_seg_head(normalized))
    return dict(pred_masks=masks, semantic_seg=semantic, presence_logit=None)


@contextmanager
def choose_variant(head, mode):
    with ExitStack() as stack:
        if mode in ("no_clone", "combined", "inplace", "static"):
            stack.enter_context(no_fpn_clone(head))
        if mode in ("reassociate", "combined"):
            stack.enter_context(reassociate_head(head))
        elif mode == "inplace":
            stack.enter_context(inplace_reassociation(head))
        elif mode == "static":
            stack.enter_context(static_reassociation(head))
        yield
