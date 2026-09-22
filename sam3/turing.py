"""Runtime patches for single-image SAM 3 inference on small CUDA GPUs.

Usage: processor = apply_turing_patch(Sam3Processor(model))
The checkpoint stays unchanged. Apply once to an eval model/processor pair;
weight conversion is permanent for that model instance. Rebuild to undo it.
"""

from collections import OrderedDict
from functools import wraps

import torch
import torch.nn.functional as F

from sam3.model import box_ops
from sam3.model.data_misc import interpolate, NestedTensor
from sam3.model.decoder import TransformerDecoderLayer
from sam3.model.model_misc import MultiheadAttention
from torch import nn


def _compile_with_owned_output(fn, mode):
    from torch.utils._pytree import tree_map

    compiled = torch.compile(fn, mode=mode)

    def call(*args, **kwargs):
        output = compiled(*args, **kwargs)
        # CUDA Graphs reuse output storage. Give subsequent compiled stages
        # tensors with independent storage before entering another graph.
        return tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output
        )

    return call


def _compile_text_grounding(model, mode):
    # Empty geometric inputs need no ROI scaling or pinned CPU allocation.
    geometry = model.geometry_encoder
    for method, field in (("_encode_points", "points"), ("_encode_boxes", "boxes")):
        original = getattr(geometry, method)

        def encode(*args, original=original, field=field, **kwargs):
            values = kwargs[field]
            if values.shape[0] == 0:
                empty = values.new_empty(
                    (0, values.shape[1], geometry.d_model),
                    dtype=geometry.label_embed.weight.dtype,
                )
                return empty, kwargs[field + "_mask"]
            return original(*args, **kwargs)

        setattr(geometry, method, encode)

    def compact(backbone_out, find_input, geometric_prompt):
        output = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        return {
            name: output[name]
            for name in (
                "pred_logits",
                "presence_logit_dec",
                "pred_boxes",
                "pred_masks",
            )
        }

    compiled = _compile_with_owned_output(compact, mode)

    def grounding(*, backbone_out, find_input, geometric_prompt, find_target):
        for name in ("box_embeddings", "point_embeddings", "mask_embeddings"):
            value = getattr(geometric_prompt, name)
            if value is not None and value.shape[0]:
                return model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    geometric_prompt=geometric_prompt,
                    find_target=find_target,
                )
        return compiled(backbone_out, find_input, geometric_prompt)

    return grounding


def _inference_call(model, fn):
    @wraps(fn)
    def call(*args, **kwargs):
        if model.training:
            raise RuntimeError("The Turing image patch is for eval/inference only")
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.float16, cache_enabled=False
        ):
            return fn(*args, **kwargs)

    return call


def apply_turing_patch(
    processor,
    *,
    text_cache_size=16,
    early_filter=False,
    compile=False,
    compile_text=False,
    packed_masks=False,
    mask_chunk_size=8,
):
    """Patch a Sam3Processor in place and return it.

    Supports one image at a time, arbitrary text and geometric box prompts,
    and the usual dense masks/probabilities/scores/boxes. SAM 1 interactivity,
    training, video and moving the model after patching are outside this patch.
    Text cache hits skip encoding; a new prompt still uses the text encoder.
    compile=True compiles the vision trunk and text-prompt grounding. Geometric
    prompts use separate compiled decoder/head stages. compile_text=True also
    compiles the text Transformer for new prompts. Smaller resolutions trade
    mask quality for speed.
    compile="max-autotune" enables additional kernel tuning.
    packed_masks=True returns masks_packed + mask_shape instead of dense masks
    and probabilities; decode individual rows with turing_masks.unpack_masks.
    """
    model = processor.model
    neck = model.backbone.vision_backbone
    head = model.segmentation_head
    if getattr(processor, "_turing_patched", False):
        raise ValueError("This processor has already been patched")
    if model.training or next(model.parameters()).device.type != "cuda":
        raise ValueError("Use a CUDA eval model")
    if (
        model.inst_interactive_predictor is not None
        or model.num_feature_levels != 1
        or neck.sam2_convs is not None
        or len(neck.convs) != 4
        or model.backbone.scalp != 1
        or head is None
        or not head.use_encoder_inputs
    ):
        raise ValueError(
            "Expected the standard SAM 3 image model without SAM 1 interactivity"
        )
    if text_cache_size < 0:
        raise ValueError("text_cache_size must be nonnegative")
    if mask_chunk_size < 1:
        raise ValueError("mask_chunk_size must be positive")
    if compile not in (False, True, "reduce-overhead", "max-autotune"):
        raise ValueError(
            "compile must be False, True, 'reduce-overhead' or 'max-autotune'"
        )
    if processor.resolution < 14 or processor.resolution % 14:
        raise ValueError("Image resolution must be a positive multiple of 14")

    # These decoder FFNs explicitly disable autocast and require FP32 weights.
    protected = {
        id(linear)
        for module in model.modules()
        if isinstance(module, TransformerDecoderLayer)
        for linear in (module.linear1, module.linear2)
    }
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            if id(module) not in protected:
                module.to(dtype=torch.float16)
        if isinstance(module, (nn.MultiheadAttention, MultiheadAttention)):
            module.half()
        if isinstance(module, nn.MultiheadAttention):
            original = module.forward

            def attention(*args, original=original, **kwargs):
                if len(args) < 5:
                    kwargs.setdefault("need_weights", False)
                return original(*args, **kwargs)

            module.forward = attention
    model.backbone.language_backbone.encoder.token_embedding.half()

    decoder = model.transformer.decoder
    side = processor.resolution // 14
    decoder.compilable_cord_cache = decoder._get_coords(side, side, processor.device)
    decoder.compilable_stored_size = (side, side)
    original_rpb = decoder._get_rpb_matrix

    def rpb(reference_boxes, feat_size):
        # Avoid reading CUDA scalar sizes in repeated cache comparisons/asserts.
        return original_rpb(reference_boxes, (side, side))

    decoder._get_rpb_matrix = rpb

    # The upstream fused MLP forces BF16. This path follows FP16 autocast.
    for block in neck.trunk.blocks:
        mlp = block.mlp

        def forward(x, mlp=mlp):
            return mlp.drop2(mlp.fc2(mlp.norm(mlp.drop1(mlp.act(mlp.fc1(x))))))

        mlp.forward = forward

    for block in neck.trunk.blocks:
        attention = block.attn
        if processor.resolution != 1008 and attention.input_size == (72, 72):
            side = processor.resolution // 14
            attention.input_size = (side, side)
            attention.freqs_cis = attention.compute_cis(
                end_x=side, end_y=side, scale_pos=attention.rope_pt_size[0] / side
            ).to(attention.freqs_cis.device)
        if compile:
            attention.use_rope_real = True
            attention.freqs_cis_real = attention.freqs_cis.real
            attention.freqs_cis_imag = attention.freqs_cis.imag

    def image_neck(tensor_list):
        x = neck.trunk(tensor_list)[-1]
        feats = [conv(x) for conv in neck.convs[:3]]
        pos = neck.position_encoding(feats[-1]).to(feats[-1].dtype)
        return feats, [None, None, pos], None, None

    neck.forward = image_neck
    model.backbone.scalp = 0
    neck.position_encoding.cache.clear()

    def embed_pixels(backbone_feats, image_ids, encoder_hidden_states):
        feats = [
            x.tensors if isinstance(x, NestedTensor) else x for x in backbone_feats
        ]
        if feats[0].shape[0] != 1:
            raise ValueError("The image patch supports batch size 1")
        # Replace a list entry; PixelDecoder does not mutate cached features.
        feats[-1] = encoder_hidden_states.permute(1, 2, 0).reshape(
            -1, *feats[-1].shape[1:]
        )
        return head.pixel_decoder(feats)

    head._embed_pixels = embed_pixels

    cache = OrderedDict()
    processor._turing_text_cache = cache
    original_text = model.backbone.forward_text

    def forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
        if (
            not text_cache_size
            or input_boxes is not None
            or additional_text is not None
        ):
            return original_text(captions, input_boxes, additional_text, device)
        key = (tuple(captions), str(device))
        if key not in cache:
            cache[key] = original_text(captions, device=device)
            if len(cache) > text_cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return dict(cache[key])

    model.backbone.forward_text = forward_text

    # Only filter queries during processor calls. Direct model calls retain the
    # full set of masks, and changing the processor threshold recomputes them.
    selecting = False
    original_heads = model._run_segmentation_heads

    def segmentation_heads(out, hs, **kwargs):
        if selecting:
            scores = (
                out["pred_logits"].sigmoid()
                * out["presence_logit_dec"].sigmoid().unsqueeze(1)
            ).squeeze(-1)
            hs = hs[:, :, scores[0] > processor.confidence_threshold]
        return original_heads(out=out, hs=hs, **kwargs)

    if early_filter:
        model._run_segmentation_heads = segmentation_heads

    compiled_grounding = None

    def grounding(state):
        nonlocal selecting
        selecting = early_filter
        try:
            outputs = (compiled_grounding or model.forward_grounding)(
                backbone_out=state["backbone_out"],
                find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )
        finally:
            selecting = False
        probs = (
            outputs["pred_logits"].sigmoid()
            * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        ).squeeze(-1)
        keep = probs > processor.confidence_threshold
        boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][keep])
        h, w = state["original_height"], state["original_width"]
        boxes = boxes * torch.tensor([w, h, w, h], device=processor.device)[None]
        masks = (
            outputs["pred_masks"][0] if early_filter else outputs["pred_masks"][keep]
        )
        if packed_masks:
            from sam3.turing_masks import resize_and_pack_masks

            state.update(
                masks_packed=resize_and_pack_masks(masks, (h, w), mask_chunk_size),
                mask_shape=(len(masks), 1, h, w),
                scores=probs[keep],
                boxes=boxes,
            )
            state.pop("masks", None)
            state.pop("masks_logits", None)
            return state
        masks = interpolate(
            masks.unsqueeze(1), (h, w), mode="bilinear", align_corners=False
        ).sigmoid_()
        state.update(
            masks_logits=masks, masks=masks > 0.5, scores=probs[keep], boxes=boxes
        )
        return state

    processor._forward_grounding = grounding
    original_reset = processor.reset_all_prompts

    def reset(state):
        original_reset(state)
        state.pop("masks_packed", None)
        state.pop("mask_shape", None)

    processor.reset_all_prompts = reset
    for method in (
        "set_image",
        "set_text_prompt",
        "add_geometric_prompt",
        "set_confidence_threshold",
    ):
        setattr(processor, method, _inference_call(model, getattr(processor, method)))

    def no_batch(*args, **kwargs):
        raise ValueError("Use set_image: the Turing image patch supports batch size 1")

    processor.set_image_batch = no_batch
    if compile:
        mode = compile if isinstance(compile, str) else "reduce-overhead"
        neck.trunk.forward = torch.compile(neck.trunk.forward, mode=mode)
        head.forward = torch.compile(head.forward, mode=mode)
        decoder.forward = _compile_with_owned_output(decoder.forward, mode)
        if not early_filter:
            compiled_grounding = _compile_text_grounding(model, mode)
    if compile_text:
        core = model.backbone.language_backbone.encoder.transformer
        core.forward = torch.compile(core.forward, mode="reduce-overhead")
    processor._turing_compile_text = bool(compile_text)
    processor._turing_compile_mode = mode if compile else None
    processor._turing_patched = True
    return processor


def freeze_text_prompts(processor, prompts):
    """Precompute a fixed vocabulary, then release the text encoder weights.

    Optional extra memory saving. Afterwards, unknown prompts raise ValueError.
    Include "visual" if geometric-only box prompts will be used.
    """
    if not getattr(processor, "_turing_patched", False):
        raise ValueError("Apply the Turing image patch first")
    prompts = tuple(dict.fromkeys(prompts))
    if not prompts or any(not isinstance(p, str) for p in prompts):
        raise ValueError("Provide a nonempty sequence of text prompts")
    backbone = processor.model.backbone
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.float16, cache_enabled=False
    ):
        cache = {
            tuple([p]): backbone.forward_text([p], device=processor.device)
            for p in prompts
        }

    def forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
        key = tuple(captions)
        if input_boxes is not None or additional_text is not None or key not in cache:
            raise ValueError("This processor only accepts its frozen text prompts")
        return dict(cache[key])

    backbone.forward_text = forward_text
    backbone.language_backbone = None
    processor._turing_text_cache.clear()
    # Compiled bound methods can form cycles around the released text module.
    import gc

    gc.collect()
    return processor


def offload_text_encoder(processor, *, int8_mlp=False, trim_padding=False):
    """Keep arbitrary text prompts while moving their encoder to CPU.

    Apply after the image patch, before inference. Cached prompts reuse GPU
    features; uncached prompts require CPU encoding and a small transfer.
    Requires compile_text=False and unquantized GPU text layers. Vision INT8 is
    compatible. Clear cached text when switching so all features use this path.
    CPU computation is FP32 by default. int8_mlp=True dynamically quantizes only
    the CPU text MLPs, trading output differences for faster uncached prompts.
    trim_padding=True skips CPU tokens after EOT and restores masked padding
    before returning features, preserving the GPU input shapes.
    """
    if not getattr(processor, "_turing_patched", False):
        raise ValueError("Apply the Turing image patch first")
    if getattr(processor, "_turing_cpu_text", False):
        raise ValueError("The text encoder is already on CPU")
    if getattr(processor, "_turing_compile_text", False):
        raise ValueError("CPU text offload requires compile_text=False")
    model = processor.model
    if model.training:
        raise ValueError("CPU text offload is for inference only")
    backbone = model.backbone
    text = backbone.language_backbone
    if text is None:
        raise ValueError("The text encoder has already been released")
    if any(hasattr(module, "weight_int8") for module in text.modules()):
        raise ValueError("CPU text offload cannot use CUDA INT8 text layers")
    if trim_padding and text.encoder.attn_mask is None:
        raise ValueError("Padding trim requires the causal text encoder")
    text.to(device="cpu", dtype=torch.float32)
    if int8_mlp:
        from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic

        for block in text.encoder.transformer.resblocks:
            quantize_dynamic(
                block.mlp, {nn.Linear: default_dynamic_qconfig}, inplace=True
            )
    if trim_padding:
        original_encoder = text.encoder.forward

        def encode(tokens):
            # CLIP EOT is the highest token ID; interior token IDs can be zero.
            length = int(tokens.argmax(1).max()) + 1
            pooled, memory = original_encoder(tokens[:, :length])
            padding = tokens.shape[1] - length
            if padding:
                memory = F.pad(memory, (0, 0, 0, padding))
                if pooled.ndim == 3:
                    pooled = F.pad(pooled, (0, 0, 0, padding))
            return pooled, memory

        text.encoder.forward = encode
    original = backbone._forward_text_no_ack_ckpt

    def forward(captions, input_boxes=None, additional_text=None, device="cuda"):
        with torch.autocast("cuda", enabled=False), torch.autocast(
            "cpu", enabled=False
        ):
            output = original(captions, input_boxes, additional_text, device="cpu")
        return {
            name: value.to(
                device=device,
                dtype=torch.float16 if value.is_floating_point() else value.dtype,
            )
            for name, value in output.items()
        }

    backbone._forward_text_no_ack_ckpt = forward
    processor._turing_text_cache.clear()
    processor._turing_cpu_text = True
    processor._turing_cpu_text_int8 = bool(int8_mlp)
    processor._turing_cpu_trim_padding = bool(trim_padding)
    return processor
