"""Round 6+ candidates. Kept separate from the accepted public patch."""

from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch import nn


def _compile_stage(fn, cfg):
    compiled = torch.compile(fn, mode=cfg.get("stage_mode", "reduce-overhead"))
    if not cfg.get("clone_stage_outputs"):
        return compiled
    from torch.utils._pytree import tree_map

    def call(*args, **kwargs):
        output = compiled(*args, **kwargs)
        return tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output
        )

    return call


def apply_next_variants(model, processor, cfg, stack):
    if cfg.get("fast_normalize"):
        from image_input import apply_fast_normalize

        apply_fast_normalize(processor, cfg["fast_normalize"], stack)
    if cfg.get("compile_image_neck"):
        neck = model.backbone.vision_backbone
        side = processor.resolution // 14
        neck.position_encoding(torch.empty(1, 1, side, side, device=processor.device))
        stack.enter_context(
            patch.object(
                neck,
                "forward",
                _compile_stage(neck.forward, {**cfg, "clone_stage_outputs": True}),
            )
        )
    if cfg.get("weight_only_int8"):
        from weight_only_int8 import apply_weight_only_int8

        apply_weight_only_int8(model, cfg.get("weight_only_tile", [64, 64, 32, 4]))
    if cfg.get("int8_layernorm"):
        from norm_int8 import apply_norm_int8

        apply_norm_int8(model, cfg.get("int8_layernorm_warps", 4))
    if cfg.get("fused_int8_mlp"):
        from fused_int8_mlp import apply_fused_int8_mlp

        apply_fused_int8_mlp(
            model,
            stack,
            cfg.get("fused_int8_warps", 4),
            cfg.get("fused_int8_tanh", False),
        )
    if cfg.get("int8_vision_attention"):
        from sam3.turing_int8 import DynamicInt8Linear

        for i, block in enumerate(model.backbone.vision_backbone.trunk.blocks):
            if cfg.get("int8_attention_windows_only") and not block.window_size:
                continue
            for name in cfg["int8_vision_attention"]:
                setattr(block.attn, name, DynamicInt8Linear(getattr(block.attn, name)))
    if cfg.get("int8_fusion_ffn"):
        from sam3.turing_int8 import DynamicInt8Linear

        for module in model.transformer.encoder.modules():
            if isinstance(getattr(module, "linear1", None), nn.Linear):
                module.linear1 = DynamicInt8Linear(module.linear1)
                module.linear2 = DynamicInt8Linear(module.linear2)
    if cfg.get("balanced_int8_alpha"):
        from balanced_int8 import apply_balanced_int8

        apply_balanced_int8(
            model, cfg["balanced_int8_alpha"], cfg.get("balanced_int8_scope", "both")
        )
    if cfg.get("adaptive_mlp"):
        from adaptive_mlp import apply_adaptive_mlp

        apply_adaptive_mlp(
            model,
            cfg["adaptive_mlp"],
            cfg.get("adaptive_mlp_layers", list(range(32))),
            stack,
        )
    if cfg.get("empty_geometry"):
        geometry = model.geometry_encoder
        for method, field in (("_encode_points", "points"), ("_encode_boxes", "boxes")):
            original = getattr(geometry, method)

            def encode(*args, original=original, field=field, **kwargs):
                values = kwargs[field]
                if values.shape[0] == 0:
                    empty = torch.empty(
                        (0, values.shape[1], geometry.d_model),
                        device=values.device,
                        dtype=geometry.label_embed.weight.dtype,
                    )
                    return empty, kwargs[field + "_mask"]
                return original(*args, **kwargs)

            stack.enter_context(patch.object(geometry, method, encode))
    if cfg.get("feature_views"):

        def features(backbone_out, img_ids):
            feats = backbone_out["backbone_fpn"][-1:]
            pos = backbone_out["vision_pos_enc"][-1:]
            assert feats[0].shape[0] == 1
            return (
                backbone_out,
                [x.flatten(2).permute(2, 0, 1) for x in feats],
                [x.flatten(2).permute(2, 0, 1) for x in pos],
                [x.shape[-2:] for x in pos],
            )

        stack.enter_context(patch.object(model, "_get_img_feats", features))
    if cfg.get("half_block_output"):
        for block in model.backbone.vision_backbone.trunk.blocks:
            original_block = block.forward

            def half_output(x, original_block=original_block):
                return original_block(x).to(torch.float16)

            stack.enter_context(patch.object(block, "forward", half_output))
    if cfg.get("int8_text_mlp"):
        from int8_linear import DynamicInt8Linear

        for block in model.backbone.language_backbone.encoder.transformer.resblocks:
            block.mlp.c_fc = DynamicInt8Linear(block.mlp.c_fc)
            block.mlp.c_proj = DynamicInt8Linear(block.mlp.c_proj)
    if cfg.get("skip_vit_blocks"):
        for index in cfg["skip_vit_blocks"]:
            stack.enter_context(
                patch.object(
                    model.backbone.vision_backbone.trunk.blocks[index],
                    "forward",
                    lambda x: x,
                )
            )
    if cfg.get("pooled_mlp"):
        selected = cfg.get("pooled_mlp_layers", list(range(32)))
        factor = cfg["pooled_mlp"]
        for i, block in enumerate(model.backbone.vision_backbone.trunk.blocks):
            if i not in selected:
                continue
            mlp = block.mlp
            original_mlp = mlp.forward

            def pooled(x, original_mlp=original_mlp):
                b, h, w, c = x.shape
                small = F.avg_pool2d(x.permute(0, 3, 1, 2), factor, factor)
                small = original_mlp(small.permute(0, 2, 3, 1))
                return F.interpolate(
                    small.permute(0, 3, 1, 2), size=(h, w), mode="nearest"
                ).permute(0, 2, 3, 1)

            stack.enter_context(patch.object(mlp, "forward", pooled))
    if cfg.get("static_rpb"):
        decoder = model.transformer.decoder
        original_rpb = decoder._get_rpb_matrix
        side = processor.resolution // 14
        decoder.compilable_cord_cache = decoder._get_coords(
            side, side, processor.device
        )
        decoder.compilable_stored_size = (side, side)

        def rpb(reference_boxes, feat_size):
            return original_rpb(reference_boxes, (side, side))

        stack.enter_context(patch.object(decoder, "_get_rpb_matrix", rpb))
    if cfg.get("fixed_feature_meta"):
        encoder = model.transformer.encoder
        side = processor.resolution // 14
        shapes = torch.tensor([[side, side]], dtype=torch.long, device=processor.device)
        starts = torch.zeros(1, dtype=torch.long, device=processor.device)
        ratios = torch.ones(1, 1, 2, device=processor.device)

        def prepare(srcs, masks, pos_embeds):
            assert len(srcs) == 1 and srcs[0].shape[0] == 1
            assert masks is None or masks[0] is None
            return (
                srcs[0].flatten(2).transpose(1, 2),
                None,
                pos_embeds[0].flatten(2).transpose(1, 2),
                starts,
                ratios,
                shapes,
            )

        stack.enter_context(
            patch.object(encoder, "_prepare_multilevel_features", prepare)
        )
    if cfg.get("int8_mlp"):
        from int8_linear import DynamicInt8Linear

        selected = cfg.get("int8_layers", list(range(32)))
        for i, block in enumerate(model.backbone.vision_backbone.trunk.blocks):
            if i in selected:
                block.mlp.fc1 = DynamicInt8Linear(block.mlp.fc1)
                if cfg["int8_mlp"] == "both":
                    block.mlp.fc2 = DynamicInt8Linear(block.mlp.fc2)
    if cfg.get("trim_text"):
        encoder = model.backbone.language_backbone
        original_tokenizer = encoder.tokenizer

        def tokenize(text, context_length):
            tokens = original_tokenizer(text, context_length=context_length)
            # CLIP's EOT is the largest token ID; an interior token can be 0.
            length = int(tokens.argmax(1).max()) + 1
            length = min(context_length, ((length + 7) // 8) * 8)
            return tokens[:, :length]

        stack.enter_context(patch.object(encoder, "tokenizer", tokenize))
    if cfg.get("embedding_half"):
        model.backbone.language_backbone.encoder.token_embedding.half()
    if cfg.get("last_scores"):
        original_scores = model._update_scores_and_boxes

        def scores(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=None,
            **kwargs,
        ):
            return original_scores(
                out,
                hs[-1:],
                reference_boxes[-1:],
                prompt,
                prompt_mask,
                dec_presence_out=(
                    dec_presence_out[-1:] if dec_presence_out is not None else None
                ),
                **kwargs,
            )

        stack.enter_context(patch.object(model, "_update_scores_and_boxes", scores))
    if cfg.get("compile_rpb"):
        decoder = model.transformer.decoder
        stack.enter_context(
            patch.object(
                decoder,
                "_get_rpb_matrix",
                torch.compile(
                    decoder._get_rpb_matrix, mode=cfg.get("stage_mode", "default")
                ),
            )
        )
    if cfg.get("native_mha"):
        from sam3.model.model_misc import MultiheadAttention

        for module in model.modules():
            if not isinstance(module, MultiheadAttention):
                continue

            def forward(
                query,
                key,
                value,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                average_attn_weights=True,
                attn_bias=None,
                module=module,
            ):
                assert attn_bias is None and module._qkv_same_embed_dim
                if module.batch_first:
                    if key is value:
                        if query is key:
                            query = key = value = query.transpose(0, 1)
                        else:
                            query, key = query.transpose(0, 1), key.transpose(0, 1)
                            value = key
                    else:
                        query, key, value = [
                            x.transpose(0, 1) for x in (query, key, value)
                        ]
                output, weights = F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    module.embed_dim,
                    module.num_heads,
                    module.in_proj_weight,
                    module.in_proj_bias,
                    module.bias_k,
                    module.bias_v,
                    module.add_zero_attn,
                    0.0,
                    module.out_proj.weight,
                    module.out_proj.bias,
                    training=False,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    average_attn_weights=average_attn_weights,
                )
                return (
                    output.transpose(0, 1) if module.batch_first else output
                ), weights

            stack.enter_context(patch.object(module, "forward", forward))
    if cfg.get("cpu_resize"):
        from PIL import Image

        original_image = processor.set_image

        def set_image(image, state=None):
            if not isinstance(image, Image.Image):
                return original_image(image, state)
            width, height = image.size
            image = image.resize(
                (processor.resolution, processor.resolution), Image.Resampling.BILINEAR
            )
            state = original_image(image, state)
            state.update(original_width=width, original_height=height)
            return state

        stack.enter_context(patch.object(processor, "set_image", set_image))
    if cfg.get("mha_no_weights"):
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                original = module.forward

                def forward(*args, original=original, **kwargs):
                    kwargs["need_weights"] = False
                    return original(*args, **kwargs)

                stack.enter_context(patch.object(module, "forward", forward))
    if cfg.get("mha_half"):
        from sam3.model.model_misc import MultiheadAttention

        for module in model.modules():
            if isinstance(module, (nn.MultiheadAttention, MultiheadAttention)):
                module.half()
    if cfg.get("decoder_half_ffn"):
        from sam3.model.decoder import TransformerDecoderLayer

        for module in model.modules():
            if isinstance(module, TransformerDecoderLayer):
                module.linear1.half()
                module.linear2.half()

                def ffn(x, module=module):
                    y = module.linear2(module.activation(module.linear1(x)))
                    return module.norm3(x + y)

                stack.enter_context(patch.object(module, "forward_ffn", ffn))
    if cfg.get("vision_half_residual"):
        model.backbone.vision_backbone.trunk.half()
    if cfg.get("compile_text_core"):
        core = model.backbone.language_backbone.encoder.transformer
        stack.enter_context(
            patch.object(core, "forward", _compile_stage(core.forward, cfg))
        )
    if cfg.get("compile_encoder"):
        encoder = model.transformer.encoder
        stack.enter_context(
            patch.object(
                encoder,
                "forward",
                _compile_stage(encoder.forward, cfg),
            )
        )
    if cfg.get("compile_decoder"):
        decoder = model.transformer.decoder
        stack.enter_context(
            patch.object(
                decoder,
                "forward",
                _compile_stage(decoder.forward, cfg),
            )
        )
    if cfg.get("compile_text"):
        encoder = model.backbone.language_backbone
        stack.enter_context(
            patch.object(
                encoder,
                "forward",
                _compile_stage(encoder.forward, cfg),
            )
        )
    if cfg.get("compile_grounding"):
        stack.enter_context(
            patch.object(
                model,
                "forward_grounding",
                torch.compile(
                    model.forward_grounding, mode=cfg.get("stage_mode", "default")
                ),
            )
        )
    if cfg.get("compact_grounding"):
        original_grounding = model.forward_grounding

        def compact(backbone_out, find_input, geometric_prompt):
            output = original_grounding(
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

        compiled_compact = torch.compile(
            compact, mode=cfg.get("compact_grounding_mode", "reduce-overhead")
        )

        def grounding(
            backbone_out, find_input, find_target, geometric_prompt, **kwargs
        ):
            for name in ("box_embeddings", "point_embeddings", "mask_embeddings"):
                value = getattr(geometric_prompt, name)
                if value is not None and value.shape[0]:
                    return original_grounding(
                        backbone_out,
                        find_input,
                        find_target,
                        geometric_prompt,
                        **kwargs,
                    )
            output = compiled_compact(backbone_out, find_input, geometric_prompt)
            return {name: value.clone() for name, value in output.items()}

        stack.enter_context(patch.object(model, "forward_grounding", grounding))
    if cfg.get("kv_pool"):
        stride = cfg["kv_pool"]
        for index, block in enumerate(model.backbone.vision_backbone.trunk.blocks):
            if block.window_size or index not in cfg.get(
                "kv_pool_layers", [7, 15, 23, 31]
            ):
                continue
            attn = block.attn

            def forward(x, attn=attn):
                b, h, w, c = x.shape
                qkv = attn.qkv(x).reshape(b, h * w, 3, attn.num_heads, -1)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                q, k = attn._apply_rope(q, k)

                def pool(a):
                    d = a.shape[-1]
                    a = a.transpose(-1, -2).reshape(b * attn.num_heads, d, h, w)
                    return (
                        F.avg_pool2d(a, stride, stride)
                        .reshape(b, attn.num_heads, d, -1)
                        .transpose(-1, -2)
                        .contiguous()
                    )

                y = F.scaled_dot_product_attention(q, pool(k), pool(v))
                return attn.proj(y.transpose(1, 2).reshape(b, h, w, c))

            stack.enter_context(patch.object(attn, "forward", forward))
