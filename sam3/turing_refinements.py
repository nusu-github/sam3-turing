"""Optional image refinements: more FP16 intermediates and fewer head operations."""

import torch
import torch.nn.functional as F
from torch import nn

from sam3.model.decoder import TransformerDecoderLayer
from sam3.model.vitdet import window_partition, window_unpartition


def _unpadded_window_block(block, side):
    attn, size = block.attn, block.window_size
    valid, _ = window_partition(
        torch.ones((1, side, side, 1), device=attn.qkv.bias.device, dtype=torch.bool),
        size,
    )
    block.register_buffer("_turing_window_valid", valid, persistent=False)

    def forward(x):
        shortcut = x
        x = block.norm1(x)
        height, width = x.shape[1:3]
        windows, padded = window_partition(attn.qkv(x), size)
        # In the original path, zero padding is projected to the QKV bias.
        windows = torch.where(
            block._turing_window_valid, windows, attn.qkv.bias.to(windows.dtype)
        )
        batches = windows.shape[0]
        parts = windows.reshape(batches, size * size, 3, attn.num_heads, -1)
        query, key, value = parts.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = attn._apply_rope(query, key)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batches, size, size, -1)
        attended = window_unpartition(attended, size, padded, (height, width))
        attended = attn.proj(attended)
        x = shortcut + block.dropout(block.drop_path(block.ls1(attended)))
        return (
            x + block.dropout(block.drop_path(block.ls2(block.mlp(block.norm2(x)))))
        ).half()

    block.forward = forward


@torch.no_grad()
def apply_image_refinements(processor, *, unpadded_projections=False):
    """Apply the measured image combination before the first inference.

    Requires apply_turing_patch(..., compile=True). Uses FP16 decoder FFNs and
    ViT block outputs and folds the mask projection. Output rounding differs
    from the base patch. The existing preprocessing and neck are preserved.
    Compatible with vision INT8, CPU text and packed masks. Returns processor.
    unpadded_projections=True moves window QKV/output projections across padding
    to skip padded rows at smaller resolutions. Attention still includes the
    original padding tokens. Windows with no padding are left unchanged.
    """
    mode = getattr(processor, "_turing_compile_mode", None)
    if not getattr(processor, "_turing_patched", False) or not mode:
        raise ValueError("Apply the compiled Turing image patch first")
    if getattr(processor, "_turing_refined", False):
        raise ValueError("Image refinements have already been applied")
    model = processor.model
    if model.training:
        raise ValueError("Image refinements are for inference only")
    side = processor.resolution // 14
    padding_blocks = [
        block
        for block in model.backbone.vision_backbone.trunk.blocks
        if unpadded_projections and block.window_size and side % block.window_size
    ]
    if any(
        block.attn.use_rel_pos
        or block.attn.cls_token
        or block.attn.use_fa3
        or block.attn.qkv.bias is None
        for block in padding_blocks
    ):
        raise ValueError("Unpadded projections require ordinary SAM3 window Attention")
    head = model.segmentation_head
    predictor = head.mask_predictor
    mlp = predictor.mask_embed
    projection = head.instance_seg_head
    if (
        head.no_dec
        or head.aux_masks
        or mlp.residual
        or not isinstance(mlp.out_norm, nn.Identity)
        or projection.kernel_size != (1, 1)
        or projection.bias is None
        or mlp.layers[-1].bias is None
    ):
        raise ValueError("Expected the standard final-query image mask head")

    final = mlp.layers[-1]
    a = projection.weight[:, :, 0, 0].double()
    b = projection.bias.double()
    w, bias = final.weight.double(), final.bias.double()
    folded_weight = torch.cat((a.T @ w, (b @ w)[None]), dim=0).to(final.weight.dtype)
    folded_bias = torch.cat((a.T @ bias, (b @ bias).reshape(1))).to(final.weight.dtype)
    predictor.register_buffer("_turing_query_weight", folded_weight, persistent=False)
    predictor.register_buffer("_turing_query_bias", folded_bias, persistent=False)

    def masks(queries, pixels):
        hidden = queries
        for layer in mlp.layers[:-1]:
            hidden = mlp.drop(F.relu(layer(hidden)))
        embedding = F.linear(
            hidden, predictor._turing_query_weight, predictor._turing_query_bias
        )
        out = torch.einsum("bqc,bchw->bqhw", embedding[..., :-1], pixels)
        return out.add_(embedding[..., -1, None, None])

    projection.forward = lambda x: x
    predictor.forward = masks
    for module in model.modules():
        if isinstance(module, TransformerDecoderLayer):
            module.linear1.half()
            module.linear2.half()

            def ffn(x, module=module):
                y = module.linear2(module.activation(module.linear1(x)))
                return module.norm3(x + y)

            module.forward_ffn = ffn
    neck = model.backbone.vision_backbone
    for block in neck.trunk.blocks:
        if block in padding_blocks:
            _unpadded_window_block(block, side)
            continue
        original = block.forward

        def half_output(x, original=original):
            return original(x).to(torch.float16)

        block.forward = half_output

    processor._turing_refined = True
    return processor
