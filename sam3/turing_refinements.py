"""Optional image refinements: more FP16 intermediates and fewer head operations."""

import torch
import torch.nn.functional as F
from torch import nn

from sam3.model.decoder import TransformerDecoderLayer


@torch.no_grad()
def apply_image_refinements(processor):
    """Apply the measured image combination before the first inference.

    Requires apply_turing_patch(..., compile=True). Uses FP16 decoder FFNs and
    ViT block outputs and folds the mask projection. Output rounding differs
    from the base patch. The existing preprocessing and neck are preserved.
    Compatible with vision INT8, CPU text and packed masks. Returns processor.
    """
    mode = getattr(processor, "_turing_compile_mode", None)
    if not getattr(processor, "_turing_patched", False) or not mode:
        raise ValueError("Apply the compiled Turing image patch first")
    if getattr(processor, "_turing_refined", False):
        raise ValueError("Image refinements have already been applied")
    model = processor.model
    if model.training:
        raise ValueError("Image refinements are for inference only")
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
        original = block.forward

        def half_output(x, original=original):
            return original(x).to(torch.float16)

        block.forward = half_output

    processor._turing_refined = True
    return processor
