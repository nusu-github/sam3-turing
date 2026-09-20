"""Keep FP16 activations for selected layers while storing all weights as INT8."""

from unittest.mock import patch

from sam3.turing_int8 import WeightOnlyInt8Linear


def apply_mixed_weight_only(model, stack, selection):
    choices = {
        "qkv_global",
        "qkv_first8",
        "qkv_all",
        "proj_all",
        "mlp_first4",
        "mlp_last4",
    }
    if selection not in choices:
        raise ValueError(selection)
    blocks = model.backbone.vision_backbone.trunk.blocks
    for index, block in enumerate(blocks):
        if selection == "qkv_global" and block.window_size:
            continue
        if selection == "qkv_first8" and index >= 8:
            continue
        if selection == "mlp_first4" and index >= 4:
            continue
        if selection == "mlp_last4" and index < len(blocks) - 4:
            continue
        if selection.startswith("mlp"):
            layers = (block.mlp.fc1, block.mlp.fc2)

            def mlp_forward(x, mlp=block.mlp):
                return mlp.drop2(mlp.fc2(mlp.norm(mlp.drop1(mlp.act(mlp.fc1(x))))))

            stack.enter_context(patch.object(block.mlp, "forward", mlp_forward))
        else:
            layers = (block.attn.proj if selection == "proj_all" else block.attn.qkv,)
        for layer in layers:

            def forward(x, layer=layer):
                return WeightOnlyInt8Linear.forward(layer, x)

            stack.enter_context(patch.object(layer, "forward", forward))
