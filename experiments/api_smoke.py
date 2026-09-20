"""Exercise state reuse, geometric prompts and packed output on the public patch."""

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.turing import apply_turing_patch, freeze_text_prompts
from sam3.turing_masks import resize_and_pack_masks, unpack_masks

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=json.loads, default={})
parser.add_argument(
    "--output", type=Path, default=ROOT / "experiments/results/api_smoke_continued.json"
)
args = parser.parse_args()

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = build_sam3_image_model(
    checkpoint_path=hf_hub_download(
        "facebook/sam3", "sam3.pt", revision="3c879f39826c281e95690f02c7821c4de09afae7"
    ),
    load_from_HF=False,
).eval()
torch.set_num_threads(4)
p = apply_turing_patch(
    Sam3Processor(model),
    compile=True,
    compile_text=args.config.get(
        "public_compile_text", not args.config.get("public_cpu_text", False)
    ),
    packed_masks=True,
)
if args.config.get("public_int8"):
    from sam3.turing_int8 import apply_int8_patch

    apply_int8_patch(
        p,
        text=args.config.get("public_int8_text", False),
        attention_projections=args.config.get("public_int8_attention", False),
        fused_mlp=args.config.get("public_int8_fused", False),
        asymmetric_gelu=args.config.get("public_int8_asymmetric", False),
        weight_only=args.config.get("public_int8_weight_only", False),
        **(
            {"optimize_weight_scales": True}
            if args.config.get("public_int8_optimize_scales")
            else {}
        ),
    )
if args.config.get("public_int4"):
    from sam3.turing_int4 import apply_int4_patch

    apply_int4_patch(
        p,
        group_size=args.config.get("public_int4_group_size", 32),
        asymmetric=args.config.get("public_int4_asymmetric", False),
    )
    try:
        apply_int4_patch(p)
    except ValueError as exc:
        assert "unquantized" in str(exc)
    else:
        raise AssertionError("INT4 conversion accepted a second application")
if args.config.get("public_cpu_text"):
    from sam3.turing import offload_text_encoder
    from sam3.turing_int8 import apply_int8_patch

    offload_text_encoder(
        p,
        int8_mlp=args.config.get("public_cpu_text_int8", False),
        trim_padding=args.config.get("public_cpu_trim_padding", False),
    )
    assert all(
        x.device.type == "cpu" for x in model.backbone.language_backbone.parameters()
    )
    if args.config.get("public_cpu_text_int8"):
        assert all(
            module.weight().device.type == "cpu"
            and module.weight().dtype == torch.qint8
            for module in model.backbone.language_backbone.modules()
            if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        )
    try:
        apply_int8_patch(p, vision=False, text=True)
    except ValueError as exc:
        assert "CPU text" in str(exc)
    else:
        raise AssertionError("CPU text accepted incompatible INT8 conversion")
if args.config.get("public_refinements"):
    from sam3.turing_refinements import apply_image_refinements

    apply_image_refinements(p)
stack = ExitStack()
if args.config:
    from next_variants import apply_next_variants

    apply_next_variants(model, p, args.config, stack)
truck = Image.open(ROOT / "assets/images/truck.jpg").convert("RGB")
groceries = Image.open(ROOT / "assets/images/groceries.jpg").convert("RGB")
a = p.set_image(truck)
a = p.set_text_prompt("truck", a)
expected = a["masks_packed"].clone()
b = p.set_image(groceries)
b = p.set_text_prompt("paper bag", b)
a = p.set_text_prompt("truck", a)
assert torch.equal(
    expected, a["masks_packed"]
), "Old image state or cached text changed after another image"
result = {
    "config": args.config,
    "truck_count": len(a["scores"]),
    "bag_count": len(b["scores"]),
    "image_state_reuse": True,
}
if args.config.get("public_int4"):
    from sam3.turing_int4 import WeightOnlyInt4Linear

    result["vision_int4_linears"] = sum(
        isinstance(module, WeightOnlyInt4Linear) for module in model.modules()
    )
    assert result["vision_int4_linears"] == 128
    result["int4_reapply_rejected"] = True
if args.config.get("public_cpu_text_int8"):
    result["cpu_text_int8_linears"] = sum(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in model.backbone.language_backbone.modules()
    )
    assert result["cpu_text_int8_linears"] == 48
with torch.inference_mode(), torch.autocast(
    "cuda", dtype=torch.float16, cache_enabled=False
):
    direct = model.forward_grounding(
        backbone_out=a["backbone_out"],
        find_input=p.find_stage,
        find_target=None,
        geometric_prompt=a["geometric_prompt"],
    )
assert "prev_encoder_out" in direct and "encoder_hidden_states" in direct
result["direct_model_output_preserved"] = True
del direct
p.reset_all_prompts(a)
assert "masks_packed" not in a and "mask_shape" not in a
a = p.add_geometric_prompt([0.5, 0.5, 0.8, 0.8], True, a)
result["geometric_count"] = len(a["scores"])
a = p.set_confidence_threshold(1.0, a)
assert len(a["scores"]) == 0
assert unpack_masks(a["masks_packed"], a["mask_shape"][-2:]).shape[0] == 0
p.set_confidence_threshold(0.5)
before_freeze = torch.cuda.memory_allocated()
text_weight_bytes = sum(
    x.numel() * x.element_size() for x in model.backbone.language_backbone.parameters()
)
freeze_text_prompts(p, ["truck", "visual"])
result["text_parameter_bytes_before_freeze"] = text_weight_bytes
result["allocated_bytes_released_by_freeze"] = (
    before_freeze - torch.cuda.memory_allocated()
)
assert model.backbone.language_backbone is None
a = p.set_text_prompt("truck", p.set_image(truck))
result["frozen_count"] = len(a["scores"])
try:
    p.set_text_prompt("unregistered", a)
except ValueError:
    result["unknown_frozen_prompt_rejected"] = True
else:
    raise AssertionError("Unknown frozen prompt was accepted")
assert not torch.is_autocast_enabled("cuda")
# Noncontiguous source strides are supported by the fused pack kernel.
x = torch.randn(3, 17, 13, device="cuda", dtype=torch.float16).transpose(1, 2)
fast = resize_and_pack_masks(x, (31, 37))
old = resize_and_pack_masks(x, (31, 37), fused=False)
result["strided_pack_changed_pixels"] = int(
    (unpack_masks(fast, (31, 37)) != unpack_masks(old, (31, 37))).sum()
)
result["autocast_restored"] = True
result["threshold_empty"] = True
result["reset_clears_packed_output"] = True
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
stack.close()
