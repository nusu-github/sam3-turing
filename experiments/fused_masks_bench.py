"""A/B fused packing against PyTorch on learned logits, including boundary cases."""

import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/archive_reference/sam3_fp16_lab")]
from gpu_common import measure, environment
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.turing import apply_turing_patch
from sam3.turing_masks import resize_and_pack_masks, unpack_masks
from fused_mask_pack import fused_resize_and_pack

torch.set_num_threads(1)
model = build_sam3_image_model(
    checkpoint_path=hf_hub_download(
        "facebook/sam3", "sam3.pt", revision="3c879f39826c281e95690f02c7821c4de09afae7"
    ),
    load_from_HF=False,
).eval()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
processor = apply_turing_patch(Sam3Processor(model))
captured = []
hook = model.segmentation_head.register_forward_hook(
    lambda m, a, o: captured.append(o["pred_masks"].detach())
)
state = processor.set_text_prompt(
    "truck",
    processor.set_image(Image.open(ROOT / "assets/images/truck.jpg").convert("RGB")),
)
hook.remove()
low = captured.pop()[0]
del model, processor, state, captured, hook
gc.collect()
torch.cuda.empty_cache()
size = (2160, 3840)
result = {
    "environment": environment(),
    "scope": "200 learned FP16 mask logits -> 4K packed output; component only",
    "variants": {},
    "checks": [],
}
for label, fn in [
    ("chunk8", lambda: resize_and_pack_masks(low, size, fused=False)),
    ("fused", lambda: fused_resize_and_pack(low, size)),
]:
    output, metrics = measure(fn, warmups=2, repetitions=9)
    result["variants"][label] = metrics
    del output
    gc.collect()
    torch.cuda.empty_cache()
    print(
        label,
        round(metrics["median_wall_seconds"] * 1000, 3),
        round(metrics["peak_allocated_bytes"] / 2**30, 3),
        flush=True,
    )
# Compare all 200 masks with bounded memory, plus small and threshold-adjacent values.
changed = 0
for start in range(0, len(low), 8):
    x = low[start : start + 8]
    ref = resize_and_pack_masks(x, size, fused=False)
    candidate = fused_resize_and_pack(x, size)
    changed += int((unpack_masks(ref, size) != unpack_masks(candidate, size)).sum())
result["checks"].append(
    {
        "case": "learned4k",
        "changed_pixels": changed,
        "compared_pixels": len(low) * size[0] * size[1],
    }
)
for dtype in (torch.float16, torch.float32):
    for n, h, w, oh, ow, scale in [
        (3, 5, 7, 13, 17, 1),
        (2, 37, 51, 17, 23, 1),
        (2, 9, 7, 32, 19, 0.002),
        (2, 1, 1, 17, 19, 1),
    ]:
        x = torch.randn(n, h, w, device="cuda", dtype=dtype) * scale
        expected = (
            F.interpolate(
                x[:, None], (oh, ow), mode="bilinear", align_corners=False
            ).sigmoid()
            > 0.5
        )
        actual = unpack_masks(fused_resize_and_pack(x, (oh, ow)), (oh, ow))
        result["checks"].append(
            {
                "case": f"{dtype}_{n}x{h}x{w}_to_{oh}x{ow}_scale{scale}",
                "changed_pixels": int((expected != actual).sum()),
                "compared_pixels": actual.numel(),
            }
        )
result["empty_shape"] = list(fused_resize_and_pack(low[:0], (13, 17)).shape)
(ROOT / "experiments/results/fused_masks.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result["checks"], indent=2), flush=True)
