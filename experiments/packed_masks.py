"""Compare dense and packed 4K output using 200 learned mask logits."""

import gc
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/archive_reference/sam3_fp16_lab")]
from gpu_common import environment, measure
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.turing import apply_turing_patch
from sam3.turing_masks import resize_and_pack_masks, unpack_masks

torch.set_num_threads(1)
checkpoint = hf_hub_download(
    "facebook/sam3", "sam3.pt", revision="3c879f39826c281e95690f02c7821c4de09afae7"
)
model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
processor = apply_turing_patch(Sam3Processor(model), early_filter=False)
captured = []
hook = model.segmentation_head.register_forward_hook(
    lambda m, a, out: captured.append(out["pred_masks"].detach())
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
indices = [0, 1, 17, 50, 99, 123, 167, 199]
with torch.inference_mode():

    def dense():
        return (
            F.interpolate(
                low[:, None], size, mode="bilinear", align_corners=False
            ).sigmoid()
            > 0.5
        )

    ref, metrics = measure(dense, warmups=1, repetitions=5)
    expected = {i: ref[i].cpu() for i in indices}
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "environment": environment(),
        "scope": "Output component only: 200 learned logits resized to 4K; binary output; eight sampled masks checked",
        "dense": metrics,
        "packed": [],
    }
    for chunk in (1, 8):
        packed, metrics = measure(
            lambda: resize_and_pack_masks(low, size, chunk), warmups=1, repetitions=5
        )
        changed = sum(
            int((unpack_masks(packed[i : i + 1], size)[0].cpu() != expected[i]).sum())
            for i in indices
        )
        result["packed"].append(
            {
                "chunk_size": chunk,
                "metrics": metrics,
                "output_bytes": packed.numel(),
                "sampled_changed_pixels": changed,
                "sampled_pixels": len(indices) * size[0] * size[1],
            }
        )
        print(
            "PACKED",
            chunk,
            "ms",
            round(metrics["median_wall_seconds"] * 1000, 2),
            "GiB",
            round(metrics["peak_allocated_bytes"] / 2**30, 3),
            "changed",
            changed,
            flush=True,
        )
        del packed
        gc.collect()
        torch.cuda.empty_cache()
    # Exercise byte padding and zero detections as well as the 4K case.
    small = torch.randn(3, 5, 7, device="cuda", dtype=torch.float16)
    packed = resize_and_pack_masks(small, (13, 17), chunk_size=2)
    reference = (
        F.interpolate(
            small[:, None], (13, 17), mode="bilinear", align_corners=False
        ).sigmoid()
        > 0.5
    )
    result["padding_roundtrip"] = bool(
        torch.equal(reference, unpack_masks(packed, (13, 17)))
    )
    result["empty_shape"] = list(resize_and_pack_masks(small[:0], (13, 17)).shape)
(ROOT / "experiments/results/packed_masks.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    "DENSE ms",
    round(result["dense"]["median_wall_seconds"] * 1000, 2),
    "GiB",
    round(result["dense"]["peak_allocated_bytes"] / 2**30, 3),
    flush=True,
)
