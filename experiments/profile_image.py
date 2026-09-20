"""GPU event breakdown of the current image patch, with no extra environment."""

import json
import statistics
import sys
from pathlib import Path
from functools import wraps
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.turing import apply_turing_patch
from huggingface_hub import hf_hub_download

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
processor = apply_turing_patch(Sam3Processor(model), compile=True)
im = Image.open(ROOT / "assets/images/truck.jpg").convert("RGB")
for _ in range(3):
    processor.set_text_prompt("truck", processor.set_image(im))
torch.cuda.synchronize()
pending = {}


def wrap(obj, method, label):
    original = getattr(obj, method)

    @wraps(original)
    def call(*args, **kwargs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        result = original(*args, **kwargs)
        b.record()
        pending.setdefault(label, []).append((a, b))
        return result

    setattr(obj, method, call)


for obj, method, label in [
    (processor, "set_image", "image_total"),
    (processor, "set_text_prompt", "prompt_total"),
    (model.backbone, "forward_image", "image_backbone"),
    (model.backbone.vision_backbone.trunk, "forward", "vision_trunk"),
    (model.backbone, "forward_text", "text_cache"),
    (model, "_encode_prompt", "encode_prompt"),
    (model, "_run_encoder", "fusion_encoder"),
    (model, "_run_decoder", "detection_decoder"),
    (model, "_run_segmentation_heads", "segmentation_head"),
]:
    wrap(obj, method, label)
for _ in range(7):
    processor.set_text_prompt("truck", processor.set_image(im))
torch.cuda.synchronize()
result = {
    "scope": "RTX3090, compiled FP16, warm truck image + cached truck prompt; nested CUDA event intervals",
    "milliseconds": {
        k: statistics.median(a.elapsed_time(b) for a, b in pairs)
        for k, pairs in pending.items()
    },
}
(ROOT / "experiments/results/profile_image.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
