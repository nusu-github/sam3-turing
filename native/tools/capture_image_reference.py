"""Capture unmodified SAM3 eager reference tensors for native model migration."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def cpu_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tensors(item) for key, item in value.items() if isinstance(item, (torch.Tensor, dict, list, tuple, int, float, bool, str, type(None)))}
    if isinstance(value, (list, tuple)):
        return [cpu_tensors(item) for item in value]
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    raise TypeError(type(value))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(713)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    start = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False, enable_inst_interactivity=True, compile=False)
    processor = Sam3Processor(model)
    original_grounding = model.forward_grounding
    raw = {}

    def capture(*a, **kw):
        output = original_grounding(*a, **kw)
        raw.clear()
        raw.update(cpu_tensors({key: output[key] for key in ["pred_boxes", "pred_logits", "pred_masks", "presence_logit_dec"]}))
        return output

    model.forward_grounding = capture
    image = Image.open(args.image).convert("RGB")
    image.save(args.output / "input.png")
    state = processor.set_image(image)
    torch.save(cpu_tensors(state["backbone_out"]), args.output / "vision.pt")
    cases = []

    def save_case(name, inputs):
        tensors = {"raw": dict(raw), "result": cpu_tensors({key: state[key] for key in ["boxes", "scores", "masks", "masks_logits"]})}
        torch.save(tensors, args.output / (name + ".pt"))
        cases.append({"name": name, "inputs": inputs, "detections": len(state["scores"]), "raw_queries": raw["pred_logits"].shape[-2], "shapes": {key: list(value.shape) for key, value in raw.items()}})
        print(json.dumps(cases[-1]), flush=True)

    for i, prompt in enumerate(args.prompt):
        processor.reset_all_prompts(state)
        processor.set_text_prompt(prompt, state)
        text_keys = ["language_features", "language_mask", "language_embeds"]
        torch.save(cpu_tensors({key: state["backbone_out"][key] for key in text_keys}), args.output / f"text-{i}.pt")
        save_case(f"text-{i}-grounding", {"text": prompt})
    processor.reset_all_prompts(state)
    positive = [0.5, 0.5, 0.8, 0.8]
    negative = [0.1, 0.1, 0.1, 0.1]
    processor.add_geometric_prompt(positive, True, state)
    save_case("box-positive", {"boxes": [positive], "labels": [True]})
    processor.add_geometric_prompt(negative, False, state)
    save_case("box-positive-negative", {"boxes": [positive, negative], "labels": [True, False]})
    processor.set_confidence_threshold(0.25, state)
    save_case("box-threshold", {"boxes": [positive, negative], "labels": [True, False], "threshold": 0.25})
    metadata = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "float32", "tf32": False,
        "resolution": processor.resolution, "num_queries": model.transformer.decoder.num_queries,
        "image": str(args.image), "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "cases": cases, "elapsed_seconds": time.perf_counter() - start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "scope": "Image detector/text/geometry reference only; interactive point/mask, batch and video fixtures remain to be captured.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
