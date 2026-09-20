"""Compare stock, serial patched and batched prompts with one image encoding."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/archive_reference/sam3_fp16_lab")]

import torch
from PIL import Image

from gpu_common import configure, environment, measure
from image_sweep import compare
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def run(args):
    configure()
    torch.set_num_threads(1)
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, load_from_HF=False
    ).eval()
    torch.set_num_threads(4)
    processor = Sam3Processor(model)
    if args.mode != "stock":
        from sam3.turing import apply_turing_patch, offload_text_encoder
        from sam3.turing_int8 import apply_int8_patch
        from sam3.turing_refinements import apply_image_refinements

        apply_turing_patch(processor, compile=True)
        apply_int8_patch(
            processor,
            attention_projections=True,
            fused_mlp=True,
            asymmetric_gelu=True,
            optimize_weight_scales=True,
        )
        offload_text_encoder(processor, trim_padding=True)
        apply_image_refinements(processor)
    from prompt_batch import PromptBatch

    batch = PromptBatch(processor) if args.mode.startswith("batch") else None
    size = int(args.mode[-1]) if batch else 1
    image = Image.open(ROOT / "assets/images/truck.jpg").convert("RGB")
    captions = ["truck", "wheel", "vehicle", "elephant"]

    def infer():
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if args.mode == "stock" else torch.float16,
            cache_enabled=args.mode == "stock",
        ):
            state = processor.set_image(image)
            outputs = []
            for start in range(0, len(captions), size):
                if batch:
                    outputs.extend(batch(captions[start : start + size], state))
                else:
                    current = processor.set_text_prompt(captions[start], state)
                    outputs.append(
                        {
                            k: current[k]
                            for k in ("masks", "masks_logits", "scores", "boxes")
                        }
                    )
            return outputs

    outputs, metrics = measure(infer, warmups=2, repetitions=9)
    outputs = [{k: v.detach().cpu() for k, v in output.items()} for output in outputs]
    directory = ROOT / "experiments/results"
    reference_path = directory / "prompt_batch_stock.pt"
    reference = (
        torch.load(reference_path, weights_only=True)
        if args.mode != "stock"
        else outputs
    )
    checks = {
        caption: compare(ref, out)
        for caption, ref, out in zip(captions, reference, outputs)
    }
    result = {
        "name": f"prompt_batch_{args.mode}",
        "config": vars(args),
        "environment": environment(),
        "scope": "One image encoding plus four prompts on truck.jpg, including preprocessing and text/mask processing; 2 warmups, 9 repeats, captions retained in text cache for patched modes. Allocator cache is not cleared after model construction; NVML can include unused build reserves.",
        "prompt_metrics": metrics,
        "checks": checks,
    }
    torch.save(outputs, directory / f"prompt_batch_{args.mode}.pt")
    (directory / f"prompt_batch_{args.mode}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        args.mode,
        "ms",
        metrics["median_wall_seconds"] * 1000,
        "allocated_GiB",
        metrics["peak_allocated_bytes"] / 2**30,
        "counts",
        [c["count"] for c in checks.values()],
        "changed_pixels",
        sum(c.get("changed_pixels", 0) for c in checks.values()),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["stock", "serial", "batch2", "batch4"])
    args = parser.parse_args()
    if args.mode:
        run(args)
        return
    for mode in ("stock", "serial", "batch2", "batch4"):
        print("RUN prompt_batch", mode, flush=True)
        path = ROOT / f"experiments/results/prompt_batch_{mode}.log"
        with path.open("w") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--checkpoint",
                    args.checkpoint,
                    "--mode",
                    mode,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=360)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code = 124
        if code:
            data = {"mode": mode, "exit_code": code, "error": path.read_text()[-4000:]}
            path.with_suffix(".failure.json").write_text(
                json.dumps(data, indent=2) + "\n"
            )
            print("FAILED", mode, data["error"][-1800:], flush=True)
            if mode == "stock":
                break
        else:
            print(path.read_text().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
