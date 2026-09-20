"""Quick CPU-only text timing within the existing patched model environment."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.turing import apply_turing_patch, offload_text_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(4302)
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, load_from_HF=False
    ).eval()
    processor = apply_turing_patch(Sam3Processor(model), text_cache_size=0)
    offload_text_encoder(processor, trim_padding=True)
    text = model.backbone.language_backbone
    prompts = {
        "short": ["truck"],
        "long": ["person wearing a red jacket and carrying a black bag near the truck"],
        "batch4": ["truck", "wheel", "vehicle", "elephant"],
    }
    references = {}
    results = []
    with torch.inference_mode():
        for precision in ("fp32", "int8_mlp"):
            if precision == "int8_mlp":
                from torch.ao.quantization import (
                    default_dynamic_qconfig,
                    quantize_dynamic,
                )

                for block in text.encoder.transformer.resblocks:
                    quantize_dynamic(
                        block.mlp, {nn.Linear: default_dynamic_qconfig}, inplace=True
                    )
            for threads in (4, 2, 8, 16, 1):
                torch.set_num_threads(threads)
                for name, captions in prompts.items():
                    for _ in range(2):
                        text(captions, device="cpu")
                    times = []
                    for _ in range(7):
                        start = time.perf_counter()
                        output = text(captions, device="cpu")
                        times.append((time.perf_counter() - start) * 1000)
                    if precision == "fp32" and threads == 4:
                        references[name] = tuple(x.clone() for x in output)
                    reference = references[name]
                    row = {
                        "precision": precision,
                        "threads": threads,
                        "prompt_group": name,
                        "wall_ms": times,
                        "median_ms": statistics.median(times),
                        "mask_equal_to_fp32_threads4": torch.equal(
                            output[0], reference[0]
                        ),
                        "memory_max_abs_vs_fp32_threads4": (output[1] - reference[1])
                        .abs()
                        .max()
                        .item(),
                        "finite": all(bool(torch.isfinite(x).all()) for x in output),
                    }
                    results.append(row)
                    print(
                        precision, threads, name, round(row["median_ms"], 3), flush=True
                    )
    data = {
        "scope": "CPU text encoder only, including tokenization and exact EOT padding trim; excludes GPU transfer, vision and grounding. 2 warmups, 7 repeats. Public FP16 weight conversion precedes CPU FP32 offload; INT8 uses the same public MLP quantization configuration.",
        "torch": torch.__version__,
        "cpu_model": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "prompts": prompts,
        "cpu_text_metrics": results,
    }
    (ROOT / "experiments/results/cpu_text_trimmed_thread_probe.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
