"""Local 6 GB GPU validation; each configuration runs in a fresh process.

Run with .venv/Scripts/python.exe experiments/local_turing_bench.py.
The FP16 reference replaces the upstream BF16-only fused MLP with linear+GELU,
exactly as the existing reproduction harness does. Stock is left unmodified.
"""

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/archive_reference/sam3_fp16_lab"))

CONFIGS = {
    "stock": {},
    "fp16_reference": {"fp16_adapter": True},
    "patched_eager": {"patch": True},
    "patched_cpu_text": {"patch": True, "cpu_text": True},
    "patched_compiled": {"patch": True, "compile": True},
    "patched_int8": {"patch": True, "int8": True, "cpu_text": True},
}


def worker(args):
    import torch
    from gpu_common import environment, measure
    from image_sweep import compare
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    cfg = CONFIGS[args.name]
    path = args.output / (args.name + ".json")
    result = {"name": args.name, "config": cfg, "status": "running"}

    def save(stage):
        result["stage"] = stage
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(stage, flush=True)

    torch.set_num_threads(1)
    torch.manual_seed(4302)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    result["environment"] = environment() | {"platform": platform.platform()}
    result["checkpoint"] = str(args.checkpoint)
    result["resolution"] = 1008
    result["confidence_threshold"] = 0.5
    result["outer_autocast"] = "fp16"
    result["timing_scope"] = (
        "set_image + set_text_prompt, including image preprocessing and transfer; "
        "excluding disk reads and model construction; repeated truck prompt "
        "hits the text cache in patched configurations"
    )
    result["baseline_note"] = (
        "fp16_reference uses the existing reproduction FP16 MLP adapter; "
        "stock keeps the upstream BF16-only fused MLP. Windows WDDM may spill "
        "allocations into shared RAM; absence of OOM does not prove VRAM residency."
    )
    try:
        save("build")
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        model = build_sam3_image_model(
            checkpoint_path=str(args.checkpoint), load_from_HF=False
        ).eval()
        result["build_seconds"] = time.perf_counter() - begin
        result["build_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        torch.set_num_threads(4)
        processor = Sam3Processor(model)
        save("patch")
        if cfg.get("patch"):
            from sam3.turing import apply_turing_patch, offload_text_encoder

            apply_turing_patch(processor, compile=cfg.get("compile", False))
            if cfg.get("int8"):
                from sam3.turing_int8 import apply_int8_patch

                apply_int8_patch(processor, attention_projections=True, fused_mlp=True)
            if cfg.get("cpu_text"):
                offload_text_encoder(processor, trim_padding=True)

        cases = [
            ("truck", "truck.jpg", "truck"),
            ("bag", "groceries.jpg", "paper bag"),
            ("child", "test_image.jpg", "child"),
            ("wheel", "truck.jpg", "wheel"),
            ("empty", "truck.jpg", "elephant"),
        ]
        loaded = [
            (n, Image.open(ROOT / "assets/images" / f).convert("RGB"), p)
            for n, f, p in cases
        ]

        def infer(im, prompt):
            from precision import execution

            ctx = (
                execution("fp16", "auto", True)
                if cfg.get("fp16_adapter")
                else torch.autocast(
                    "cuda", dtype=torch.float16, cache_enabled=not cfg.get("patch")
                )
            )
            with torch.inference_mode(), ctx:
                return processor.set_text_prompt(prompt, processor.set_image(im))

        gc.collect()
        torch.cuda.empty_cache()
        save("inference")
        state, metrics = measure(
            lambda: infer(loaded[0][1], loaded[0][2]), warmups=2, repetitions=args.reps
        )
        del state
        result["metrics"] = metrics
        save("quality")
        result["checks"] = {}
        refs = args.output / "references"
        refs.mkdir(exist_ok=True)
        for name, im, prompt in loaded:
            state = infer(im, prompt)
            actual = {
                k: state[k].cpu() for k in ("masks", "masks_logits", "scores", "boxes")
            }
            del state
            check = {
                "count": len(actual["scores"]),
                "finite": all(bool(torch.isfinite(v).all()) for v in actual.values()),
            }
            torch.save(actual, refs / f"{args.name}_{name}.pt")
            for ref in ("stock", "fp16_reference", "patched_eager"):
                source = refs / f"{ref}_{name}.pt"
                if ref != args.name and source.exists():
                    check["vs_" + ref] = compare(
                        torch.load(source, weights_only=True), actual
                    )
            result["checks"][name] = check
            save("quality_" + name)
        result["status"] = (
            "ok" if all(v["finite"] for v in result["checks"].values()) else "nonfinite"
        )
        save("complete")
    except Exception as exc:
        result["status"] = "oom" if isinstance(exc, torch.OutOfMemoryError) else "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save(result["stage"])
        traceback.print_exc()
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=CONFIGS)
    parser.add_argument("--variants", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/results/local_rtx2060"
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download

        args.checkpoint = Path(
            hf_hub_download(
                "facebook/sam3",
                "sam3.pt",
                revision="3c879f39826c281e95690f02c7821c4de09afae7",
            )
        )
    if args.name:
        return worker(args)
    for name in args.variants:
        print("RUN", name, flush=True)
        command = [
            sys.executable,
            __file__,
            "--name",
            name,
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(args.output),
            "--reps",
            str(args.reps),
        ]
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                    )
                else:
                    process.kill()
                process.wait()
                code = 124
        path = args.output / (name + ".json")
        data = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"name": name}
        )
        if code:
            if data.get("status") not in ("oom", "error"):
                data["status"] = "timeout" if code == 124 else "error"
            data["exit_code"] = code
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(
            name,
            data.get("status"),
            data.get("metrics", {}).get("median_wall_seconds"),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
