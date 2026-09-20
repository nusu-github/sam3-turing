"""Quick image A/B loop using the supplied reproduction archives.

Run from the checkout with the container's Python. No environment creation.
Each candidate runs in a fresh process to release GPU allocations between runs.
"""

import argparse
import ast
import contextlib
import faulthandler
import gc
import json
import inspect
import os
from pathlib import Path
import statistics
import signal
import subprocess
import sys
import time
import textwrap
import types
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extra_patches(model, processor, cfg, stack):
    if cfg.get("rope_real"):
        for block in model.backbone.vision_backbone.trunk.blocks:
            a = block.attn
            a.use_rope_real = True
            a.freqs_cis_real = a.freqs_cis.real
            a.freqs_cis_imag = a.freqs_cis.imag
    if cfg.get("resolution", 1008) != 1008:
        for block in model.backbone.vision_backbone.trunk.blocks:
            a = block.attn
            if a.input_size == (72, 72):
                side = cfg["resolution"] // 14
                a.input_size = (side, side)
                a.freqs_cis = a.compute_cis(
                    end_x=side, end_y=side, scale_pos=a.rope_pt_size[0] / side
                ).to(a.freqs_cis.device)
    if cfg.get("compile_mlp"):

        @torch.compile
        def compiled_mlp(x, w1, b1, w2, b2):
            return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)

        for block in model.backbone.vision_backbone.trunk.blocks:
            m = block.mlp
            stack.enter_context(
                patch.object(
                    m,
                    "forward",
                    lambda x, m=m: compiled_mlp(
                        x, m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias
                    ),
                )
            )
    if cfg.get("compile_trunk"):
        trunk = model.backbone.vision_backbone.trunk
        stack.enter_context(
            patch.object(
                trunk,
                "forward",
                torch.compile(
                    trunk.forward, mode=cfg.get("compile_policy", "reduce-overhead")
                ),
            )
        )
    if cfg.get("pixel_inplace"):
        decoder = model.segmentation_head.pixel_decoder

        def pixels(feats):
            previous = feats[-1]
            for i, current in enumerate(feats[:-1][::-1]):
                previous = F.interpolate(
                    previous, size=current.shape[-2:], mode=decoder.interpolation_mode
                )
                previous.add_(current)
                index = 0 if decoder.shared_conv else i
                previous = F.relu_(
                    decoder.norms[index](decoder.conv_layers[index](previous))
                )
            return previous

        stack.enter_context(patch.object(decoder, "forward", pixels))
    if cfg.get("mlp") in ("fused", "inplace", "tanh"):
        for block in model.backbone.vision_backbone.trunk.blocks:
            m = block.mlp

            def forward(x, m=m):
                if cfg["mlp"] == "fused":
                    dtype = torch.get_autocast_dtype("cuda")
                    flat = x.to(dtype).reshape(-1, x.shape[-1])
                    y = torch.ops.aten._addmm_activation(
                        m.fc1.bias.to(dtype),
                        flat,
                        m.fc1.weight.to(dtype).T,
                        use_gelu=True,
                    ).reshape(*x.shape[:-1], m.fc1.out_features)
                else:
                    y = m.fc1(x)
                    if cfg["mlp"] == "inplace":
                        y = torch.ops.aten.gelu_(y)
                    else:
                        y = F.gelu(y, approximate="tanh")
                return m.drop2(m.fc2(m.norm(m.drop1(y))))

            stack.enter_context(patch.object(m, "forward", forward))
    if cfg.get("channels_last"):
        model.to(memory_format=torch.channels_last)
    if cfg.get("text_cache"):
        original = model.backbone.forward_text
        cache = {}

        def text(captions, input_boxes=None, additional_text=None, device="cuda"):
            if input_boxes is not None or additional_text is not None:
                return original(captions, input_boxes, additional_text, device)
            key = (tuple(captions), str(device))
            if key not in cache:
                if len(cache) >= 16:
                    cache.pop(next(iter(cache)))
                cache[key] = original(captions, device=device)
            return dict(cache[key])

        stack.enter_context(patch.object(model.backbone, "forward_text", text))
    if cfg.get("output_inplace"):
        original = processor._forward_grounding
        fn = inspect.unwrap(getattr(original, "__func__", original))
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "sigmoid"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "interpolate"
            ):
                node.attr = "sigmoid_"
        scope = dict(fn.__globals__)
        if fn.__closure__:
            scope.update(
                {
                    key: cell.cell_contents
                    for key, cell in zip(fn.__code__.co_freevars, fn.__closure__)
                }
            )
        exec(
            compile(ast.fix_missing_locations(tree), "<inplace-output>", "exec"), scope
        )
        replacement = scope[fn.__name__]
        if hasattr(original, "__func__"):
            replacement = types.MethodType(replacement, processor)
        stack.enter_context(patch.object(processor, "_forward_grounding", replacement))
    if cfg.get("compile_head"):
        stack.enter_context(
            patch.object(
                model.segmentation_head,
                "forward",
                torch.compile(
                    model.segmentation_head.forward,
                    mode=cfg.get("compile_policy", "reduce-overhead"),
                ),
            )
        )


def compare(reference, actual):
    from scipy.optimize import linear_sum_assignment
    from torchvision.ops import box_iou

    nr, na = len(reference["scores"]), len(actual["scores"])
    result = {"reference_count": nr, "count": na}
    if not nr or not na:
        result["matched"] = 0
        return result
    rows, cols = linear_sum_assignment(
        -box_iou(reference["boxes"].float(), actual["boxes"].float()).numpy()
    )
    ious, changed, pixels, prob_abs = [], 0, 0, 0.0
    for r, a in zip(rows, cols):
        rm, am = reference["masks"][r], actual["masks"][a]
        union = (rm | am).sum().item()
        ious.append((rm & am).sum().item() / union if union else 1.0)
        changed += (rm != am).sum().item()
        pixels += rm.numel()
        prob_abs += (
            (reference["masks_logits"][r].float() - actual["masks_logits"][a].float())
            .abs()
            .sum()
            .item()
        )
    result.update(
        matched=len(rows),
        mean_mask_iou=statistics.mean(ious),
        min_mask_iou=min(ious),
        changed_pixels=changed,
        compared_pixels=pixels,
        probability_mae=prob_abs / pixels,
        score_max_abs=(
            reference["scores"][rows].float() - actual["scores"][cols].float()
        )
        .abs()
        .max()
        .item(),
        box_max_abs_px=(
            reference["boxes"][rows].float() - actual["boxes"][cols].float()
        )
        .abs()
        .max()
        .item(),
    )
    return result


def run(args):
    faulthandler.enable()
    cfg = json.loads(args.config)
    lab = Path(args.lab_root)
    fp = (
        lab / "SAM3_FP16_reproduction"
        if lab
        else ROOT / "experiments/archive_reference"
    )
    if not fp.exists():
        fp = ROOT / "experiments/archive_reference"
    sys.path[:0] = [
        str(fp / "sam3_fp16_lab"),
        str(fp / "sam3_claims_lab"),
        str(fp / "sam3_gpu_lab"),
    ]
    import claims_variants as cv
    from gpu_common import measure, environment
    from precision import execution, cast_linear_weights, mlp_arena
    from variants import static_reassociation, inplace_reassociation
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # This NGC build intermittently crashes in parallel CPU trunc_normal_.
    torch.set_num_threads(1)
    torch.manual_seed(4302)
    # TF32 is absent on the target Turing generation; use the same setting for all.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, load_from_HF=False
    ).eval()
    torch.set_num_threads(4)
    result = {
        "name": args.name,
        "config": cfg,
        "environment": environment(),
        "build_seconds": time.perf_counter() - start,
        "build_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    if cfg.get("half"):
        cast_linear_weights(model, preserve_precision=True)
    if cfg.get("freeze"):
        model.requires_grad_(False)
    processor = Sam3Processor(
        model, confidence_threshold=0.5, resolution=cfg.get("resolution", 1008)
    )
    if cfg.get("final"):
        from sam3.turing import apply_turing_patch, freeze_text_prompts

        apply_turing_patch(
            processor,
            text_cache_size=cfg.get("text_cache_size", 16),
            early_filter=cfg.get("early_filter", False),
            compile=cfg.get("compile", False),
        )
        if cfg.get("fixed_text"):
            freeze_text_prompts(
                processor,
                ["truck", "paper bag", "child", "wheel", "elephant", "visual"],
            )
    cases = [
        ("truck", "images/truck.jpg", "truck"),
        ("bag", "images/groceries.jpg", "paper bag"),
        ("child", "images/test_image.jpg", "child"),
        ("wheel", "images/truck.jpg", "wheel"),
        ("empty", "images/truck.jpg", "elephant"),
    ][: args.cases]
    loaded = [
        (n, Image.open(ROOT / "assets" / f).convert("RGB"), prompt)
        for n, f, prompt in cases
    ]
    with torch.inference_mode(), contextlib.ExitStack() as stack:
        mode = cfg.get("mode", "stock")
        stack.enter_context(cv.apply_variant(model, processor, mode))
        if cfg.get("fold") == "static":
            stack.enter_context(static_reassociation(model.segmentation_head))
        if cfg.get("fold") == "dynamic":
            stack.enter_context(inplace_reassociation(model.segmentation_head))
        if cfg.get("arena"):
            stack.enter_context(mlp_arena(model))
        extra_patches(model, processor, cfg, stack)

        def infer(im, text):
            if cfg.get("precision", "fp16") == "bf16":
                ctx = torch.autocast(
                    "cuda", dtype=torch.bfloat16, cache_enabled=cfg.get("cache", False)
                )
            else:
                ctx = execution(
                    "fp16", cfg.get("backend", "auto"), cfg.get("cache", False)
                )
            with ctx:
                return processor.set_text_prompt(text, processor.set_image(im))

        gc.collect()
        torch.cuda.empty_cache()
        state, metrics = measure(
            lambda: infer(loaded[0][1], loaded[0][2]), warmups=2, repetitions=args.reps
        )
        del state
        result["metrics"] = metrics
        result["checks"] = {}
        for name, im, prompt in loaded:
            state = infer(im, prompt)
            actual = {
                k: state[k].cpu() for k in ("masks", "masks_logits", "scores", "boxes")
            }
            del state
            check = {
                "count": len(actual["scores"]),
                "finite": all(bool(torch.isfinite(x).all()) for x in actual.values()),
            }
            for ref in ("stock", "fp16"):
                path = out / "references" / ref / (name + ".pt")
                if args.name == ref:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(actual, path)
                elif path.exists():
                    check["vs_" + ref] = compare(
                        torch.load(path, weights_only=True), actual
                    )
            result["checks"][name] = check
            (out / (args.name + ".json")).write_text(
                json.dumps(result, indent=2) + "\n"
            )
        print(
            json.dumps(
                {
                    "name": args.name,
                    "ms": round(metrics["median_wall_seconds"] * 1000, 2),
                    "allocated_GiB": round(metrics["peak_allocated_bytes"] / 2**30, 3),
                    "nvml_GiB": round(
                        metrics["nvml"]["sampled_device_used_peak_bytes"] / 2**30, 3
                    ),
                    "checks": result["checks"],
                }
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="stock")
    p.add_argument("--config", default='{"precision":"bf16","cache":true}')
    p.add_argument("--lab-root", default=str(ROOT / "experiments/archive_reference"))
    p.add_argument("--checkpoint")
    p.add_argument("--output", default=str(ROOT / "experiments/results"))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cases", type=int, default=3)
    p.add_argument("--sweep", type=Path)
    p.add_argument(
        "--timeout", type=int, default=180, help="Maximum seconds per candidate"
    )
    args = p.parse_args()
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download

        args.checkpoint = hf_hub_download(
            "facebook/sam3",
            "sam3.pt",
            revision="3c879f39826c281e95690f02c7821c4de09afae7",
        )
    if not args.sweep:
        run(args)
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in json.loads(args.sweep.read_text()).items():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--name",
            name,
            "--config",
            json.dumps(cfg),
            "--output",
            str(out),
            "--checkpoint",
            args.checkpoint,
            "--lab-root",
            args.lab_root,
            "--reps",
            str(args.reps),
            "--cases",
            str(args.cases),
        ]
        print("RUN", name, flush=True)
        with (out / (name + ".log")).open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        if code:
            error = (out / (name + ".log")).read_text()[-3000:]
            (out / (name + ".failure.json")).write_text(
                json.dumps(
                    {"name": name, "config": cfg, "exit_code": code, "error": error},
                    indent=2,
                )
                + "\n"
            )
            print("FAILED", name, error[-1800:], flush=True)
        else:
            data = json.loads((out / (name + ".json")).read_text())
            m = data["metrics"]
            diff = [
                v.get("vs_fp16", v.get("vs_stock", {})).get("changed_pixels", 0)
                for v in data["checks"].values()
            ]
            print(
                name,
                "ms",
                round(m["median_wall_seconds"] * 1000, 2),
                "allocated_GiB",
                round(m["peak_allocated_bytes"] / 2**30, 3),
                "nvml_GiB",
                round(m["nvml"]["sampled_device_used_peak_bytes"] / 2**30, 3),
                "changed_pixels",
                sum(diff),
                flush=True,
            )


if __name__ == "__main__":
    main()
