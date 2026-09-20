"""Short SAM 3.1 video A/B runs in the existing container Python."""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/archive_reference/sam3_fp16_lab")]

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from gpu_common import DeviceMemorySampler, configure, environment
from sam3.model_builder import build_sam3_multiplex_video_predictor


def compare_frames(outputs, reference):
    frames = {}
    for index, out in outputs.items():
        ref = reference[index]
        masks = np.asarray(out["out_binary_masks"], dtype=bool)
        target = np.asarray(ref["out_binary_masks"], dtype=bool)
        pair_iou = np.zeros((len(masks), len(target)))
        for a, mask in enumerate(masks):
            for b, expected in enumerate(target):
                union = np.logical_or(mask, expected).sum()
                pair_iou[a, b] = np.logical_and(mask, expected).sum() / max(union, 1)
        rows, cols = linear_sum_assignment(-pair_iou)
        matched_iou = pair_iou[rows, cols]
        frames[str(index)] = {
            "count": len(masks),
            "reference_count": len(target),
            "matched": len(rows),
            "mean_mask_iou": float(matched_iou.mean()) if len(rows) else None,
            "min_mask_iou": float(matched_iou.min()) if len(rows) else None,
            "changed_pixels": int(
                sum(np.count_nonzero(masks[a] != target[b]) for a, b in zip(rows, cols))
            ),
            "compared_pixels": int(len(rows) * masks.shape[-2] * masks.shape[-1]),
            "score_max_abs": (
                float(
                    np.max(
                        np.abs(
                            np.asarray(out["out_probs"])[rows]
                            - np.asarray(ref["out_probs"])[cols]
                        )
                    )
                )
                if len(rows)
                else None
            ),
            "matched_id_agreement": int(
                sum(
                    out["out_obj_ids"][a] == ref["out_obj_ids"][b]
                    for a, b in zip(rows, cols)
                )
            ),
        }
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="video_stock_bf16_b1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--grounding-batch", type=int, default=1)
    parser.add_argument("--reference", default="video_stock_bf16_b1")
    parser.add_argument("--variant", default="stock")
    args = parser.parse_args()
    outdir = ROOT / "experiments/results"
    outdir.mkdir(exist_ok=True)
    frame_dir = ROOT / ".cache/video-probe" / str(args.frames)
    frame_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        (ROOT / "assets/videos/0001").glob("*.jpg"), key=lambda p: int(p.stem)
    )[: args.frames]
    for index, source in enumerate(sources):
        link = frame_dir / f"{index:05d}.jpg"
        if not link.exists():
            link.symlink_to(source)

    configure()
    torch.set_num_threads(1)
    started = time.perf_counter()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=args.checkpoint,
        use_fa3=False,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    # The builder enables TF32. Keep it off for both sides of this comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    predictor.model.batched_grounding_batch_size = args.grounding_batch
    data = {
        "name": args.name,
        "config": vars(args),
        "environment": environment(),
        "build_seconds": time.perf_counter() - started,
        "build_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": "SAM 3.1, first consecutive frames of assets/videos/0001, person. Uncompiled, FA3 off, TF32 off, frames on CPU, grounding batch explicitly configured. Timed add_prompt + forward propagation; frame loading/session construction excluded.",
    }
    if args.variant != "stock":
        from video_variants import apply_video_variant

        apply_video_variant(predictor, args.variant)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def run():
        response = predictor.handle_request(
            dict(
                type="start_session",
                resource_path=str(frame_dir),
                offload_video_to_cpu=True,
            )
        )
        session = response["session_id"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        response = predictor.handle_request(
            dict(type="add_prompt", session_id=session, frame_index=0, text="person")
        )
        outputs = {response["frame_index"]: response["outputs"]}
        for response in predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session,
                propagation_direction="forward",
                start_frame_index=0,
                max_frame_num_to_track=args.frames,
            )
        ):
            outputs[response["frame_index"]] = response["outputs"]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        predictor.handle_request(
            dict(
                type="close_session",
                session_id=session,
                run_gc_collect=False,
                clear_cache_threshold=0,
            )
        )
        return outputs, elapsed

    with torch.inference_mode(), DeviceMemorySampler() as sampler:
        outputs, cold = run()
        print(
            f"COLD {args.name}: {cold:.3f}s, counts {[len(v['out_obj_ids']) for v in outputs.values()]}",
            flush=True,
        )
        times = []
        for rep in range(args.reps):
            outputs, elapsed = run()
            times.append(elapsed)
            print(f"RUN {rep + 1}: {elapsed:.3f}s", flush=True)
    data["video_metrics"] = {
        "cold_seconds": cold,
        "wall_seconds": times,
        "median_wall_seconds": statistics.median(times),
        "median_ms_per_frame": statistics.median(times) * 1000 / len(outputs),
        "frames": len(outputs),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nvml": sampler.result(),
    }
    reference_path = outdir / f"{args.reference}.pt"
    reference = (
        torch.load(reference_path, weights_only=False)
        if reference_path.exists()
        else outputs
    )
    data["reference"] = args.reference if reference_path.exists() else "self"
    data["checks"] = compare_frames(outputs, reference)
    torch.save(outputs, outdir / f"{args.name}.pt")
    (outdir / f"{args.name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["video_metrics"]), flush=True)
    # Undo both constructor-entered contexts in reverse order.
    predictor.bf16_context.__exit__(None, None, None)
    predictor.model.tracker.bf16_context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
