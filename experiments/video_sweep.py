"""Run bounded SAM 3.1 probes sequentially on the one available GPU."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--variants", nargs="+", default=["stock", "fp16", "fp16_efficient"]
    )
    args = parser.parse_args()
    output = ROOT / "experiments/results"
    for variant in args.variants:
        name = "video_stock_bf16_b1" if variant == "stock" else f"video_{variant}_b1"
        command = [
            sys.executable,
            str(ROOT / "experiments/video_probe.py"),
            "--checkpoint",
            args.checkpoint,
            "--name",
            name,
            "--variant",
            variant,
            "--reps",
            str(args.reps),
        ]
        print("RUN", name, flush=True)
        logpath = output / f"{name}.log"
        with logpath.open("w") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code = 124
        if code:
            data = {
                "name": name,
                "variant": variant,
                "exit_code": code,
                "error": logpath.read_text()[-4000:],
            }
            (output / f"{name}.failure.json").write_text(
                json.dumps(data, indent=2) + "\n"
            )
            print("FAILED", name, data["error"][-1800:], flush=True)
            # Fix the failing path before spending GPU time on dependent probes.
            break
        data = json.loads((output / f"{name}.json").read_text())
        metrics = data["video_metrics"]
        print(
            name,
            "ms/frame",
            round(metrics["median_ms_per_frame"], 2),
            "allocated_GiB",
            round(metrics["peak_allocated_bytes"] / 2**30, 3),
            "NVML_GiB",
            round(metrics["nvml"]["sampled_device_used_peak_bytes"] / 2**30, 3),
            "counts",
            [c["count"] for c in data["checks"].values()],
            "changed_pixels",
            sum(c["changed_pixels"] for c in data["checks"].values()),
            flush=True,
        )


if __name__ == "__main__":
    main()
