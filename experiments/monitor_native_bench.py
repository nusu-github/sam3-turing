"""Run one benchmark process and sample whole-device NVML memory/clock/power.

Usage: python experiments/monitor_native_bench.py OUTPUT COMMAND [ARG ...]
The native inference process receives a Windows-only PATH, without Python.
Sampling includes setup and is a lower bound on the instantaneous device peak.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pynvml as nv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    args.output.mkdir(parents=True, exist_ok=True)
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    samples = []
    stop = threading.Event()

    def collect():
        while not stop.is_set():
            row = {"time": time.time(), "used_bytes": nv.nvmlDeviceGetMemoryInfo(handle).used}
            for key, read in (
                ("temperature_c", lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)),
                ("sm_mhz", lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)),
                ("power_mw", lambda: nv.nvmlDeviceGetPowerUsage(handle)),
            ):
                try:
                    row[key] = read()
                except nv.NVMLError:
                    pass
            samples.append(row)
            stop.wait(0.02)

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    executable = Path(args.command[0]).name.lower()
    if os.name == "nt" and executable.endswith(".exe") and "python" not in executable:
        env["PATH"] = r"C:\Windows\System32;C:\Windows"
    thread = threading.Thread(target=collect)
    thread.start()
    start = time.time()
    try:
        with (args.output / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                code = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
    finally:
        stop.set()
        thread.join()
        nv.nvmlShutdown()
    report = {
        "command": args.command, "exit_code": code, "wall_seconds": time.time() - start,
        "benchmark_environment": {key: env.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
            "TORCH_BLAS_PREFER_CUBLASLT", "SAM3_BENCH_CUDNN",
            "SAM3_PROFILE_NVTX", "SAM3_PROFILE_CAPTURE",
            "SAM3_FUSED_NORM_CAST",
            "SAM3_EXPERIMENT_MLP", "SAM3_EXPERIMENT_PROJECTION",
            "SAM3_EXPERIMENT_RESTORE", "SAM3_EXPERIMENT_ATTENTION",
            "SAM3_EXPERIMENT_QKV_ROPE",
            "SAM3_EXPERIMENT_ATTENTION_SCOPE", "SAM3_EXPERIMENT_PROJECTION_SCOPE",
            "SAM3_EXPERIMENT_KITCHEN_CENTER",
            "SAM3_EXPERIMENT_KITCHEN_CENTER_SCOPE",
            "SAM3_EXPERIMENT_MLP_PART", "SAM3_EXPERIMENT_FC2_CALIBRATION", "SAM3_EXPERIMENT_FC2_MEAN_BIAS",
            "SAM3_EXPERIMENT_OBSERVE_QKV", "SAM3_EXPERIMENT_QKV_CALIBRATION", "SAM3_EXPERIMENT_QKV_MEAN_BIAS",
            "SAM3_EXPERIMENT_OBSERVE_QKV_ERROR", "SAM3_EXPERIMENT_QKV_SHADOW_CALIBRATION", "SAM3_EXPERIMENT_QKV_OUTPUT_BIAS",
            "SAM3_EXPERIMENT_BOUNDARY", "SAM3_EXPERIMENT_FC2_NORM",
            "SAM3_EXPERIMENT_INT8_LT", "SAM3_EXPERIMENT_INT4_FC2", "SAM3_EXPERIMENT_KITCHEN_LAYOUT", "SAM3_EXPERIMENT_PIXEL", "SAM3_EXPERIMENT_MLP_SCOPE",
        )},
        "samples": samples, "sampled_peak_bytes": max((s["used_bytes"] for s in samples), default=None),
        "scope": "Whole-device NVML at approximately 20ms, including setup; not process-only or exact peak",
    }
    (args.output / "telemetry.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    print((args.output / "process.log").read_text(encoding="utf-8")[-3500:])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
