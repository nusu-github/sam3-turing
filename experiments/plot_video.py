"""Plot the completed short-video candidates from their saved measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = [
    ("Stock BF16", "video_stock_bf16_b1"),
    ("FP16", "video_fp16_b1"),
    ("FP16 / efficient attention", "video_fp16_efficient_b1"),
    ("INT8", "video_fp16_int8_b1"),
    ("INT8 + ViT compile", "video_fp16_int8_compile_b1"),
    ("INT8 + compile + CPU text", "video_fp16_int8_cpu_compile_b1"),
    ("Same / efficient attention", "video_fp16_int8_cpu_compile_efficient_b1"),
    ("INT8 + compile + trimmed CPU text", "video_fp16_int8_cpu_trim_compile_b1"),
    ("Same / efficient attention", "video_fp16_int8_cpu_trim_compile_efficient_b1"),
    ("INT8 + CPU trim + ViT/decoder compile", "video_fp16_int8_cpu_trim_compile_decoder_b1"),
]
metrics = [
    json.loads((ROOT / f"experiments/results/{name}.json").read_text())["video_metrics"]
    for _, name in rows
]
latency = [m["median_ms_per_frame"] for m in metrics]
allocated = [m["peak_allocated_bytes"] / 2**30 for m in metrics]
nvml = [m["nvml"]["sampled_device_used_peak_bytes"] / 2**30 for m in metrics]
y = list(range(len(rows)))
fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 7.1), sharey=True)
colors = (
    ["#8b96a5"]
    + ["#447bc5"] * 3
    + ["#20875b"]
    + ["#447bc5"] * 2
    + ["#20875b", "#447bc5"]
    + ["#9b4f96"]
)
left.barh(y, latency, color=colors, height=0.6)
left.set_yticks(y, [label for label, _ in rows])
left.invert_yaxis()
left.set_xlim(0, max(latency) * 1.2)
left.set_xlabel("Milliseconds per frame (lower is better)")
for i, value in enumerate(latency):
    left.text(value + 4, i, f"{value:.1f}", va="center", fontsize=9)
right.barh(y, nvml, color="#b7cdec", height=0.6, label="Whole GPU / NVML")
right.barh(y, allocated, color="#447bc5", height=0.32, label="PyTorch allocated")
right.axvline(6, color="#b84a4a", linestyle="--", linewidth=1)
right.set_xlim(0, max(nvml) * 1.15)
right.set_xlabel("Peak GPU memory / GiB")
right.legend(loc="lower right", fontsize=8)
for i, value in enumerate(nvml):
    right.text(value + 0.08, i, f"{value:.2f}", va="center", fontsize=9)
for ax in (left, right):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
fig.suptitle("SAM 3.1: short-video patch experiments on RTX 3090", fontsize=14)
fig.text(
    0.03,
    0.02,
    "24 frames, person prompt, grounding batch 1, TF32 off; median of 3 runs after a cold run.\nAll candidates retain 4 people and matching IDs on every frame. This is not a Turing GPU measurement.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.10, 1, 0.95))
output = ROOT / "docs/images/video_benchmarks.png"
fig.savefig(output, dpi=160)
print(output)
