"""Plot accepted configurations from their saved measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ("Stock BF16", "r14_stock_control"),
    ("Patched FP16", "r14_fp16_control"),
    ("INT8 weights only, FP16 linear", "accepted_weight_only_attention"),
    ("INT8 MLP", "r14_int8_control"),
    ("Asymmetric INT8 MLP + fused GELU", "accepted_asymmetric_mlp"),
    ("INT8 + projections + fused GELU", "accepted_fused_attention"),
    ("INT8 + CPU text encoder/cache", "accepted_cpu_text_int8"),
    ("Asym INT8 + tuned weights + CPU text", "accepted_optimized_asymmetric_cpu"),
    ("INT8 + refinements + trimmed CPU text", "accepted_refined_trimmed_fp32"),
    ("INT8 + fixed text vocabulary", "compact_fixed_all_int8"),
]


def main():
    rows = []
    for label, name in CONFIGS:
        data = json.loads((ROOT / "experiments/results" / (name + ".json")).read_text())
        checks = [c["vs_stock"] for c in data["checks"].values() if "vs_stock" in c]
        count = sum(c["matched"] for c in checks)
        iou = (
            sum(c.get("mean_mask_iou", 0) * c["matched"] for c in checks) / count
            if count
            else 1
        )
        m = data["metrics"]
        rows.append(
            (
                label,
                m["median_wall_seconds"] * 1000,
                m["nvml"]["sampled_device_used_peak_bytes"] / 2**30,
                iou,
            )
        )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 6.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.3, 1.1, 1]},
    )
    colors = [
        "#718096",
        "#2b6cb0",
        "#4c86b8",
        "#38a169",
        "#319795",
        "#16806a",
        "#b7791f",
        "#b45309",
        "#c05621",
        "#805ad5",
    ]
    for ax, column, title, maximum in zip(
        axes,
        [1, 2, 3],
        ["Image latency (ms)", "Whole GPU NVML peak (GiB)", "Mask IoU vs stock"],
        [250, 7.2, 1.004],
    ):
        values = [r[column] for r in rows]
        ax.barh(range(len(rows)), values, color=colors, height=0.62)
        ax.set_title(title)
        ax.set_xlim(0.99 if column == 3 else 0, maximum)
        ax.grid(axis="x", alpha=0.16)
        ax.set_axisbelow(True)
        for y, value in enumerate(values):
            text = f"{value:.5f}" if column == 3 else f"{value:.2f}"
            ax.text(
                value + maximum * 0.013 if column != 3 else value + 0.00015,
                y,
                text,
                va="center",
                fontsize=9,
            )
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    axes[2].set_xticks([0.99, 0.995, 1.0], ["0.990", "0.995", "1.000"])
    fig.suptitle(
        "SAM 3 image patch — RTX 3090, existing RunPod environment", x=0.54, fontsize=13
    )
    fig.text(
        0.02,
        0.025,
        "Latency: truck image + prompt, 9-run median; trimmed CPU text uses no cache. IoU: 3 images / 5 prompts, agreement with stock.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    path = ROOT / "docs/images/accepted_image_benchmarks.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
