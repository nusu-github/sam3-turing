"""Build a compact comparison table from saved image sweep results."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for path in sorted((ROOT / "results").glob("*.json")):
    data = json.loads(path.read_text())
    if "metrics" not in data or not data.get("measurement_valid", True):
        continue
    metrics = data["metrics"]
    checks = [c["vs_stock"] for c in data["checks"].values() if "vs_stock" in c]
    matched = sum(c["matched"] for c in checks)
    pixels = sum(c.get("compared_pixels", 0) for c in checks)
    changed = sum(c.get("changed_pixels", 0) for c in checks)
    rows.append(
        {
            "candidate": path.stem,
            "median_ms": round(metrics["median_wall_seconds"] * 1000, 3),
            "cold_ms": round(metrics["cold_seconds"] * 1000, 3),
            "allocated_GiB": round(metrics["peak_allocated_bytes"] / 2**30, 4),
            "reserved_GiB": round(metrics["peak_reserved_bytes"] / 2**30, 4),
            "nvml_GiB": round(
                metrics["nvml"]["sampled_device_used_peak_bytes"] / 2**30, 4
            ),
            "detections": sum(c["count"] for c in data["checks"].values()),
            "cases": len(data["checks"]),
            "matched_detections": matched,
            "mask_iou_mean": (
                round(
                    sum(c.get("mean_mask_iou", 0) * c["matched"] for c in checks)
                    / matched,
                    8,
                )
                if matched
                else None
            ),
            "mask_iou_min": min(
                (c["min_mask_iou"] for c in checks if "min_mask_iou" in c), default=None
            ),
            "changed_pixels": changed if checks else None,
            "compared_pixels": pixels if checks else None,
            "changed_percent": 100 * changed / pixels if pixels else None,
            "score_max_abs": max(
                (c.get("score_max_abs", 0) for c in checks), default=None
            ),
            "box_max_abs_px": max(
                (c.get("box_max_abs_px", 0) for c in checks), default=None
            ),
        }
    )
rows.sort(key=lambda r: r["median_ms"])
with (ROOT / "results/summary.csv").open("w") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# All image candidates",
    "",
    "Latency is truck.jpg + truck; output differences cover the listed number of cases.",
    "",
    "| Candidate | ms | CUDA allocated GiB | NVML GiB | Cases | Masks | Mean mask IoU vs stock | Changed pixels |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    iou = f'{r["mask_iou_mean"]:.6f}' if r["mask_iou_mean"] is not None else "—"
    lines.append(
        f'| [{r["candidate"]}]({r["candidate"]}.json) | {r["median_ms"]:.2f} | {r["allocated_GiB"]:.3f} | {r["nvml_GiB"]:.3f} | {r["cases"]} | {r["detections"]} | {iou} | {r["changed_pixels"] if r["changed_pixels"] is not None else "—"} |'
    )
(ROOT / "results/README.md").write_text("\n".join(lines) + "\n")
print(f"Summarized {len(rows)} saved measurements")
