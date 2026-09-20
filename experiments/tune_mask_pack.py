"""Tune the fused mask kernel; reuse the learned logits after the first capture."""

import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/archive_reference/sam3_fp16_lab")]
from gpu_common import environment, measure
from sam3.turing_masks import resize_and_pack_masks, unpack_masks
from fused_mask_pack import fused_resize_and_pack

path = ROOT / "experiments/results/mask_logits.pt"
if not path.exists():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/fused_masks_bench.py"),
            "--capture-only",
        ],
        check=True,
    )
low = torch.load(path, weights_only=True).cuda()
size = (2160, 3840)
ref = resize_and_pack_masks(low, size)
result = {
    "environment": environment(),
    "scope": "Kernel tuning on 200 saved learned FP16 mask logits -> 4K",
    "variants": [],
}
# Check the cutoff against every FP16 bit pattern (including infinities/NaNs).
values = (
    torch.arange(65536, device="cuda", dtype=torch.int32)
    .to(torch.int16)
    .view(torch.float16)
)
result["fp16_cutoff_mismatches"] = int(
    ((values.sigmoid() > 0.5) != (values > 2**-10)).sum()
)
del values
for direct in [False, True]:
    for block, warps in [(64, 4), (128, 4), (256, 4), (128, 8), (256, 8)]:
        out, metrics = measure(
            lambda: fused_resize_and_pack(
                low, size, block=block, warps=warps, direct=direct
            ),
            warmups=2,
            repetitions=7,
        )
        changed_bytes = int((ref != out).sum())
        result["variants"].append(
            {
                "block": block,
                "warps": warps,
                "direct_threshold": direct,
                "metrics": metrics,
                "changed_bytes_vs_accepted": changed_bytes,
            }
        )
        print(
            block,
            warps,
            direct,
            round(metrics["median_wall_seconds"] * 1000, 3),
            changed_bytes,
            flush=True,
        )
        del out
        gc.collect()
        torch.cuda.empty_cache()
(ROOT / "experiments/results/mask_pack_tuning.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("FP16 cutoff mismatches:", result["fp16_cutoff_mismatches"], flush=True)
