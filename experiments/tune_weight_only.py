"""Short kernel-only tile search at SAM 3's MLP shapes; no model quality claim."""

import json
import sys
from pathlib import Path

import torch
from torch import nn
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weight_only_int8 import WeightOnlyInt8Linear


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    tiles = [
        (64, 128, 32, 4),
        (64, 128, 64, 4),
        (64, 128, 128, 4),
        (128, 128, 64, 4),
        (128, 128, 64, 8),
        (64, 256, 64, 8),
        (128, 256, 64, 8),
        (128, 64, 64, 4),
        (64, 64, 64, 4),
        (32, 128, 64, 4),
    ]
    result = {
        "scope": "Kernel-only on RTX 3090; random inputs, SAM 3 MLP shapes. End-to-end validation is separate.",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shapes": [],
    }
    path = ROOT / "experiments/results/weight_only_tiles.json"
    with torch.inference_mode():
        for k, n in [(1024, 4736), (4736, 1024)]:
            x = torch.randn(5184, k, device="cuda", dtype=torch.float16)
            linear = nn.Linear(k, n, device="cuda", dtype=torch.float16).eval()
            quantized = WeightOnlyInt8Linear(linear, tiles[0], group=8)
            dequantized = (
                quantized.weight_int8.float() * quantized.weight_scale[:, None]
            ).half()
            reference = torch.nn.functional.linear(x, dequantized, quantized.bias)
            item = {"m": 5184, "k": k, "n": n, "candidates": []}
            item["fp16_ms"] = triton.testing.do_bench(
                lambda: linear(x), warmup=25, rep=75
            )
            result["shapes"].append(item)
            for tile in tiles:
                for group in (4, 8, 16):
                    quantized.tile, quantized.group = tile, group
                    record = {"tile": tile, "group": group}
                    try:
                        actual = quantized(x)
                        record["ms"] = triton.testing.do_bench(
                            lambda: quantized(x), warmup=25, rep=75
                        )
                        record["max_abs_vs_dequantized_fp16"] = (
                            (actual - reference).abs().max().item()
                        )
                        record["finite"] = bool(actual.isfinite().all())
                    except Exception as exc:
                        record["error"] = str(exc)[-1200:]
                    item["candidates"].append(record)
                    path.write_text(json.dumps(result, indent=2) + "\n")
            best = sorted(
                (c for c in item["candidates"] if "ms" in c), key=lambda c: c["ms"]
            )[:5]
            print(
                json.dumps(
                    {"shape": [5184, k, n], "fp16_ms": item["fp16_ms"], "best": best}
                ),
                flush=True,
            )
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
