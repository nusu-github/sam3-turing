"""Compare native primitives against the frozen repository's Triton pathways.

Development only; requires the original Python/Triton environment and a CUDA GPU.
The native deployed library does not import or link them.
"""
import argparse
import json
from pathlib import Path
import torch
from sam3.model.edt import edt_triton
from sam3.perflib.triton.connected_components import connected_components_triton
from sam3.perflib.triton.nms import nms_triton
from sam3.turing_masks import resize_and_pack_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library.resolve()))
    ops = torch.ops.sam3_native
    torch.manual_seed(482)
    counts = {"mask": 0, "nms": 0, "components": 0, "edt": 0}
    for dtype in [torch.float16, torch.float32]:
        for n in [0, 1, 9]:
            x = torch.randn(n, 17, 13, device="cuda", dtype=dtype)
            torch.testing.assert_close(ops.resize_and_pack_masks(x, 31, 41), resize_and_pack_masks(x, (31, 41), fused=False), rtol=0, atol=0)
            counts["mask"] += 1
    for n in [0, 1, 65, 513]:
        scores = torch.randint(0, 8, (n,), device="cuda").float()
        ious = torch.randint(0, 3, (n, n), device="cuda").float() / 2
        torch.testing.assert_close(ops.generic_nms(ious, scores, 0.5), nms_triton(ious, scores, 0.5), rtol=0, atol=0)
        counts["nms"] += 1
    for shape in [(1, 17, 31), (3, 65, 67)]:
        for max_value in [2, 4]:
            x = torch.randint(0, max_value, shape, device="cuda")
            native_labels, native_sizes = ops.connected_components(x)
            original_labels, original_sizes = connected_components_triton(x)
            torch.testing.assert_close(native_labels, original_labels, rtol=0, atol=0, check_dtype=False)
            torch.testing.assert_close(native_sizes, original_sizes, rtol=0, atol=0, check_dtype=False)
            counts["components"] += 1
        x = torch.randint(0, 2, shape, device="cuda").bool()
        x[:, 0] = False
        x[:, -1] = False
        x[:, :, 0] = False
        x[:, :, -1] = False
        torch.testing.assert_close(ops.euclidean_distance_transform(x), edt_triton(x), rtol=1e-6, atol=1e-6)
        counts["edt"] += 1
    report = {"passed": counts, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
