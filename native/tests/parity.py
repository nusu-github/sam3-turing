"""Development-only parity checks. The tested shared library has no Python ABI."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def nms_reference(ious, scores, threshold):
    order = torch.argsort(scores.cpu(), descending=True, stable=True).tolist()
    matrix = (ious > threshold).cpu()
    result = []
    while order:
        current, *remaining = order
        result.append(current)
        order = [j for j in remaining if not matrix[current, j]]
    return torch.tensor(result, dtype=torch.int64)


def check(device, ops):
    cases = 0
    for n, h, w in [(0, 1, 1), (1, 1, 1), (3, 3, 3), (2, 17, 31), (5, 4, 128)]:
        x = torch.rand(n, w, h, device=device).gt(0.5).transpose(1, 2)
        packed = ops.pack_masks(x)
        reference = np.packbits(x.cpu().reshape(n, h * w).numpy(), axis=1, bitorder="little")
        torch.testing.assert_close(packed.cpu(), torch.from_numpy(reference), rtol=0, atol=0)
        torch.testing.assert_close(ops.unpack_masks(packed, h, w).squeeze(1), x, rtol=0, atol=0)
        # A padded byte need not have zero high bits to decode correctly.
        padded = packed.clone()
        if h * w % 8 and n:
            padded[:, -1] |= (255 << (h * w % 8)) & 255
            torch.testing.assert_close(ops.unpack_masks(padded, h, w).squeeze(1), x)
        cases += 1
    for dtype in [torch.float16, torch.float32, torch.float64]:
        for n, shape, size in [(0, (2, 2), (3, 7)), (3, (7, 3), (11, 13)), (2, (3, 7), (1, 1))]:
            x = torch.randn(n, *shape, device=device, dtype=dtype).transpose(1, 2)
            for chunk in [1, 8]:
                actual = ops.resize_and_pack_masks(x, *size, chunk)
                if n:
                    expected = F.interpolate(x[:, None], size, mode="bilinear", align_corners=False).sigmoid() > 0.5
                    torch.testing.assert_close(ops.unpack_masks(actual, *size), expected, rtol=0, atol=0)
                else:
                    assert actual.shape == (0, (size[0] * size[1] + 7) // 8)
                cases += 1
        x = torch.tensor([-0.001, 0, 0.0001, 0.0009765625, 0.001], dtype=dtype, device=device)[None, None]
        decoded = ops.unpack_masks(ops.resize_and_pack_masks(x, 1, 5), 1, 5)
        torch.testing.assert_close(decoded, (x.sigmoid() > 0.5)[:, None], rtol=0, atol=0)
        cases += 1
    for n in [0, 1, 2, 17, 257]:
        for threshold in [0.0, 0.5, 1.0]:
            # Deliberate ties and exact threshold values. Non-symmetric matrix
            # also catches accidental transpose in the generic matrix API.
            scores = torch.randint(0, 4, (n,), device=device).float()
            ious = (torch.randint(0, 3, (n, n), device=device) / 2).T
            actual = ops.generic_nms(ious, scores, threshold)
            expected = nms_reference(ious, scores, threshold)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            cases += 1
    # Cross the largest block size in the original Triton tuning list.
    n = 8203
    ious = torch.zeros(n, n, device=device)
    scores = torch.ones(n, device=device)
    torch.testing.assert_close(ops.generic_nms(ious, scores, 0.5), torch.arange(n, device=device))
    cases += 1
    for call in [
        lambda: ops.pack_masks(torch.zeros(1, 2, 3, device=device)),
        lambda: ops.unpack_masks(torch.zeros(1, 1, dtype=torch.uint8, device=device), 3, 3),
        lambda: ops.resize_and_pack_masks(torch.zeros(1, 2, 3, device=device), 2, 3, 0),
        lambda: ops.generic_nms(torch.zeros(2, 3, device=device), torch.zeros(2, device=device), 0.5),
    ]:
        try:
            call()
        except RuntimeError:
            cases += 1
        else:
            raise AssertionError("invalid input accepted")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.manual_seed(713)
    torch.ops.load_library(str(args.library.resolve()))
    start = time.perf_counter()
    result = {"cpu_cases": check("cpu", torch.ops.sam3_native)}
    if args.cuda:
        assert torch.cuda.is_available()
        # All CUDA checks run on a nondefault stream to verify stream handling.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result["cuda_cases"] = check("cuda", torch.ops.sam3_native)
        stream.synchronize()
        result["gpu"] = torch.cuda.get_device_name()
    result["elapsed_seconds"] = time.perf_counter() - start
    result["torch"] = torch.__version__
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
