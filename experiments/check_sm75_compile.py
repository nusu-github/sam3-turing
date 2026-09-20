"""Offline SM75 compilation only: this does not measure a Turing GPU."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from fused_int8_gemm import _int8_epilogue
from sam3.turing_int8 import _gelu_asymmetric, _gelu_quantize, _quantize
from sam3.turing_masks import _pack, _resize_pack, _unpack

results = []
jobs = [
    (
        "public_quantize",
        _quantize,
        {"src": "*fp32", "dst": "*i8", "scales": "*fp32"},
        {"K": 1024, "BLOCK": 1024},
        4,
    )
]
gelu_signature = {
    "mm": "*i32",
    "row_scales": "*fp32",
    "weight_scales": "*fp32",
    "bias": "*fp16",
    "q": "*i8",
    "scales": "*fp32",
}
jobs.extend(
    [
        (
            "public_gelu_quantize",
            _gelu_quantize,
            gelu_signature,
            {"K": 4736, "BLOCK": 8192},
            8,
        ),
        (
            "public_gelu_asymmetric",
            _gelu_asymmetric,
            {**gelu_signature, "zeros": "*i32"},
            {"K": 4736, "BLOCK": 8192},
            4,
        ),
    ]
)
for half in (True, False):
    jobs.append(
        (
            f"public_resize_pack_{'fp16' if half else 'fp32'}",
            _resize_pack,
            {"src": "*fp16" if half else "*fp32", "dst": "*u8"},
            {
                "IH": 288,
                "IW": 288,
                "OH": 2160,
                "OW": 3840,
                "S0": 288 * 288,
                "S1": 288,
                "S2": 1,
                "HALF": half,
                "BLOCK": 256,
            },
            4,
        )
    )
jobs.extend(
    [
        (
            "public_pack",
            _pack,
            {"src": "*i1", "dst": "*u8"},
            {"PIXELS": 2160 * 3840, "BYTES": 2160 * 3840 // 8, "BLOCK": 128},
            4,
        ),
        (
            "public_unpack",
            _unpack,
            {"src": "*u8", "dst": "*i1"},
            {"PIXELS": 2160 * 3840, "BYTES": 2160 * 3840 // 8, "BLOCK": 1024},
            4,
        ),
    ]
)
for bm, bn, warps in [(64, 128, 4), (128, 128, 8)]:
    jobs.append(
        (
            f"int8_fc1_{bm}x{bn}",
            _int8_epilogue,
            {
                "x": "*i8",
                "w": "*i8",
                "row_scales": "*fp32",
                "weight_scales": "*fp32",
                "bias": "*fp16",
                "y": "*fp16",
            },
            {
                "M": 5184,
                "N": 4736,
                "K": 1024,
                "BM": bm,
                "BN": bn,
                "BK": 64,
                "GROUP": 8,
                "GELU": True,
            },
            warps,
        )
    )
for name, kernel, signature, constants, warps in jobs:
    start = time.perf_counter()
    item = {"name": name}
    try:
        compiled = triton.compile(
            ASTSource(kernel, signature, constexprs=constants),
            target=GPUTarget("cuda", 75, 32),
            options={
                "num_warps": warps,
                "num_stages": 3,
                "enable_fp_fusion": not name.startswith("public_resize_pack"),
            },
        )
        ptx = compiled.asm["ptx"]
        item.update(
            compiled=True,
            cubin_bytes=len(compiled.asm["cubin"]),
            shared_bytes=compiled.metadata.shared,
            contains_mma_sync="mma.sync" in ptx,
            contains_dp4a="dp4a" in ptx,
        )
    except Exception as exc:
        item.update(compiled=False, error_type=type(exc).__name__, error=str(exc))
    item["seconds"] = time.perf_counter() - start
    results.append(item)
    print(json.dumps(item), flush=True)
assert not torch.cuda.is_initialized(), "Offline check unexpectedly initialized CUDA"
output = {
    "target": "sm75",
    "triton": triton.__version__,
    "gpu_execution": False,
    "note": "Offline compilation only, not runtime compatibility or performance on RTX20/GTX16.",
    "checks": results,
}
(ROOT / "experiments/results/sm75_compile_check.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
