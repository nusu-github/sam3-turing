"""Windows/CUDA 13 NPP EDT feasibility probe; does not change model dispatch.

Compare preallocated NPP work against the current allocating native operator.
This is an operator feasibility test, not an end-to-end speed comparison.
"""
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys

import torch


class Size(C.Structure):
    _fields_ = [("width", C.c_int), ("height", C.c_int)]


class Context(C.Structure):
    # CUDA 13.0 nppdefs.h: NppStreamContext, including its reserved field.
    _fields_ = [("stream", C.c_void_p), ("device", C.c_int),
                ("multiprocessors", C.c_int), ("threads_per_sm", C.c_int),
                ("threads_per_block", C.c_int), ("shared_bytes", C.c_size_t),
                ("major", C.c_int), ("minor", C.c_int),
                ("flags", C.c_uint), ("reserved", C.c_int)]


def check(code):
    if code != 0:
        raise RuntimeError(f"NPP/CUDA status {code}")


def measure(fn):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def main():
    cuda = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"))
    with os.add_dll_directory(str(cuda / "bin/x64")):
        npp = C.CDLL(str(cuda / "bin/x64/nppif64_13.dll"))
        runtime = C.CDLL(str(cuda / "bin/x64/cudart64_13.dll"))
    torch.ops.load_library(str(Path("build/native-windows-cu130/sam3_native.dll").resolve()))
    npp.nppiDistanceTransformPBAGetBufferSize.argtypes = [Size, C.POINTER(C.c_size_t)]
    edt = npp.nppiDistanceTransformPBA_8u32f_C1R_Ctx
    edt.argtypes = [C.c_void_p, C.c_int, C.c_ubyte, C.c_ubyte,
                   C.c_void_p, C.c_int, C.c_void_p, C.c_int,
                   C.c_void_p, C.c_int, C.c_void_p, C.c_int,
                   Size, C.c_void_p, Context]
    runtime.cudaStreamGetFlags.argtypes = [C.c_void_p, C.POINTER(C.c_uint)]
    prop = torch.cuda.get_device_properties(0)
    stream = torch.cuda.Stream()
    flags = C.c_uint()
    check(runtime.cudaStreamGetFlags(stream.cuda_stream, C.byref(flags)))
    ctx = Context(stream.cuda_stream, 0, prop.multi_processor_count,
                  prop.max_threads_per_multi_processor, prop.max_threads_per_block,
                  prop.shared_memory_per_block, prop.major, prop.minor, flags.value, 0)
    results = []
    torch.manual_seed(9524)
    with torch.inference_mode(), torch.cuda.stream(stream):
        for h, w in [(64, 64), (65, 97), (288, 288), (1008, 1008)]:
            size = Size(w, h)
            count = C.c_size_t()
            check(npp.nppiDistanceTransformPBAGetBufferSize(size, C.byref(count)))
            scratch = torch.empty(count.value, device="cuda", dtype=torch.uint8)
            out = torch.empty((1, h, w), device="cuda", dtype=torch.float32)
            for kind in ["random", "single_site", "all_sites"]:
                x = torch.randint(0, 2, (1, h, w), device="cuda", dtype=torch.uint8)
                if kind == "single_site":
                    x.fill_(1)
                elif kind == "all_sites":
                    x.zero_()
                x[0, h // 2, w // 2] = 0  # Required: at least one site.

                def npp_call():
                    check(edt(x.data_ptr(), w, 0, 0, None, 0, None, 0, None, 0,
                              out.data_ptr(), w * 4, size, scratch.data_ptr(), ctx))

                def native_call():
                    return torch.ops.sam3_native.euclidean_distance_transform(x)

                npp_call()
                ref = native_call()
                error = (out - ref).abs().max().item()
                assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6), (h, w, kind, error)
                row = dict(height=h, width=w, kind=kind, max_abs_error=error,
                           bit_equal=torch.equal(out, ref), scratch_bytes=count.value,
                           npp_preallocated_ms=measure(npp_call), native_ms=measure(native_call))
                results.append(row)
                print(row, flush=True)
    Path(sys.argv[1]).write_text(json.dumps({"cases": results,
        "limitations": "NPP requires dimensions >=64 and at least one site; B=1 only tested. NPP buffers preallocated; native includes allocation."}, indent=2))


if __name__ == "__main__":
    main()
