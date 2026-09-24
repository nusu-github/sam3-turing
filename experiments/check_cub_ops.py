"""Exact CPU parity, stream/graph checks and CUDA-event timings for CUB ops."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch

torch.set_num_threads(4)
torch.manual_seed(32024)
torch.ops.load_library(str(Path(sys.argv[1]).resolve()))
ops = torch.ops.sam3_native
digest = hashlib.sha256()
rows = []

def record(t):
    digest.update(t.cpu().contiguous().numpy().tobytes())

def timing(fn):
    if "--check-only" in sys.argv:
        return None
    for _ in range(10): fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        result = fn()
    for _ in range(10): graph.replay()
    times = []
    for _ in range(30):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20): graph.replay()
        b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / 20)
    return statistics.median(times)

stream = torch.cuda.Stream()
with torch.inference_mode(), torch.cuda.stream(stream):
    for shape in [(0, 3, 7), (1, 1, 1), (1, 1, 7), (1, 1, 8), (1, 1, 9),
                  (1, 1, 33), (3, 7, 13), (2, 72, 72), (1, 1008, 1008)]:
        cpu = torch.randint(0, 2, shape, dtype=torch.bool)
        x = cpu.cuda()
        packed = ops.pack_masks(x)
        assert torch.equal(packed.cpu(), ops.pack_masks(cpu))
        unpacked = ops.unpack_masks(packed, shape[1], shape[2])
        assert torch.equal(unpacked[:, 0], x)
        record(packed); record(unpacked)
        raw = torch.randint(0, 256, packed.shape, dtype=torch.uint8)
        assert torch.equal(ops.unpack_masks(raw.cuda(), shape[1], shape[2]).cpu(),
                           ops.unpack_masks(raw, shape[1], shape[2]))
        if shape[0]:
            rows.append(dict(op="pack", shape=shape, ms=timing(lambda: ops.pack_masks(x))))
            rows.append(dict(op="unpack", shape=shape, ms=timing(lambda: ops.unpack_masks(packed, shape[1], shape[2]))))
    for shape in [(0, 3, 7), (2, 1, 1), (2, 31, 37), (1, 256, 256)]:
        for kind in ["random", "zero", "one"]:
            cpu = torch.randint(-2, 3, shape, dtype=torch.int64)
            if kind == "zero": cpu.zero_()
            if kind == "one": cpu.fill_(1)
            cpu = cpu.transpose(1, 2)  # Also exercise normalization of strided inputs.
            x = cpu.cuda()
            actual = ops.connected_components(x)
            expected = ops.connected_components(cpu)
            for a, b in zip(actual, expected):
                assert torch.equal(a.cpu(), b), (shape, kind)
                record(a)
            if shape[0]:
                rows.append(dict(op="components", shape=shape, kind=kind,
                                 ms=timing(lambda: ops.connected_components(x))))
    # Warm every operation before capturing; verify reuse on changed input.
    x = torch.ones((2, 31, 37), device="cuda", dtype=torch.bool)
    for _ in range(3):
        p = ops.pack_masks(x); u = ops.unpack_masks(p, 31, 37)
        labels, sizes = ops.connected_components(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        p = ops.pack_masks(x); u = ops.unpack_masks(p, 31, 37)
        labels, sizes = ops.connected_components(x)
    for value in [False, True]:
        x.fill_(value); graph.replay()
        assert torch.equal(u[:, 0], x)
        reference = ops.connected_components(x.cpu())
        assert all(torch.equal(a.cpu(), b) for a, b in zip((labels, sizes), reference))
Path(sys.argv[2]).write_text(json.dumps(dict(cases=rows, timing="CUDA graph replay, median of 30 batches of 20 replays; allocations excluded", sha256=digest.hexdigest(),
                                          cpu_parity=True, graph_replay=True), indent=2))
print("CPU parity, nondefault stream, graph replay passed:", digest.hexdigest())
