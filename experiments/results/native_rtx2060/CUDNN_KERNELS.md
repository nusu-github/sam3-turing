# cuDNN and vision kernel investigation

Measured 2026-09-23 on the same RTX 2060 Max-Q and native FP16 truck workload.
Base commit 19f8bf9 plus local Windows fixes and opt-in diagnostics.

## cuDNN is active

Standalone LibTorch reports cuDNN available/enabled, runtime version 91200
(9.12.0). Nsight Systems captures actual `cudnn_turing` and `__5x_cudnn`
convolution kernels. The earlier CMake `USE_CUDNN=0` message does not mean that
the prebuilt LibTorch convolution implementation runs without cuDNN.

Four independent processes ran default/tuned/tuned/default, each with two
warmups and ten timed inferences. Only `setBenchmarkCuDNN` was changed.

| Setting | Pooled wall median | Process medians | Cold inference |
|---|---:|---|---|
| Default, benchmark off | 561.68 ms | 559.70 / 565.63 ms | 1.053 / 1.057 s |
| cuDNN benchmark on | 560.98 ms | 557.82 / 562.96 ms | 1.683 / 1.702 s |

The 0.12% median difference does not establish a useful speedup. Autotuning
adds about 0.63 s to this cold inference. Both tuned processes retain the one
detection, score and query index but differ from the default result by three
mask pixels and at most 0.011536 pixels in box coordinates. Default runs remain
byte-identical to the previous native result. Benchmark remains off by default.

## Vision breakdown and actual kernels

Nsight Systems 2025.3.2 captured three warm inferences using CUDA-profiler
start/stop, CUDA/cuBLAS traces and nested NVTX labels. No concurrent GPU jobs.
The following values are NVTX GPU-projected ranges per inference, not CPU call
durations. Nested rows must not be added to their parents.

| Range | GPU-projected time | Share of vision |
|---|---:|---:|
| Entire vision | 397.06 ms | 100% |
| MLP, including GELU and final residual | 191.11 ms | 48.1% |
| Attention including projections and RoPE | 157.39 ms | 39.6% |
| All pyramid necks | 15.86 ms | 4.0% |
| Other/gaps (embedding, norms, layouts, etc.) | about 32.7 ms | about 8.2% |

Inside attention: QKV 58.43 ms, SDPA 69.16 ms, RoPE 10.42 ms and output
projection 19.01 ms. These are subsets of the attention row. The four global
attention blocks each take about 19 ms for the complete block; local-window
blocks are mostly about 11 ms.

Across the full inference trace, the leading FP16 Tensor Core GEMM kernel
`turing_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn` accounts for 50.6% of
summed kernel duration. The CUTLASS SM75 memory-efficient attention kernel
accounts for 20.1%. This is not an unfused math-attention fallback. These
percentages are kernel-duration shares, distinct from elapsed stage shares.

Trace wall median was 526.37 ms, versus roughly 560 ms in the separately
monitored benchmark runs. Do not present this as a speedup: profiler conditions,
clock behavior and monitoring differ. All trace outputs match the previous
native result byte for byte.

## Hardware counter spot check

Nsight Compute 2025.3.1 profiled the first matching MLP GEMM inside
`vision.block.0/vision.mlp` (grid 37 x 41, block 128). Kernel replay, 12 passes,
without locking GPU clocks or flushing caches. The report confirms the NVTX
stack and records approximately:

- 1.95 ms duration at 1.79 GHz SM clock.
- DRAM throughput 40.41%, compute throughput 47.54% of reported peak.
- Achieved occupancy 24.74%, theoretical 25%, limited by registers/shared memory.
- Execution-pipe wait accounts for about 62% of cycles between issued instructions
  according to the profiler's diagnostic rule.

This spot check does not support calling this GEMM simply DRAM-bandwidth
saturated. It suggests investigating GEMM algorithm/tiling and execution-pipeline
utilization. It does not establish the bottleneck of all kernels or a guaranteed
speedup. The profiler's suggested speedup is not a measured improvement.
A separate first-GEMM capture ran at only 812 MHz, illustrating why uncontrolled
replay durations cannot serve as production latency measurements.

## Reproduce and evidence

`sam3_image_latency` keeps its existing arguments. Environment switches:

- `SAM3_PROFILE_NVTX=1`: enable nested vision labels in the runtime. On builds
  without CUDA/NVTX3 headers, these diagnostic labels are no-ops.
- `SAM3_PROFILE_CAPTURE=1`: CUDA profiler start/stop around measured repetitions,
  excluding setup, cold inference and warmup.
- `SAM3_BENCH_CUDNN=1`: enable cuDNN algorithm benchmarking in this diagnostic
  executable only. Leave unset for the existing default.

Capture with `nsys profile --trace=cuda,nvtx,cublas --sample=none
--capture-range=cudaProfilerApi --capture-range-end=stop --output=TRACE EXE ...`.
Use three repetitions and two warmups. Export summaries with
`nsys stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum --format csv TRACE.nsys-rep`.
The CPU context-switch trace required administrator privileges and was disabled;
the CUDA/NVTX capture succeeded without changing privileges.

For the MLP counter check, set both profiling environment variables and run
`ncu --section SpeedOfLight --section Occupancy --section SchedulerStats
--section WarpStateStats --profile-from-start off --nvtx --nvtx-include vision.mlp/
--clock-control none --cache-control none
--kernel-name regex:turing_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn
--launch-count 1 --export REPORT EXE ...`, with one measured repetition.

[cudnn-and-kernels.json](cudnn-and-kernels.json) retains all timing samples,
kernel/NVTX summaries and comparisons. Counter exports:
[MLP](mlp-counters.txt), [first GEMM](qkv-counters.txt). Large profiler archives
and raw outputs remain under `.cache/native-perf`. The instrumented runtime
rebuilt successfully and passed 41/41 tests. No inference operators, shapes,
precision or default cuDNN settings were changed.
