# Turing FP16 Attention local experiment — 2026-09-24

## Decision

The external d=64 kernel improves long-sequence global Attention on this RTX 2060 Max-Q. Keep it **opt-in**: the five-case model smoke check misses the preset box-coordinate gate on `child`. Do not enable it globally or claim validated production parity. Short local Attention remains on existing SDPA.

This follows [the source reassessment](PORTABILITY_REASSESSMENT.md). It does not repeat hardware INT8 support, cuDNN, or previously unsuccessful MLP GEMM heuristic searches.

## Implementation and provenance

- Base repository commit: `ba018a0`, plus existing local Windows, profiling and INT8 experiments.
- External donor: [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing), commit `9ef98fcb506bb1e2fe3cece50935e2935bf6b124`.
- CUTLASS submodule: `df18f5e4f5de76bed8be1de8e4c245f2f5ec3020` (BSD-3-Clause). No explicit parent donor LICENSE/NOTICE was found in the inspected tree. Parent redistribution terms remain unresolved; donor code is not vendored into tracked runtime sources.
- Optional CMake object target reads an external checkout. Generated header copies omit the unused `torch/extension.h` include. CUDA 13 requires changing CUTLASS's unused `memsetDevice` wrapper annotation from host/device to host, since it invokes a host-only virtual method. Compiler cross-execution-space checking remains enabled.
- Native ATen/CUDA wrapper, no Python extension at inference; SM75, FP16, dense equal-length self-attention, head dimension 64, scale 1/8. Current CUDA stream and device guard, checked CUDA calls and launch errors.
- Existing Q/K token-major views require no copy. Packed V is made contiguous. Output/LSE allocation and packing are included in operator timings.
- Kernel uses fixed 128×128 tiles, 8 warps. Actual shared-memory requirement is 32 KiB; 64 KiB reservation is retained as a control. Tail kernel uses 228 registers/thread, even kernel 234, with zero stack/spills in compiler output. Both reservations allow only **one block per SM**: reducing shared memory does not double occupancy.

## Timing

Windows, CUDA 13.0, MSVC 14.44, LibTorch 2.10 cu130, RTX 2060 Max-Q 6 GB. Existing INT8 MLP boundary fusion and QKV INT8 are fixed in every model comparison; restore uses the original flat kernel. No new attention quantization is introduced.

| Path | Global operator, representative median | Model median, 30 samples | Model p95 | Vision stage mean |
|---|---:|---:|---:|---:|
| Existing SDPA | 9.45 ms | 412.73 ms | 428.04 ms | 290.01 ms |
| Global donor, 32 KiB | 7.56 ms | 402.74 ms | 417.00 ms | 283.60 ms |
| Global donor, 64 KiB | 7.59 ms | 400.56 ms | 413.01 ms | 284.07 ms |

Operator values summarize four alternating timing passes, including packing. Global shape is B=1,H=16,N=5184,D=64. At local B=9,N=576 the donor lost in every paired pass; do not use `all32` as a recommended setting.

Model timing uses two separate processes per mode, five warmups and fifteen samples each, in order exact/32/64/64/32/exact. Text features are cached; image features are recomputed. Includes RGB transfer, preprocessing, Vision, Detector and postprocessing; excludes weight load and file I/O. Stage timings use CUDA events, with no Nsight trace during timing.

Global operator time falls about 20%; whole-model median falls 2.4% for 32 KiB and 2.9% for 64 KiB. The unchanged Detector also fluctuates by roughly 4–5 ms, so the entire wall-time gain cannot be attributed to Attention. Vision savings are about 6 ms. There is no established whole-model winner between 32/64 KiB. Tail SM clock medians varied 1245–1305 MHz and maximum temperature 61–69 °C; clocks were not locked. Peak allocated memory is effectively unchanged (2,788,970,496 vs 2,788,249,600 bytes).

## Numerical validation

Unit shapes (B,N): (1,1), (1,64), (1,128), (2,129), (9,576), (1,5184), all H=16,D=64. Inputs and kernels run on a nondefault CUDA stream. Tests include noncontiguous V and tail masking. Small shapes use explicit FP32 attention as reference; large shapes use existing FP16 SDPA.

Preset unit gates: finite output, max absolute error ≤0.02, RMSE ≤0.002. All pass; observed worst absolute error 0.000488281, global RMSE 0.000008702. 32/64 KiB outputs are bit-identical in every unit case and all five model cases.

Model gates were set before examining results: same detection count, matched mask IoU ≥0.98, score difference ≤0.02, box-coordinate difference ≤1 pixel, all against the previous QKV INT8 path. Matching uses maximum-total-IoU Hungarian assignment.

| Case | Count | Minimum IoU vs previous | Max score difference | Max box difference, px |
|---|---:|---:|---:|---:|
| truck | 1 | 0.999796 | 0.000488 | 0.0662 |
| paper bag | 4 | 0.998849 | 0.000977 | 0.1135 |
| child | 6 | 0.997387 | 0.005859 | **1.0154 — FAIL** |
| wheel | 4 | 0.994160 | 0.001465 | 0.1380 |
| empty | 0 | n/a | n/a | n/a |

Detection counts all match. Relative to the original FP16 path, minimum mask IoU is 0.992350, maximum score difference 0.012695 and maximum box difference 1.044128 px. These are consistency checks on five cases, not ground-truth accuracy measurements. The 1-pixel gate remains unchanged; the summarizer writes all results then exits nonzero to preserve the failed outcome.

Compute Sanitizer memcheck: all six unit cases complete, zero errors. Racecheck also completes all six cases with zero hazards, errors or warnings (83 seconds). Both runs filter the donor `forward64` kernels; this is not a sanitizer audit of the entire model. Existing CTest suite: **51/51 pass**, 139.44 seconds, with the newly built DLL and experimental Attention unset (existing SDPA). Log: `build/native-windows-cu130/windows-validation-attention.log`. These regression passes do not override the failed experimental model quality gate. Python drivers compile and `git diff --check` passes.

## Reproduce

With the existing Windows CUDA/LibTorch toolchain configured:

```powershell
cmake -S native -B build/native-windows-cu130 -DSAM3_EXPERIMENT_TURING_ATTENTION=ON -DSAM3_TURING_DONOR=D:/PycharmProjects/sam3-turing/.cache/portability-review/flash-attention-turing
cmake --build build/native-windows-cu130 --target sam3_turing_attention_bench sam3_image_latency
.venv/Scripts/python.exe experiments/run_attention_native.py timing
.venv/Scripts/python.exe experiments/run_attention_native.py quality
.venv/Scripts/python.exe experiments/summarize_attention_native.py
```

Drivers refuse to overwrite existing run directories. The summarizer intentionally exits 1 for the recorded box gate failure. Raw runs: `.cache/native-perf/attention-ba018a0`. Durable results: [model JSON](attention-ba018a0.json), [operator JSON](attention-micro.json), [memcheck unit JSON](attention-memcheck.json).

CMake option defaults OFF. When built ON, unset `SAM3_EXPERIMENT_ATTENTION` or `exact` still selects existing SDPA. `global32`/`global64` replace only the four 5184-token Vision global layers in the tested image path. `all32` is diagnostic only. Other paths, including Detector and video memory Attention, are outside this experiment's validation scope.

The next independent candidate remains QKV INT32 restoration plus RoPE fusion: earlier profiling measured restoration at ~17.7 ms and RoPE at ~6 ms. Avoid assuming this Attention substitution proves an INT8 Attention kernel or a different tile shape will be faster.
