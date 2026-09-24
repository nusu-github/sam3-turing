# INT8 MLP boundary fusion after upstream ba018a0

2026-09-24, RTX 2060 Max-Q / SM75, native C++ LibTorch 2.10 cu130, CUDA 13.0.

## Upstream integration

Fast-forwarded `codex/native-onboarding` from `908e476` to `ba018a0`, including implementation commit `1ee3440` (per-axis exact Vision positions and fused FP32 normalization) and the subsequent documentation commit. Position fusion is enabled. Preserved local Windows support, approximation experiments, and NVTX diagnostics. Resolved CMake and README overlaps; the pre-integration local changes remain backed up in the stash named `local Windows INT8 profiling before ba018a0`.

## Implementation

New opt-in mode: `SAM3_EXPERIMENT_MLP=int8_boundary`. Default FP16 remains available. The original `int8` mode produced byte-identical outputs and was later removed in the repository cleanup (it remains in commit `7d7fddb`).

The original INT8 FC1 output was restored with bias and erf GELU to global FP16 storage, then read by a second kernel for row absmax and INT8 packing. The new `approx_restore_quant` kernel holds the restored activations in registers, reduces absmax across a row, and writes INT8 values plus their scale directly for FC2. A 256-thread block handles each row; the actual 4736-wide activation uses 19 elements per thread and warp shuffle reductions.

The existing FP16 rounding before absmax is preserved. This change introduces no additional intended numerical approximation beyond the existing INT8 MLP experiment. FC1 still materializes its INT32 accumulator; its input quantization and FC2 output restoration are unchanged. This is boundary fusion, not a custom fused GEMM epilogue.

Source: `native/tools/approx_boundary.cu`; integration: `native/src/vision_encoder.cpp`. The comparison diagnostic is `sam3_boundary_bench`.

## Timing on the same upstream build

Truck image, prompt `truck`, resolution 1008, batch 1, 200 queries, threshold 0.5. Cached text, recomputed image features. Setup and disk IO excluded. No Nsight capture or NVTX during these timings; CUDA events record stage intervals.

Order: FP16 → INT8 → fused INT8 → fused INT8 → INT8 → FP16. Each fresh process has one cold inference, five warmups, and 15 timed inferences: 30 samples per mode.

| Mode | Wall median | Wall p95 | Mean Vision CUDA-event interval |
|---|---:|---:|---:|
| FP16 exact | 491.51 ms | 505.74 ms | 371.35 ms |
| Original INT8 MLP | 441.13 ms | 461.03 ms | 323.88 ms |
| Fused-boundary INT8 MLP | **419.85 ms** | **432.39 ms** | **300.25 ms** |

The boundary change saves **21.28 ms / 4.82%** of whole-inference median versus original INT8. Combined with INT8 MLP, it is **14.58% shorter than FP16** on this build and workload. Vision stage time falls by 23.63 ms relative to original INT8.

Per-process wall medians: FP16 483.95 / 496.60 ms; INT8 438.87 / 448.68 ms; fused 419.49 / 420.48 ms. Clocks were not locked: final 100 NVML samples' median SM clock was 1260–1305 MHz for FP16, 1260–1275 MHz for INT8, and 1252.5–1275 MHz for fused. Peak temperatures spanned 63–69 C. These are device-level telemetry samples, including output/setup effects, not precise inference-only clock averages. Do not infer the upstream commit's speedup from earlier sessions with different thermal/clock conditions.

The isolated 5184×4736 boundary benchmark measured separate restore+quantize at 1.457–1.469 ms and fused at 0.682–0.719 ms. This synthetic comparison supports the mechanism; full-model timings above are the application result.

Peak allocated memory: FP16 2,375,946,240 bytes, original INT8 2,687,193,088 bytes, fused INT8 2,688,241,664 bytes. Although the intermediate FP16 activation is removed, **whole-model peak allocation did not improve** (fused is 1 MiB above original INT8). Both INT8 modes still retain original FP16 weights in this experiment.

## Validation and reproduction

The upstream full build and the fused-boundary build succeeded. All **51 component tests passed** after the changes (137.37 seconds), including the newly imported position tests. Log: `build/native-windows-cu130/windows-validation-boundary.log`.

The boundary diagnostic checks widths 32, 257, 1024, 4736 and 8192, including non-aligned tails, random, zero and constant inputs. INT8 values and FP32 row scales are exactly equal to separate restore+quantize for all cases. Both timed fused Truck runs' masks, scores, boxes and query indices are byte-identical to original INT8. Upstream FP16 and original INT8 Truck outputs also match their respective pre-update outputs.

All five actual-model cases produced **byte-identical masks, scores, boxes and query indices between fused and original INT8**. Counts were Truck 1, bag 4, child 6, wheel 4 and empty 0. Versus FP16, the minimum matched mask IoU was 0.999670 / 0.998122 / 0.996927 / 0.989715 respectively; the empty case remained empty. Maximum score difference was 0.0112305 and maximum matched box-coordinate difference was 0.64856 pixels. These differences are the same for both INT8 paths. Byte parity against original INT8 does not establish ground-truth accuracy; this remains a five-case smoke test, and the original quantization error versus FP16 still applies.

- `experiments/run_boundary_native.py timing|quality` and `summarize_boundary_native.py` ran the six timing processes and the FP16 / original INT8 / fused INT8 five-case comparison (`boundary-ba018a0.json`). These drivers were removed after the experiment; they remain in commit `7d7fddb`.
- `sam3_boundary_bench OUTPUT.json` — kernel correctness and isolated timing.

Raw metrics, stage intervals, binary outputs and NVML telemetry: `.cache/native-perf/boundary-ba018a0/`. Kernel diagnostic: `boundary-micro.json` in this results directory.
