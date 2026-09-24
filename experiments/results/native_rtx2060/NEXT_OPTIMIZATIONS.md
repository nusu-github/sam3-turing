# Further optimization search after MLP boundary fusion

2026-09-24. RTX 2060 Max-Q, SM75, LibTorch 2.10 cu130, upstream base `ba018a0` plus local Windows/INT8 experiments. All configurations below use `SAM3_EXPERIMENT_MLP=int8_boundary`.

## Findings

**QKV W8A8 is a repeatable but small additional improvement: about 1.8–1.9% of inference time.** Quantizing the attention output projection did not help. Reindexing restoration by row, to avoid per-element integer division, did not establish an additional whole-model improvement worth recommending.

No approximation is enabled by default. The new opt-in `SAM3_EXPERIMENT_PROJECTION` accepts `exact` (default), `qkv`, `proj`, or `both`. It quantizes the selected Vision attention weights per output channel once at setup, dynamically quantizes inputs per row, runs GPU INT8 GEMM and restores FP16 with bias. Attention QK/softmax/PV remain FP16/FP32 as before; this is projection quantization, not INT8 Attention.

`SAM3_EXPERIMENT_RESTORE=rows` is a separate experimental switch. It changes the restoration launch to a 2D grid, keeping the same expression and FP16 rounding. Unset or `flat` uses the original kernel. Inputs with more than 65535 rows retain the original launch. Keep this experiment disabled for now.

## Updated kernel attribution

Nsight Systems, five warmups and five captured Truck inferences per trace. Actual GPU kernel sums, correlated from CUDA launch to NVTX, exclude idle gaps and API memory transfers. These are attribution traces, not the wall-time comparison below.

| Region | Fused MLP baseline | + QKV INT8 |
|---|---:|---:|
| QKV projection total | 55.30 ms | 49.43 ms |
| QKV INT8 GEMM | — | 26.47 ms |
| QKV input quantization | — | 5.28 ms |
| QKV restoration | — | 17.68 ms |
| Global Vision SDPA (4 layers) | 40.25 ms | 40.58 ms |
| Local Vision SDPA (28 layers) | 31.36 ms | 31.86 ms |
| Vision output projection | 19.01 ms | 18.95 ms |
| Fused MLP boundary | 23.50 ms | 23.38 ms |

QKV GEMM is much faster, but **22.96 ms of quantization/restoration** consumes most of its gain. A future restore+RoPE fusion is a concrete candidate: the separate RoPE path costs another ~6 ms. This potential saving is not measured, and a fused implementation must preserve/test rounding and layout. Global Attention remains a larger untouched region (~40 ms across only four layers); separate QK/PV costs cannot be extracted from this fused SDPA kernel trace.

## Same-build timing comparisons

Truck image/prompt, batch 1, resolution 1008, 200 queries, threshold 0.5, cached text, no image-feature reuse. Setup/file IO excluded. No Nsight/NVTX in timing runs. Each process uses one cold inference, five warmups and 15 measured inferences; each variant appears twice in forward/reverse order, giving 30 samples per variant.

First series order: baseline → QKV → output projection → both → both → output projection → QKV → baseline.

| Projection quantized | Median wall | p95 wall | Mean Vision stage |
|---|---:|---:|---:|
| None (fused MLP baseline) | 454.37 ms | 470.24 ms | 324.81 ms |
| QKV only | **446.09 ms** | 456.50 ms | **315.87 ms** |
| Output only | 454.55 ms | 464.94 ms | 326.08 ms |
| Both | 449.87 ms | 465.83 ms | 317.70 ms |

QKV alone saves 8.28 ms (1.82%). Output-only is effectively unchanged; both is worse than QKV alone. Extra quantization of the smaller output projection is not justified by these results.

Second series order: baseline → row restore → QKV → QKV+row restore → reverse order. This independently repeats the baseline/QKV comparison while testing row-indexed restoration.

| Configuration | Median wall | p95 wall | Mean Vision stage |
|---|---:|---:|---:|
| Baseline | 419.50 ms | 433.89 ms | 300.40 ms |
| Row restoration only | 423.07 ms | 437.26 ms | 301.29 ms |
| QKV INT8 | **411.67 ms** | 422.54 ms | 292.63 ms |
| QKV INT8 + row restoration | 411.06 ms | 421.61 ms | 290.47 ms |

QKV saves 7.83 ms (1.87%) in this second series. Adding row restoration changes the wall median by only 0.60 ms (0.15%) versus QKV; that is too small relative to observed run variation to claim a reliable improvement. Row restoration alone is slower in this comparison.

Do not subtract wall times across the two series: clocks/system conditions were not fixed. Within-series order reversal reduces drift but is not a formal significance test. Per-process medians, samples, clock/temperature observations and stage intervals are retained in the JSON reports. QKV's similar benefit in both series and the kernel attribution support the modest gain, not a large speed claim.

QKV adds approximately **96 MiB** to peak allocated memory (2,688,241,664 → 2,788,970,496 bytes), since this diagnostic implementation retains the original FP16 weights alongside INT8 weights/scales.

## Quality

All projection variants retained detection counts on the five smoke cases: Truck 1, bag 4, child 6, wheel 4, empty 0. Output is not bit-identical after projection quantization.

| Projection quantized | Minimum matched mask IoU vs FP16 | Maximum score error | Maximum box-coordinate error |
|---|---:|---:|---:|
| None (existing INT8 MLP) | 0.989715 | 0.0112305 | 0.64856 px |
| QKV | 0.990915 | 0.0097656 | 0.35895 px |
| Output | 0.991888 | 0.0063477 | 0.28137 px |
| Both | 0.989424 | 0.0058594 | 0.83112 px |

These are differences from the FP16 implementation, not ground-truth accuracy. Slightly better aggregate errors on a few cases do not demonstrate that quantization improves quality. Full per-case IoU matching, unmatched counts and comparison to the preceding INT8 model are in `projection-ba018a0.json`.

The row-restoration kernel passed the existing standalone random/zero/constant checks at widths 32, 257, 1024, 4736 and 8192, with identical quantized activations/scales to the fused-boundary reference. All timing-run Truck outputs and all five quality cases in both row-restoration modes match their corresponding flat-restoration mode byte-for-byte (masks, scores, boxes and query indices).

## Reproduction

Builds succeeded and all **51 component tests passed** (135.81 seconds). Test log: `build/native-windows-cu130/windows-validation-projection.log`. Baseline timing outputs remain byte-identical to the preceding boundary-fusion build. The optional projection paths were also exercised by 30 timed inferences and five quality cases per variant; row-restoration parity was checked separately as described above.

- `run_projection_native.py timing|quality` and `summarize_projection_native.py`: `.cache/native-perf/projection-ba018a0/`, `projection-ba018a0.json`.
- `run_restore_rows_native.py timing|quality` and `summarize_restore_rows_native.py`: `.cache/native-perf/restore-rows-ba018a0/`, `restore-rows-ba018a0.json`.
- `profile_native_hotspots.py OUTPUT_ROOT MODE...` now accepts a separate root/mode list. `analyze_native_hotspots.py TRACE_ROOT REPORT_JSON REFERENCE_ROOT` accepts a matching output reference directory or benchmark root.
- Profiles: `.cache/native-perf/hotspots-boundary/`, `.cache/native-perf/hotspots-qkv/`; exported attribution: `hotspots-boundary.json`, `hotspots-qkv.json`.

Recommended experimental combination: `SAM3_EXPERIMENT_MLP=int8_boundary`, `SAM3_EXPERIMENT_PROJECTION=qkv`, with `SAM3_EXPERIMENT_RESTORE` unset. The original fused-MLP-only mode remains the lower-memory option without additional projection quantization error.
