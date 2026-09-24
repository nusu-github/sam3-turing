# Current native SAM3 hotspots — RTX 2060 Max-Q

Measured 2026-09-24 at base commit `908e476`, with the existing local INT8 MLP experiment and new opt-in NVTX instrumentation. No new approximation was introduced for this profile.

## Method and limits

- Native C++ / LibTorch 2.10 cu130, CUDA 13.0, SM75, FP16 compute storage; fused Vision normalization/layout and paired RoPE enabled.
- One truck image, prompt `truck`, resolution 1008, batch 1, all 200 queries, threshold 0.5. Text cached, image features recomputed. File IO and model setup excluded.
- Nsight Systems 2025.3.2, CUDA + NVTX tracing, no CPU sampling/context-switch tracing. One cold inference and five warmups precede five captured inferences per process.
- Order: **FP16 → INT8 MLP → INT8 MLP → FP16**. Ten captured inferences per mode. Tables average the two runs' per-inference kernel sums.
- CUDA launch correlation maps each kernel to its enclosing CPU NVTX ranges. Values below are **actual GPU kernel durations**, not CPU range durations. Inclusive parent ranges overlap; use the non-overlapping category table when adding values. Memory-copy API transfers and GPU idle gaps are excluded; cast/copy CUDA kernels are included.
- GPU clocks were not fixed. NVML samples with power above 40 W, including warmup/setup, span 1125–1815 MHz for FP16 (median 1320), 1170–1815 MHz for INT8 (median 1350). Consequently small differences in unchanged operations are noise, not optimization effects.
- The profiled whole-pipeline wall medians were 521.54 ms FP16 and 498.78 ms INT8. Per-process medians were 512.95 / 539.92 ms FP16 and 503.25 / 480.72 ms INT8. This short instrumented run is for attribution; use the earlier uninstrumented 30-sample comparison for the prior speed claim.

## Non-overlapping major GPU costs

All times are milliseconds per inference, across all relevant layers.

| Operation | FP16 baseline | INT8 MLP enabled |
|---|---:|---:|
| Vision MLP, including GELU and quantization | 187.12 | 143.25 |
| Vision QKV projection | 57.24 | 58.35 |
| Vision attention output projection | 19.24 | 19.35 |
| Vision global SDPA, 4 layers | 41.14 | 42.83 |
| Vision local SDPA, 28 layers | 31.98 | 33.76 |
| Vision normalization / residual / other block work | 17.57 | 19.45 |
| Vision paired RoPE | 6.36 | 6.77 |
| Vision neck | 16.41 | 17.06 |
| Vision embedding / position / other | 3.63 | 3.79 |
| Detector encoder, all work | 34.62 | 35.59 |
| Detector decoder, all work | 15.44 | 15.40 |
| Detection heads, all work | 30.47 | 32.55 |
| Other detector setup | 2.59 | 2.56 |
| Outside Vision / Detector (pre/post etc.) | 2.23 | 2.01 |
| **Total GPU kernel time** | **466.04** | **432.72** |

Vision total: 380.69 → 344.61 ms. Detector total: 83.12 → 86.10 ms, with no algorithm change there. These totals are parents of rows above, not additional costs.

CUDA-event stage intervals average 381.68 / 345.82 ms for Vision and 137.62 / 143.14 ms for Detector. Detector has roughly 55–57 ms beyond summed kernels in these traces. That difference includes host launch gaps, transfers and synchronization effects; it must not be treated as quantizable matrix arithmetic or as measured CPU compute alone.

## Where the INT8 savings go

| Vision MLP component | FP16 | INT8 |
|---|---:|---:|
| FC1 matrix multiply | 88.11 | 42.73 |
| FC2 matrix multiply | 85.11 | 34.47 |
| Separate GELU | 13.89 | — |
| FC1 input quantization | — | 5.72 |
| FC1 restore + bias + erf GELU | — | 31.20 |
| FC2 input quantization | — | 23.03 |
| FC2 restore + bias | — | 6.10 |
| **Total** | **187.12** | **143.25** |

The INT8 GEMMs themselves take **77.20 ms**, versus 173.23 ms for FP16 GEMMs. However, INT8 quantization/restoration costs **66.05 ms**, or **46.1% of the INT8 MLP path**. Net MLP improvement is 43.87 ms (23.4%), not the GEMM-only reduction of 96.03 ms.

Both INT8 traces use `cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16` on the GPU. There is no CPU fallback for these GEMMs. The direct quant/restore ranges each contain their expected custom CUDA kernel. Local/global SDPA launch counts are exactly 28 / 4 per inference.

## Next experiments, ranked from these measurements

1. **Improve the existing MLP quantization boundaries.** FC1 restore/GELU followed by FC2 input quantization costs **54.24 ms**. Today it writes a large FP16 activation and the row quantizer reads it for absmax and packing. A fused restore/GELU/row-scale/INT8-pack path is the first candidate. The reduction and register/shared-memory cost remain: 54.24 ms is the current affected region, not a promised saving. Calibrated or block scales could also reduce runtime reduction cost, but change quantization error and require separate quality checks. A GEMM epilogue approach could additionally avoid materializing the INT32 accumulator; that is a larger implementation change. FC1 input quantization (5.72 ms) is a smaller candidate for fusion with the existing norm path.
2. **Try W8A8 on Vision QKV projection**, currently **58.35 ms**; output projection adds **19.35 ms** as a separate experiment. QKV is the larger untouched projection. Keep the quantization/restore cost in the measurement, and assess accuracy separately because Q/K errors affect softmax. These traces do not establish how much faster either W8A8 projection will be.
3. **Then test Attention alternatives, starting with the four global blocks.** Global SDPA alone costs **42.83 ms**, about 10.71 ms per block, versus 1.21 ms per local block. All Vision SDPA is 76.60 ms. A QK-INT8/PV-FP16 scheme accelerates only part of that region; these fused kernels do not expose separate QK, softmax and PV durations. Do not use 76.60 ms as the potential saving from QK quantization alone. Measure an FP16 attention baseline replacement before attributing an improvement to quantization.
4. **Detector is a secondary target, with different causes.** Encoder SDPA is 24.34 ms. Detection heads take 32.55 ms, including about 13.46 ms in the dominant float-input cast/copy kernel, versus 4.61 ms in the largest cuDNN convolution kernel. Investigate dtype/layout traffic before blanket INT8 conversion. These are kernel observations, not yet proof of a memory-bandwidth bottleneck. Decoder relative-position bias is 7.09 ms, while its explicitly FP32 FFN is only **0.94 ms**: lowering that FFN's precision has little whole-model upside.

Thus the next bounded experiment should be the **MLP FC1→FC2 boundary**, followed by **QKV W8A8**, with global Attention as another substantial but more complex candidate. No new optimization was implemented in this profiling pass.

## Reproduction and validation

`experiments/profile_native_hotspots.py` captures the four runs into `.cache/native-perf/hotspots-current/`. It refuses to overwrite an existing run directory. `experiments/analyze_native_hotspots.py` attributes kernel times and writes `hotspots-current.json`, including individual kernel names, inclusive and exclusive ranges, launch counts and output hashes.

The profiler is enabled with `SAM3_PROFILE_NVTX=1`; it adds no synchronization or tensor operations. It is normally disabled, though range construction still has small CPU overhead. NVTX implementation is isolated from model headers to avoid Windows macro leakage.

Build succeeded; all 48 existing tests passed (141.11 seconds, `build/native-windows-cu130/windows-validation-profiling.log`). All four captured runs' masks, scores, boxes and query indices are byte-identical to the previous run of the **same mode**. This verifies the instrumentation; it does not mean INT8 equals FP16. Existing INT8 quality differences remain documented in `APPROXIMATION_EXPERIMENTS.md`.

Raw `.nsys-rep`, SQLite, stage intervals, metrics and NVML telemetry are retained under `.cache/native-perf/hotspots-current/`. Machine-readable per-run attribution: `experiments/results/native_rtx2060/hotspots-current.json`.
