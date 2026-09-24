# MLP boundary and FC2 fusion experiments — 2026-09-24

## Outcome

Four implementation candidates were tested. Changing the MLP boundary thread count or explicitly using read-only loads did not improve its operator benchmark; retain the existing 256-thread implementation. FC2 restoration fused with residual addition and next-layer normalization reduces the corresponding operator time and Vision stage time, but the whole-model improvement is small: **405.11 → 402.99 ms median (0.52%)**. Keep it opt-in.

All five real-model cases preserve the preceding INT8 model's masks, scores, boxes and query IDs byte for byte. This is not equivalence to the original unquantized model. No numerical tolerance was relaxed and no additional quantization was introduced.

## Boundary: measure before changing scheduling

Nsight Compute 2025.3.1, original `restore_quant_rows<19>`, synthetic production shape [5184,4736], 256 threads, one captured launch after 100 matching launches, 12 kernel-replay passes:

- DRAM throughput **79.15%** of reported peak; compute throughput **45.29%**.
- Achieved occupancy **97.98%**, theoretical **100%**.
- Long-scoreboard memory waits account for about **61.9%** of sampled cycles between issued instructions.
- Replay SM clock 1.80 GHz; caches/clocks uncontrolled. This is a single-shape counter spot check, not an all-model roofline or a sustained-clock latency benchmark.

The evidence favors memory-traffic work over trying to increase an already high occupancy. Three alternatives retain the same Half rounding, GELU expression, per-row absmax and scales:

| Boundary implementation | Pooled operator median |
|---|---:|
| Existing 256 threads | **0.6946 ms** |
| 128 threads, 37 values/thread | 0.7134 ms |
| 512 threads, 10 values/thread | 0.7642 ms |
| 256 threads, explicit read-only scale/bias loads | 0.6989 ms |

Two processes per variant, forward/reverse order, four alternating separate/fused timing passes per process. These are medians of operator-pass medians. All existing random/zero/constant checks at widths 32/257/1024/4736/8192 pass. Candidate scheduling changes apply only to width 4736. The variants were not promoted to expensive model comparisons because no operator improvement was established.

Data: [boundary counters](boundary-counters.txt), [boundary sweep](boundary-sweep.json). Runner: `experiments/run_boundary_sweep.py`. Raw counter report: `.cache/native-perf/boundary-counters.ncu-rep`. The first capture attempts selected no kernels; only the final demangled-name capture contains the reported evidence.

## FC2: restore + residual + next normalization

The existing path writes restored FP16 FC2 output, then reads it in the residual/next-normalization kernel. The new `fc2_residual_norm_kernel` reads INT32 accumulators and scales, restores and rounds each value to FP16, adds the FP32 residual, and applies the existing Welford reduction and normalization order. It also produces the next layer's window-partitioned input where needed. No GEMM epilogue was changed in this experiment: the INT32 GEMM output is still materialized.

The runtime uses the fused path on the first 31 Vision blocks when INT8 boundary mode, fused normalization and FP16 projection are active. The last block retains its original restoration path because it has no next-layer normalization. The existing residual/normalization kernels remain unchanged. Device, shape, dtype, layout/alignment and launch checks are included; execution uses the current CUDA stream.

Five operator cases cover [1,1,1,1024], [1,72,72,1024] and [2,48,48,1024], with and without output partitioning where applicable. Random, zero and regenerated random values are tested on a nondefault stream. Residual FP32 and normalized FP16 bit representations match the separate reference. At 5184 rows, separate restore/normalization takes roughly 0.45–0.50 ms, fused 0.327–0.342 ms, including output allocation.

## Whole model

RTX 2060 Max-Q 6 GB, Windows, CUDA 13.0, MSVC 14.44, LibTorch 2.10 cu130. Base `ba018a0` with existing local experiments. Fixed settings: MLP `int8_boundary`, projection `qkv`, QKV/RoPE `fused`, Attention `exact`, restore `flat`, boundary `exact`.

| Metric | Previous QKV/RoPE-fused baseline | + FC2/norm fusion |
|---|---:|---:|
| Whole-model median | 405.11 ms | 402.99 ms |
| Whole-model p95 | 429.56 ms | 417.82 ms |
| Vision stage mean | 287.39 ms | 282.68 ms |
| Detector stage mean | 117.53 ms | 117.03 ms |
| Peak allocated bytes | 2,788,839,424 | 2,789,232,640 |

Order: baseline/fused/fused/baseline, separate processes, five warmups and fifteen measured samples each, 30 samples per mode. Baseline process medians 404.61/405.52 ms; fused 400.97/404.57 ms. Tail clocks 1230–1245 MHz; maximum temperatures 64–70 °C; no clock locking. No builds or other GPU work overlapped timing. Transfer, preprocessing, Vision, Detector and postprocessing included; model/file loading excluded; text cached, image features recomputed.

The ~4.70 ms Vision mean reduction agrees with the operator experiment. Whole-model medians improve by only ~2.12 ms; do not promise a large gain or a stable tail-latency improvement from this small sample. Do not subtract the preceding task's ~401 ms result from this series: operating conditions differ.

Truck/paper bag/child/wheel/empty detection counts remain 1/4/6/4/0. All four output files match the previous QKV/RoPE-fused baseline byte for byte, including timed truck runs. Compute Sanitizer memcheck and racecheck cover all five operator cases, filtered to the new FC2 kernel: zero errors, warnings or hazards.

Data: [model comparison](mlp-sweep-ba018a0.json), [operator comparison](fc2-norm-micro.json), [memcheck](fc2-norm-memcheck.json), [racecheck](fc2-norm-racecheck.json). Raw runs: `.cache/native-perf/mlp-sweep-ba018a0`.

## What remains in the GEMM

A separate Nsight Compute capture of the first FC2 INT8 GEMM in the real model verifies NVTX stack `vision.block.0/vision.mlp.fc2/vision.mlp.fc2.int8_gemm` and kernel `cutlass_75_tensorop_i8816gemm_s8_128x128_tn_align16`:

- Achieved occupancy **23.93%**, theoretical **25%**, limited to two blocks/SM by both registers and shared memory.
- Compute throughput **62.72%**, DRAM **37.24%**, L1/TEX **85.47%** of their reported peaks.
- Average eligible warps/scheduler 0.30; no eligible warp in about 77% of scheduler cycles.
- Replay clock 1.69 GHz, duration 0.782 ms, 12 passes; these timings are not substituted for sustained inference measurements.

This is a different constraint from the boundary kernel. An SM75 INT8 GEMM tile/pipeline experiment is justified; increasing occupancy alone does not guarantee a gain, and the high L1/TEX activity also matters. The profiler's estimated speedup is not a measured result. No universal percentage of whole-model theoretical maximum has been established.

Counter evidence: [FC2 INT8 counters](fc2-int8-counters.txt), `.cache/native-perf/fc2-int8-counters.ncu-rep`. Output still matches the corresponding unprofiled model. Nsight replay backed up GPU memory to host RAM; its ~5.77 GB device-memory observation and ~1.99 s profiled sample are profiler overhead, not runtime regressions.

## Reproduce and switches

```powershell
cmake --build build/native-windows-cu130 --target sam3_boundary_bench sam3_fc2_norm_bench sam3_image_latency
.venv/Scripts/python.exe experiments/run_boundary_sweep.py
.venv/Scripts/python.exe experiments/run_mlp_sweep_native.py timing
.venv/Scripts/python.exe experiments/run_mlp_sweep_native.py quality
.venv/Scripts/python.exe experiments/summarize_mlp_sweep_native.py
```

Model driver refuses to overwrite previous runs. Both new switches default to `exact`. On the existing tested INT8/QKV/RoPE configuration, opt in with `SAM3_EXPERIMENT_FC2_NORM=fused`; keep `SAM3_EXPERIMENT_BOUNDARY=exact`. The rejected boundary candidates remain diagnostic options `threads128`, `threads512`, `readonly`. Earlier benchmark drivers clear these switches to preserve their prior comparisons.

Existing CTest suite: **51/51 pass**, 138.62 seconds, using the rebuilt DLL with new experiment switches unset. Log: `build/native-windows-cu130/windows-validation-mlp-sweep.log`. The new paths are exercised separately by the operator/model comparisons above. Python compilation and `git diff --check` also pass. These validations do not establish all-image, all-batch, cross-platform or video parity.
