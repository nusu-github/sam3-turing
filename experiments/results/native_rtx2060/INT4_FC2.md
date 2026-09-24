# Native SM75 INT4 FC2 experiment — 2026-09-24

## Decision

Keep INT4 opt-in. All 32 FC2s on W4A4 reduce the paired whole-model median from **414.34 to 396.53 ms (4.30%)**, but cause large mask/score/box deviations. Limiting INT4 to the last 1, 2, 4 or 8 blocks does not pass every fixed FP16 quality gate. Smaller scopes also do not establish a reliable model speedup in this run. The current INT8 configuration remains the recommended measured path.

## Scope and implementation

This is an opt-in W4A4 experiment on the second MLP projection (FC2), built using the separately licensed external CUTLASS checkout already described in [INT8 GEMM search](INT8_GEMM_SEARCH.md). It executes native SM75 `m8n8k32` signed INT4 tensor operations with INT32 accumulation. There is no CPU GEMM fallback, and this is not a weight-only decompression-to-FP16 path.

`native/tools/int4_experiment.cu` supplies per-row symmetric quantization with scale `max(abs(x))/7`, a `1e-12` floor, nearest-even rounding, and values in [-7,7]. Adjacent values occupy the low/high nibble of a byte, owned by one CUDA thread. A fused FC1 INT32 restore/GELU/FP16-round/INT4-pack kernel supplies FC2 activations directly. FC2 weight scales are per output channel. Existing FC2 residual/next-Norm fusion consumes the resulting INT32 accumulators and scales unchanged.

The current FP16 weights remain resident for the experiment; the selected FC2 INT8 copies are replaced by packed INT4 copies. This is not a claim of a fully compressed model or deployment package.

`SAM3_EXPERIMENT_INT4_FC2` selects `exact` (default), `all` (32 FC2s), `last8` (24–31), `last4` (28–31), `last2` (30–31), or `last` (31). Other projections remain on the preceding INT8/FP16 paths. The non-exact modes require the optional `SAM3_EXPERIMENT_INT8_GEMM` CUTLASS development build and MLP `int8_boundary`. Production defaults remain unchanged.

## Operator validation and speed

The standalone benchmark uses a nondefault CUDA stream, a tail shape (M,N,K)=(80,80,128) and production FC2=(5184,1024,4736). It checks random, zero and regenerated random FP16 quantization against an independent CPU float-division/nearest-even reference. ATen elementwise division can use reciprocal multiplication and differs at rounding ties, so its rounded quantization expression is not used as a bit-exact reference. Packed random bytes additionally exercise the full signed nibble range including -8. Both INT4 GEMM tiles match INT8 GEMM on the unpacked integer values exactly. Fused restore/GELU/quant outputs and scales match separate restore/GELU then INT4 quantization exactly.

Two separate benchmark processes each run four forward/reverse passes with 10 warmups and 20 event samples per method after 100 baseline GEMM warmups. Input quantization and packing are excluded from this GEMM-only measurement; allocation/wrapper calls are included. No simultaneous build/GPU job or clock changes were used.

| FC2 GEMM | Median of eight pass medians |
|---|---:|
| ATen INT8 on identical unpacked values | 1.16732 ms |
| INT4 CTA128×128×128, warp64×64×128 | **0.689024 ms** |
| INT4 CTA64×128×128, warp32×64×128 | 0.924568 ms |

The first INT4 tile is 40.97% faster (1.69× throughput) in this operator test and is selected for model evaluation. It compiles with 216 registers/thread; the smaller tile uses 136. Both report zero stack/spills. Quantization and fused boundary kernels use 51/52 registers and 48 shared bytes, with zero spills.

Data: [operator run 1](int4-micro-1.json), [run 2](int4-micro-2.json), [correctness](int4-check.json). Raw logs/telemetry: `.cache/native-perf/int4-*`.

## Quality gates fixed before evaluation

Five cases: truck, paper bag, child, wheel, empty. Gates: equal detection count, minimum matched-mask IoU >=0.98, maximum score difference <=0.02, maximum box coordinate difference <=1 pixel, finite outputs. Masks are matched by maximum total IoU; differences are not restricted to identical query IDs. Evaluate against both the preceding optimized INT8 output and the FP16 reference, because a small incremental error may accumulate with earlier quantization.

| INT4 FC2 scope | Cases passing vs INT8 | Cases passing vs FP16 | Worst IoU vs FP16 | Max score difference vs FP16 | Max box difference vs FP16 |
|---|---:|---:|---:|---:|---:|
| All 32 | 1/5 | 1/5 | 0.47504 | 0.38818 | 31.97161 px |
| Last 8 | 2/5 | 3/5 | 0.96728 | 0.02344 | 1.96704 px |
| Last 4 | 4/5 | 3/5 | 0.97280 | 0.00879 | 1.07556 px |
| Last 2 | 4/5 | 4/5 | 0.97433 | 0.00732 | 0.75867 px |
| Last 1 | 5/5 | 4/5 | 0.97373 | 0.00977 | 0.66687 px |

The last-one mode passes the incremental INT8 comparison but fails the original FP16 wheel-mask gate. No tested scope passes all five cases against both references. Empty detections contribute one passing case; that alone is not evidence of useful quality. Small masks are particularly sensitive in this fixture, and quality does not improve monotonically with fewer quantized blocks. No threshold was relaxed after seeing these results. Five cases are a diagnostic fixture, not a dataset-wide accuracy evaluation.

## Whole-model timing

Same-build order: exact/all/last8/last4/last2/last, then reverse. Each is a separate process, 5 warmups + 15 timed samples (30 per mode), truck image/prompt, cached text, fresh visual features, CUDA stage events. Fixed settings: MLP `int8_boundary`, QKV INT8, fused QKV/RoPE, fused FC2/Norm, existing SDPA, default boundary kernel, flat restore, ATen INT8 GEMM. This includes activation quantization/packing and restoration, unlike the GEMM-only table.

| INT4 scope | Pooled median wall | p95 wall | Vision mean | Detector mean |
|---|---:|---:|---:|---:|
| None / INT8 baseline | 414.34 ms | 430.96 ms | 291.53 ms | 117.36 ms |
| All 32 | **396.53 ms** | 411.84 ms | **275.61 ms** | 116.37 ms |
| Last 8 | 413.68 ms | 431.20 ms | 289.36 ms | 119.23 ms |
| Last 4 | 409.81 ms | 421.24 ms | 292.09 ms | 114.92 ms |
| Last 2 | 414.58 ms | 424.38 ms | 292.85 ms | 117.25 ms |
| Last 1 | 413.61 ms | 428.13 ms | 292.65 ms | 118.25 ms |

The all-32 process medians are 391.03/400.53 ms versus baseline 407.80/415.67 ms; both process comparisons favor INT4. Vision mean drops by 15.93 ms, consistent with changing its FC2s. The last-four wall median improves while Vision does not; unchanged Detector shifts, so do not attribute that wall difference to a useful INT4 speedup. Small-scope effects are not established by these samples. No clock locking or confidence interval was used; these data do not predict performance on other conditions/hardware.

Reported PyTorch peak allocated bytes fall from 2,789,232,640 to 2,703,289,856 for all-32 (81.96 MiB), with original FP16 weights still resident. This is process allocator accounting, distinct from sampled whole-device NVML peaks.

The full quality and timing data are in [the machine-readable report](int4-fc2-ba018a0.json).

## Sanitizer and regression checks

Compute Sanitizer memcheck on the full operator benchmark reports zero errors. Racecheck restricted to the new `quant4` kernels (ordinary quantization and fused boundary) reports zero hazards/errors/warnings. Both finish all operator cases, including the production shape. Data: [memcheck](int4-memcheck.json), [racecheck](int4-racecheck.json). The CUTLASS GEMM kernels were memory-checked and numerically checked; this round does not claim a separate racecheck of those library kernels.

The default INT8 model outputs remain byte-identical to the preceding FC2/norm-fused baseline on all five cases. Python compilation and `git diff --check` pass. Existing CTest suite: **51/51 pass**, 139.58 seconds, with experiment selectors unset. Log: `build/native-windows-cu130/windows-validation-int4.log`. This suite validates default-path regression; the INT4 paths are covered by the separate operator and image checks above. The complete model matrix contains 30 quality runs and 12 timing runs (180 timed samples).

## What this establishes

Native INT4 arithmetic and the previous fusions coexist on this SM75 GPU. The speed opportunity is real for the operator, but simple row/channel absmax W4A4 is not accurate enough for broad FC2 replacement. The next INT4 work should change the quantization scheme rather than merely expand its scope: activation/weight error isolation, finer scale groups or calibration/preconditioning are possible experiments, not validated fixes. Extra scale/reduction work must be included in timing. No claim is made that FP16-level quality is impossible with another INT4 scheme.

## Reproduction

Build `sam3_int4_bench` and `sam3_image_latency` with the existing CUDA 13 / MSVC 14.44 development configuration. The model runs used `experiments/run_int4_native.py quality|timing exact all last8 last4 last2 last` and `summarize_int4_native.py`; these drivers were removed after the experiment and remain in commit `7d7fddb`.

The driver refused to overwrite an existing run directory. The summarizer intentionally records failed gates without suppressing performance data: these runs explore tradeoffs, not deployment acceptance. It can summarize partial runs; check the matrix is complete before drawing a final conclusion. Raw model outputs/telemetry are under `.cache/native-perf/int4-fc2-ba018a0`. Modes are opt-in, no model defaults were changed, and no commit/push was made.
