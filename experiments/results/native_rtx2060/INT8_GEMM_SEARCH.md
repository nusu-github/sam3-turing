# SM75 INT8 GEMM search — 2026-09-24

## Decision

Keep the existing ATen INT8 GEMM path. Six standalone CUTLASS tile configurations all lose on the tested FC2/FC1/QKV shapes. Explicit cuBLASLt heuristic selection shows small operator-level differences, but caching the selected algorithm and descriptors does **not establish a reliable whole-model improvement**. None of these new paths is enabled by default.

This follows the [MLP sweep](MLP_SWEEP.md), whose FC2 counter capture showed low occupancy limited by registers/shared memory. The new experiment directly tests whether reducing those resources helps; it does not treat occupancy as a performance target by itself.

## CUTLASS tile comparison

External CUTLASS commit `df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`, BSD-3-Clause, from the existing local submodule checkout. This experiment includes CUTLASS headers only; it does not use the parent flash-attention implementation with unresolved redistribution terms. Generated compatibility header changes the unused host-only `memsetDevice` wrapper annotation for CUDA 13, retaining license text. No attention math or third-party GEMM math is edited.

Native wrapper: contiguous INT8 A[M,K], W[N,K], INT32 output [M,N], current stream/device, SM75 check, two pipeline stages, 16-element input alignment, `m8n8k16` tensor operations. The output epilogue copies INT32 accumulators without a float conversion. K is limited to 8192, so even full-range signed INT8 products fit the tested INT32 accumulation bound.

| ID | CTA M×N×K | Warp M×N×K | Registers/thread | Shared bytes | Blocks/SM | FC2 ms | FC1 ms | QKV ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ATen | existing vendor kernel | — | — | — | — | **1.2146** | **1.3987** | **0.9175** |
| 0 | 128×128×64 | 64×64×64 | 216 | 32768 | 2 | 1.3908 | 1.4998 | 0.9754 |
| 1 | 64×64×64 | 32×32×64 | 92 | 16384 | 4 | 2.7785 | 2.4123 | 1.5479 |
| 2 | 64×128×64 | 32×64×64 | 136 | 24576 | 2 | 1.6395 | 1.7250 | 1.1148 |
| 3 | 128×64×64 | 64×32×64 | 134 | 24576 | 2 | 2.6056 | 2.2024 | 1.3845 |
| 4 | 64×64×32 | 32×32×32 | 70 | 8192 | 7 | 3.2169 | 2.9154 | 1.8975 |
| 5 | 64×128×32 | 32×64×32 | 118 | 12288 | 4 | 2.0386 | 2.1484 | 1.3819 |

All custom kernels use 128 threads; reported local-memory bytes are zero and compiler output reports no spills. Resources are queried from compiled kernels on this GPU. Increasing residency to seven blocks per SM makes FC2 substantially slower. Reduced tile reuse/more work distribution overhead are possible explanations, but no candidate-specific hardware-counter capture was taken to identify a unique cause.

Shapes M,N,K: FC2=(5184,1024,4736), FC1=(5184,4736,1024), QKV=(5184,3072,1024). Two separate benchmark processes; each shape has 100 ATen warmup calls then four forward/reverse passes across all seven methods, with ten warmups/twenty timed samples per method. Table pools the eight pass medians. Allocation/host wrapper work is included in event timing. Clock/cache state is uncontrolled; do not compare these absolute operator times with older traces at other clocks.

All six configurations match `_int_mm` exactly on all three production shapes and a tail shape (80,80,128). Inputs cover random [-127,127], constant 127, and full signed random [-128,127]. Constant inputs additionally compare with K×127², above 2²⁴ on production shapes, guarding against unintended float output conversion. Work runs on a nondefault CUDA stream. Compute Sanitizer memcheck, filtered to CUTLASS kernels, completes all check-only cases with zero errors.

Data: [run 1](int8-tiles-micro-1.json), [run 2](int8-tiles-micro-2.json), [memcheck](int8-tiles-memcheck.json). Raw output roots: `.cache/native-perf/int8-tiles-micro-1`, `int8-tiles-micro-2`, `int8-tiles-memcheck`.

## Why also examine cuBLASLt

The PyTorch 2.10 source routes `_int_mm` through `at::cuda::blas::int8_gemm`, which calls cuBLASLt with INT32 compute/scale/output and automatic algorithm selection, without explicit workspace. The traced CUTLASS-style kernel name is not evidence that a standalone header-only CUTLASS instantiation is identical to the vendor implementation. Sources: [Blas.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Blas.cpp), [CUDABlas.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CUDABlas.cpp).

The separate INT8 heuristic probe requests up to 32 candidates with a 32 MiB workspace budget. CUDA 13.0 returns three candidates for each shape. All have zero actual workspace, all reproduce INT32 reference results exactly. Algorithm 21/tile 20/split-K 1 is fastest; algorithms 0 and 20 are substantially slower. This is a new INT8 search, not a repetition of the earlier FP16 MLP search.

| Shape | Run 1 ATen / best explicit, ms | Run 2 ATen / best explicit, ms |
|---|---:|---:|
| FC2 | 1.2191 / 1.2016 | 1.2090 / 1.1644 |
| FC1 | 1.3980 / 1.3905 | 1.4502 / 1.3995 |
| QKV | 0.9115 / 0.8892 | 0.9075 / 0.8894 |

These are four paired pass medians per process. The small difference could include CPU-side descriptor/setup costs and operating-condition variation; it is not proof of a faster arithmetic kernel. Data: [INT8 LT run 1](int8-lt-1.json), [run 2](int8-lt-2.json).

## Test descriptor/algorithm reuse in the model

An opt-in `int8_lt_cached` path caches descriptors and the measured zero-workspace candidate per thread/device/M/N/K. It validates SM75 and operands, uses the current stream and the PyTorch cuBLASLt handle, and rejects unavailable measured algorithm configurations. `exact` retains ATen; `fc2` changes only FC2; `all` changes every active Vision INT8 linear operation. The standalone CUTLASS tiles are **not** connected to model dispatch.

Fixed model settings: MLP boundary INT8, QKV INT8, QKV/RoPE fusion, FC2/norm fusion, existing SDPA, existing boundary kernel and flat restore. Same truck input/prompt, cached text, recomputed image features, no file/load time. Order exact/fc2/all/all/fc2/exact, each a separate process with five warmups and fifteen samples (30 samples/mode), CUDA stage events, no profiling or concurrent build/GPU job.

| Path | Median wall | p95 wall | Vision mean | Detector mean |
|---|---:|---:|---:|---:|
| Existing ATen | 403.49 ms | 411.11 ms | 282.48 ms | 115.46 ms |
| Cached LT, FC2 only | 401.83 ms | 413.52 ms | 283.83 ms | 114.37 ms |
| Cached LT, all INT8 | 404.61 ms | 422.70 ms | 283.99 ms | 117.04 ms |

Baseline process medians are 397.86/405.49 ms; FC2 400.46/404.21 ms; all 406.20/401.96 ms. FC2's pooled median is 0.41% lower, but Vision does not improve, the unchanged Detector shifts, and paired process comparisons are inconsistent. All-INT8 reuse is slower in the pooled comparison. **Do not recommend either as an established speedup.** No universal theoretical limit, all possible tiles, swizzles, layouts or pipelines has been exhausted.

All three modes retain byte-identical masks, scores, boxes and query IDs on truck/paper bag/child/wheel/empty (counts 1/4/6/4/0), compared with the preceding FC2/norm-fused baseline. Timed truck outputs also match. Existing CTest suite: **51/51 pass**, 139.07 seconds, rebuilt DLL with experimental runtime switches unset; log `build/native-windows-cu130/windows-validation-int8-search.log`. The opt-in paths are covered by the separate operator/model checks above. Python compilation and `git diff --check` pass. Data: [model comparison](int8-cache-ba018a0.json); raw runs `.cache/native-perf/int8-cache-ba018a0`.

## Reproduction and scope

```powershell
cmake -S native -B build/native-windows-cu130 -DSAM3_EXPERIMENT_INT8_GEMM=ON -DSAM3_CUTLASS_ROOT=D:/PycharmProjects/sam3-turing/.cache/portability-review/flash-attention-turing/csrc/cutlass
cmake --build build/native-windows-cu130 --target sam3_int8_gemm_bench sam3_int8_lt_bench sam3_image_latency
.venv/Scripts/python.exe experiments/monitor_native_bench.py .cache/native-perf/int8-tiles-new build/native-windows-cu130/sam3_int8_gemm_bench.exe .cache/native-perf/int8-tiles-new.json
```

The model runs used `experiments/run_int8_cache_native.py timing|quality` and
`summarize_int8_cache_native.py`, removed after the experiment and kept in commit
`7d7fddb`. Model runners refused to overwrite existing runs. Custom CUTLASS build option defaults OFF; runtime cached-LT selector `SAM3_EXPERIMENT_INT8_LT` defaults to `exact`. Earlier model runners clear the new selector. Generated compatibility headers now retain timestamps when content is unchanged, avoiding unnecessary CUDA rebuilds on reconfiguration.

No candidate is promoted to the recommended inference configuration. The evidence shifts priority away from small-tile/occupancy tuning alone and toward eliminating intermediate memory traffic or changing Attention arithmetic, each requiring its own implementation and quality evaluation.
