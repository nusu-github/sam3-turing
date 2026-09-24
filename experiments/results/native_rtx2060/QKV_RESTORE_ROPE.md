# QKV INT8 restoration + RoPE fusion — 2026-09-24

## Result

An opt-in native CUDA kernel reduces median image inference from **409.43 to 401.04 ms (2.05%)**, on the same INT8 MLP/QKV baseline and existing SDPA. All five model cases retain byte-identical masks, scores, boxes and query IDs. Keep this independent of the previous external Attention experiment, whose model box gate failed.

The experiment adds no further quantization or relaxed numerical tolerance. It preserves the previous INT8 path's FP16 rounding before RoPE and Windows complex-multiply order. This is parity with the existing approximate INT8 model, not a claim of equality to the original FP16 model.

## Implementation

`approx_restore_rope` in `native/tools/approx_kernels.cu` takes INT32 QKV GEMM output, per-row activation scales, per-channel weight scales, FP16 bias and complex FP32 frequencies. Each CUDA thread restores two adjacent channels. Q/K values round to FP16 before rotating; V is restored without rotation. The real component uses the same complex multiplication as `rotary_pair_cuda.cu`; the imaginary component preserves that file's explicit platform-dependent FMA order.

Outputs share a `[3,B,N,16,64]` allocation, exposed as BHND views. Q/K need no intermediate packed write/read, and V is contiguous in token-major order. Compared with separate restore and RoPE, this removes about 40.5 MiB of Q/K intermediate traffic per tested Vision block (5184 total tokens), plus one launch per block. The INT8 quantization and GEMM are unchanged. The wrapper checks operands, guards the CUDA device, uses the current stream and checks launch errors. It uses no shared memory or cross-thread communication.

`SAM3_EXPERIMENT_QKV_ROPE=fused` opts in; unset or `exact` keeps the old path. Fusion requires QKV INT8 weights (`SAM3_EXPERIMENT_PROJECTION=qkv` or `both`). The tested configuration is `qkv`, with existing MLP boundary fusion, original flat restore elsewhere, and Attention `exact`. No external donor is needed for this kernel.

## Measurements

RTX 2060 Max-Q 6 GB, Windows, CUDA 13.0, MSVC 14.44, LibTorch 2.10 cu130. Repository base `ba018a0` plus the existing local experiments and this change.

| Metric | Separate restore + RoPE | Fused |
|---|---:|---:|
| Whole model, pooled median | 409.43 ms | 401.04 ms |
| Whole model, p95 | 418.96 ms | 423.88 ms |
| Vision mean | 290.17 ms | 282.24 ms |
| Detector mean (unchanged) | 115.64 ms | 116.45 ms |
| Peak allocated bytes | 2,788,970,496 | 2,788,839,424 |

Four independent processes in exact/fused/fused/exact order, each five warmups and fifteen samples. Process medians: exact 410.05/409.26 ms; fused 398.67/402.95 ms. Each mode has 30 measured samples. Includes transfer, preprocessing, Vision, Detector and postprocessing, cached text features, recomputed image features; excludes model load and file I/O. Stage events are enabled, Nsight disabled for these timing runs. Tail clock medians 1245–1275 MHz, maximum temperature 62–68 °C, no clock locking. The gain is supported by both process comparisons and the Vision stage; **tail latency did not improve** in this small sample.

Operator microbenchmark includes allocations and compares four alternating passes. B=1,N=5184: separate 0.664–0.685 ms, fused 0.451–0.471 ms. B=9,N=576: separate 0.744–0.842 ms, fused 0.502–0.527 ms. Clock ramp affects the local passes; use whole-model paired measurements for the practical estimate.

Separate Nsight captures (five warmups, five recorded samples each) attribute GPU kernel durations through CPU launch correlation IDs and NVTX ranges. Across all 32 layers, existing QKV restore **17.706 ms** plus RoPE **5.981 ms** becomes fused restore/RoPE **14.596 ms**, a **9.091 ms reduction (38.4%)**. QKV INT8 GEMM remains 26.44/26.53 ms, and input quantization 5.30/5.34 ms. This supports the intended source of the whole-model gain. Trace wall times include instrumentation and are not substituted for unprofiled benchmark results. Both captures preserve the prior binary outputs.

Trace data: [separate](hotspots-rope-exact.json), [fused](hotspots-rope-fused.json). The remaining largest Vision categories in this trace are global/local SDPA (41.55/32.22 ms), MLP FC1/FC2 GEMM (40.40/32.95 ms), QKV GEMM (26.53 ms), and MLP boundary processing (23.64 ms). These are measured categories, not a claim that every category can be accelerated without tradeoffs.

## Validation

- Four operator shapes: (B,N)=(1,1),(2,129),(9,576),(1,5184), H=16,D=64. Random, zero and regenerated random accumulators/scales/bias, nontrivial complex frequencies. Nondefault CUDA stream. Q/K/V match, and changing V's layout leaves the following SDPA output unchanged.
- The final check-only executable compares the FP16 bit representations as `int16`, including signed zero. All cases pass under Compute Sanitizer memcheck, zero errors, filtered to the new `restore_rope` kernel. The earlier timing executable used numerical tensor equality; the kernel itself was unchanged between those runs.
- Model cases: truck 1 detection, paper bag 4, child 6, wheel 4, empty 0. All masks/scores/boxes/query files match the previous QKV INT8 baseline byte for byte, for both separate and fused modes. Timed truck outputs also match. No tolerance gate was relaxed.
- Existing CTest suite: **51/51 pass**, 138.09 seconds, with the rebuilt DLL and the new opt-in setting unset. Log: `build/native-windows-cu130/windows-validation-qkv-rope.log`. The fused path is exercised by the separate operator and real-model tests above. Python compilation and `git diff --check` also pass.

Durable data: [model timing and parity](qkv-rope-ba018a0.json), [operator timings](qkv-rope-micro.json), [memcheck unit results](qkv-rope-memcheck.json). Raw outputs and telemetry: `.cache/native-perf/qkv-rope-ba018a0`; sanitizer log: `.cache/native-perf/qkv-rope-memcheck/process.log`.

## Reproduce

Using the existing configured Windows native build:

```powershell
cmake --build build/native-windows-cu130 --target sam3_qkv_rope_bench sam3_image_latency
```

The model comparisons used `experiments/run_qkv_rope_native.py timing|quality`
and `summarize_qkv_rope_native.py`, which failed if any required binary output
differed. These drivers were removed after the experiment and remain in commit
`7d7fddb`. Runtime configuration:

```powershell
$env:SAM3_EXPERIMENT_MLP='int8_boundary'
$env:SAM3_EXPERIMENT_PROJECTION='qkv'
$env:SAM3_EXPERIMENT_ATTENTION='exact'
$env:SAM3_EXPERIMENT_QKV_ROPE='fused'
```

These switches remain experimental. Validation covers the listed image cases, not all inputs, GPU architectures, batches or video trajectories. Combining with external global Attention requires separate validation and does not erase that experiment's quality failure.
