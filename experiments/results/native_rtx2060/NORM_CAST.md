# Fused LayerNorm-to-FP16 prototype

Historical experiment: superseded locally by upstream `571edc8`/`908e476` on
2026-09-24. The old `SAM3_FUSED_NORM_CAST` switch is no longer used by the current
runtime. See [PERFORMANCE.md](../../../docs/native/PERFORMANCE.md#vision-normlayoutresidual-fusion) for the
upstream implementation and CMake controls. The old prototype is preserved in
the pre-update stash and `.cache/local-prototype-before-908e476`.

Implemented and measured 2026-09-23 on Windows, RTX 2060 Max-Q, standalone
LibTorch 2.10.0+cu130. This opt-in prototype removes the intermediate FP32
LayerNorm output and its separate FP16 conversion before vision QKV/MLP GEMMs.
It preserves FP32 normalization/affine arithmetic and rounds to FP16 at the
same boundary as the existing linear autocast path. It does not use FP16
statistics, approximate GELU, or a different GEMM algorithm.

## Measurements

Six fresh processes ran OFF/ON/ON/OFF/OFF/ON, each with one cold inference,
three warmups and ten measured truck inferences. Resolution 1008, threshold
0.5, all 200 queries, cached text, fixed FP16 compute-weight storage, four CPU
threads, TF32 and cuDNN benchmark disabled. No concurrent builds/GPU jobs.

| Measurement | Original | Fused |
|---|---:|---:|
| Overall pooled median, 30 samples each | 515.00 ms | 505.25 ms |
| Process medians | 509.00 / 514.50 / 518.61 ms | 505.45 / 499.67 / 506.04 ms |
| Vision CUDA interval mean | 396.42 ms | 385.62 ms |

Observed overall latency reduction is **1.89%** (9.75 ms). Vision mean falls
by 10.80 ms. This is a modest local improvement with visible run-to-run
variation, not a universal speedup claim. Historical approximately 560 ms runs
must not be treated as the control: even the OFF runs are faster in this batch.
The inference allocation peak did not decrease on this workload.

For synthetic `[5184,1024]` input, the unit diagnostic measured approximately
0.341 ms for LayerNorm plus cast and 0.186 ms fused. That diagnostic is a
single sequential microbenchmark; the complete-model comparison above is the
main performance evidence. A separate Nsight trace confirms 64 fused kernel
launches per inference, replacing the two normalizations in each of 32 blocks.

## Correctness and scope

- 42/42 CTests pass with the prototype enabled.
- 29 normalization fixtures pass on a non-default CUDA stream: empty and large
  batches, zero/small/large scales, offset inputs, noncontiguous and unaligned
  layouts, infinity and NaN. Finite results are compared bitwise; NaNs are
  compared by classification. Invalid width is rejected separately.
- All six truck benchmark runs preserve masks, scores, boxes and query indices
  byte for byte against the existing native result.
- Four more image/prompt cases (paper bag, child, wheel, empty elephant) also
  preserve those four output files byte for byte, including counts 4/6/4/0.
- The existing actual-weight compute-storage probe compares fusion OFF/ON for
  batches 1 and 2, including trunk, all available pyramid heads/positions and
  text fixtures. Every binary output and tensor metadata file matches exactly.

This covers local SAM3 image fixtures, not SAM3.1, video, arbitrary inputs or
other LibTorch/GPU/platform combinations. The prototype therefore remains
opt-in rather than changing the runtime default.

## Enable and reproduce

Build the CUDA runtime and `sam3_norm_cast_test` with the existing Windows
toolchain. In a fresh process, set:

```powershell
$env:SAM3_FUSED_NORM_CAST = '1'
```

Then run the existing `sam3_image_latency` command. Set the variable to `0` or
leave it unset for the original path. The setting is read once per process.
Only fixed-FP16 vision instances use it; CPU, FP32/BF16 and original
precision-switchable constructors retain their existing paths. The specialized
kernel requires width 1024, contiguous/aligned FP32 inputs and affine tensors;
unsupported layouts use the original ATen operations. Epsilon remains 1e-5.
The public predictor/C ABI signatures and class layouts do not change.

Source: `native/src/norm_cast_cuda.cu` and the guarded normalization calls in
`vision_encoder.cpp`. Welford reduction code is adapted from PyTorch v2.10.0;
the upstream copyright/license is retained in `native/third_party/PYTORCH_LICENSE`.

[norm-cast.json](norm-cast.json) contains raw samples, output checks, feature
comparisons, the unit diagnostic log and kernel summary. Local raw artifacts
are in `.cache/native-perf/norm-*`; the full test log is
`build/native-windows-cu130/windows-validation-norm.log`.
