# Exact optimizations in the native runtime

These optimizations are enabled by default and preserve outputs: every one was
checked byte-for-byte against the preceding native build on actual-weight
feature tensors and the owning C/C++ regression cases. None changes weights, prompts, query or object counts,
image resolution or tracking policy. Native-to-native exactness does not remove
the remaining FP16 difference from the original Python source
([VALIDATION.md](VALIDATION.md)).

Unless noted, timings come from an RTX PRO 4500 Blackwell (Linux, LibTorch 2.10.0
cu130), FP16 batch 1, alternating independent baseline/current processes with
TF32 disabled and four CPU threads. They are component timings, not
whole-predictor throughput, and do not predict Turing performance. Measurements
on the RTX 2060 are in
[experiments/results/native_rtx2060](../../experiments/results/native_rtx2060/README.md).

## Summary

| Change | Where | Effect (median) | Evidence |
|---|---|---|---|
| FP16 compute-parameter residency | `VisionEncoder`/`TextEncoder` Half storage | Vision −4.9 to −5.0%, text −18 to −20%; parameters 3.05 → 1.64 GiB (SAM3) | `compute-storage-validation.json` |
| Generate only consumed position maps | `VisionEncoder::forward(..., levels)` | Encoder −2.0 to −2.4%; peak extra 264.9 → 154–213 MiB | `vision-position-selection-validation.json` |
| Fused rotary conversion + complex multiply | `sam3/rotary.h` | Vision/grounding/interactive encoders −3.3 to −3.6% | `rotary-fusion-validation.json` |
| Norm/window/residual/next-norm fusion, paired Q/K RoPE | `sam3/vision_fusion.h`, `sam3/rotary_pair.h` | Whole vision 57.8 → 53.3 ms (SAM3), −7.7 to −7.8% | `vision-fusion-validation.json` |
| In-place exact GELU on the fresh FC1 output | `vision_block.cpp` | Whole vision −2.6% (batch 1), −0.4 to −0.5% (batch 2) | `vision-gelu-validation.json` |
| Per-axis position tables; FP32 norm fusion | `sam3/vision_position.h` | Whole vision −2.0 to −2.9% (FP16), −0.7 to −1.1% (FP32) | `position-fusion-validation.json` |
| Direct interpolation into the result tensor | `postprocess_image` | Postprocessing −8% (640×540) to −26% (4K) | `image-postprocess-copy-validation.json` |
| Reuse text encoding for identical semantic text | owning `VideoPredictor` | Same-text `add_prompt` 804.6 → 133.5 ms (SAM3), 877.8 → 227.2 ms (SAM3.1) | `semantic-text-validation.json` |
| zlib CRC32 for weight integrity | `WeightStore::read` | Full store check 18.8 → 2.6 s; module loading −83% | [WEIGHTS.md](WEIGHTS.md) |
| Packed-cache fetch without rewrite | `VideoPredictor` output cache | Packed CPU fetch −19%, packed disk −38% | [OUTPUT_CACHE.md](OUTPUT_CACHE.md) |
| Indexed seeking with a verified RGB window | `MediaSource` | Full reverse read of a 360-frame H.264 fixture: 64,980 → 544 decoded frames | [MEDIA_IO.md](MEDIA_IO.md) |

Combined vision fusion (norm/QK, GELU, positions) measured directly against
`3a58138`: SAM3 57.922 → 50.745 ms (−12.39%), SAM3.1 59.858 → 52.606 ms
(−12.12%). Percentages from separate rows must not be added.

## Details and invariants

### FP16 compute-parameter residency

FP16 CUDA contexts and owning predictors keep vision/text linear and convolution
parameters in Half. The loader reads and CRC-checks one tensor, transfers it,
then converts on the device with the same conversion as the previous autocast
path, so no full FP32 GPU module is held during loading. Layer norms, positional
values, text embeddings and complex RoPE frequencies keep their source precision.
FP32, BF16-reference and CPU paths keep FP32 parameters.

C++ callers opt in with the four-argument constructors
(`VisionEncoder(store, model, device, at::kHalf)`); such instances reject
non-FP16 forward modes. The C ABI selects it automatically for FP16 CUDA
contexts. Owners that release text weights after encoding keep only the vision
part of the 1.412 GiB saving.

### Position maps

Detection and tracking consume only level 2 (72×72); interactive image decoding
consumes none. High-level callers request only those levels through the
four-argument `forward` overload; unrequested entries are undefined tensors and
the returned vector keeps its length. An image created with grounding capability
keeps level 2 even if its first request is interactive.

The position encoding itself computes FP32 sine/cosine tables per axis and
broadcasts/casts directly into the final channels-last output
(`sam3::vision_position_encoding`), instead of materializing full-resolution FP32
intermediates. At FP16 `[1,256,288,288]` the operator alone goes from 1.267 ms /
215 MB peak extra to 0.125 ms / 43 MB. CPU, empty and other dtypes use the
original ATen expression. `SAM3_FUSE_VISION_POSITION=OFF` restores the old path
in `VisionEncoder`.

### Rotary embedding

`rotary_embedding` fuses the FP32 conversion, the `c10::complex<float>`
multiplication and the conversion back into one kernel for FP32/FP16/BF16 values
with a contiguous last dimension. Other cases (CPU, FP64, conjugate views, empty
tensors) use the ATen expression. Output strides follow TensorIterator's
allocation rules, which matter for size-1 dimensions.

The imaginary component's FMA order must match the running LibTorch:
`fma(b,c,round(a*d))` on Linux and `fma(a,d,round(b*c))` on Windows (found on
the RTX 2060, where the ordinary expression differed on 330,340 of one million
products). Re-run the exact rotary tests after changing compiler, LibTorch,
CUDA version or GPU.

### Vision norm/layout/residual fusion

Each block combines:

- FP32 LayerNorm, optional 24×24 window partition and projection dtype conversion;
- attention inverse-window layout, FP32 residual, LayerNorm and MLP input
  conversion;
- MLP output residual with the next block's LayerNorm, window partition and QKV
  input conversion (the prepared input lives only inside that forward);
- Q and K RoPE in one launch (`rotary_embedding_pair`).

The LayerNorm kernel keeps the reference Welford order (PyTorch attribution in
`native/third_party`). Unsupported layouts, CPU inputs and degenerate window
grids use the reference ATen expressions. With autocast disabled, projection
inputs stay FP32. CMake switches `SAM3_FUSE_VISION_NORM` and
`SAM3_FUSE_VISION_QK` (default ON) affect only compilation of the vision sources;
the exported fused operators remain tested either way.

### In-place GELU

The ordinary MLP applies `at::gelu_` to its freshly allocated FC1 output, which
has no other consumer. The exact GELU expression and the FP16 rounding point
between GEMM and GELU are unchanged; the BF16-reference epilogue is untouched.

### Image result interpolation

Later bilinear resize chunks are written into the final probability tensor with
`upsample_bilinear2d_out`. Because `out=` calls bypass autocast, inputs are
converted explicitly to the dtype chosen by the first ordinary interpolation;
one sigmoid is applied to the whole result as before. The chunk size still
bounds temporaries without limiting detections. Peak allocation is unchanged
because the full probability result dominates it. All 672 COCO-slice outputs of
the image audit stayed byte-identical.

### Semantic text reuse

An owning predictor keeps its completed text encoding across the internal reset
of `add_prompt` when the optional UTF-8 text is identical. Absent, empty and the
source's special `visual` string are distinct keys. Reuse also requires the same
Torch inference settings (threads, TF32/FP32 policy, reduction and deterministic
settings, attention backend enablement and priority); any change or a public
`reset()` re-encodes. Geometry, visual prompts, observations, image features and
tracking state still reset. `sam3_semantic_text_probe` verifies reuse by hiding
the text shards through a temporary hard-linked store view.

## Profile after fusion and rejected candidates

Nsight Systems on one warmed FP16 batch-1 forward at `1ee3440` (Blackwell):

| Model | Kernels before (`3a58138`) | Kernels now | GEMM share | Attention share |
|---|---:|---:|---:|---:|
| SAM3 | 711 | 416 | 67.94% | 11.95% |
| SAM3.1 | 705 | 430 | 65.81% | 11.60% |

Residual/normalization kernels account for about 6%, GELU 2.7%, paired RoPE
1.7%. Linear GEMMs in the MLP and QKV/output projections remain the main target.

| Candidate | Outcome |
|---|---|
| Image feature reuse | Depends on repeated images; reverted |
| Half GELU lookup table / vectorized GELU | Exact, but no consistent whole-vision or batch-2 benefit |
| RoPE output allocation reuse | Exact; negligible at real shapes, batch-2 regression |
| CUDA Graph replay | Small batch-1 gain, batch-2 regression, ~395 MiB (batch 1) / 798 MiB (batch 2) extra reserve |
| Neck in-place GELU | Latency and peak unchanged |
| BLAS backend preference | Latency and peak unchanged |
| Column-major Half projection weights | 8–15% slower, more allocation |
| Alternative mask-cache packing | Exact, slower CPU path |

A fused GEMM+GELU epilogue must keep the pre-activation Half rounding point;
substituting the upstream BF16 epilogue is not an exact FP16 optimization.

## Reproduce

`sam3_vision_fusion_benchmark STORE MODEL MODE FRAME.ppm BATCH OUTPUT.json SAMPLES`
measures the whole vision encoder (all heads and positions, device 0, three
warmups). Compare builds with the CMake switches above in alternating fresh
processes and record clocks, driver, LibTorch and build flags. Related probes:
`sam3_vision_positions_benchmark`, `sam3_compute_storage_probe`,
`sam3_semantic_text_probe`, `sam3_image_results_test cuda --benchmark` and
`sam3_image_benchmark ... --save-probabilities` with
`native/eval/compare_image_runs.py`. See [SDK.md](SDK.md) for building them
against an installed SDK.
