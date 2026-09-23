# FP16 compute-parameter storage

CUDA contexts and owning predictors configured for FP16 now retain vision/text
linear and convolution parameters at their FP16 compute dtype. Previously each
forward kept the FP32 parameters and converted them for autocast operations.
Layer norms, positional values, text embeddings and complex rotary frequencies
retain their source precision. The inference operators, prompts, query counts,
image resolution, tracking policies and on-disk weights are unchanged.

The loader reads and checks one tensor, transfers it to the target device, then
converts selected compute parameters there. This uses the same device conversion
as the prior autocast path and avoids holding a complete FP32 GPU module before
conversion. No CPU rounding substitution, new quantization or precision-specific
checkpoint is introduced. The preceding complete CRC checks remain enabled.

## API behavior

The C ABI needs no new options: an FP16 CUDA context/owner uses this storage
automatically. FP32, BF16 reference and CPU paths retain their source parameters.
Owning predictors still release temporary text weights after encoding; this
change does not add a persistent text model to each session.

Original C++ constructors retain precision switching and their existing class
layouts. New four-argument overloads opt into fixed FP16 CUDA storage:

```cpp
sam3::VisionEncoder vision(store, "sam3", at::Device("cuda:0"), at::kHalf);
sam3::TextEncoder text(store, "sam3", at::Device("cuda:0"), at::kHalf);
// These instances require forward(..., "fp16"). Use the original constructor
// (or at::kFloat storage) when the application switches inference precision.
```

Other storage dtypes and Half storage on CPU are rejected. Half-storage instances
explicitly reject FP32/BF16 forward requests. All preceding public strong symbols
remain available (462 CPU / 469 CUDA); four constructor symbols are added to each.
The C ABI stays at version 1. Injected shared vision modules keep caller-selected
storage and must support the owning predictor's configured mode.

## Validation and measurement

Both models retain exact tensor bytes, dtypes and layouts for image batches 1
and 2, all vision heads, and 12 text cases per process: batch 1/4, length 1/7/32,
int32/int64 IDs, empty/UTF-8/multiple text prompts. The final installed native
`sam3_compute_storage_probe` also checks invalid storage and mode rejection.
These fixed examples are test fixtures; supported runtime input is unchanged.

Eight old/new process pairs through the original constructor retain FP16/BF16/
FP32 batch-1 and FP16 batch-2 vision outputs. CPU23/CUDA41 CTests pass. Nineteen
owner/low-level API cases and six packed-cache C lifecycle cases retain all
1,848 files. Recovered SDK C lifecycle and the new probe retain 153 and 50 files
respectively with Python absent from PATH and runtime libraries from the SDK.

Three alternating independent process pairs per model on RTX PRO 4500 Blackwell
measure the following. Vision uses batch 1 with all heads; text uses four 32-token
prompts. Each process warms both encoders twice, then measures five vision and
16 text calls. No competing builds or GPU jobs run during measurement.

| Model | Component | FP32 parameter storage | FP16 compute storage | Time reduction |
| --- | --- | ---: | ---: | ---: |
| SAM3 | Vision | 60.566 ms | 57.551 ms | 4.98% |
| SAM3.1 | Vision | 62.520 ms | 59.477 ms | 4.87% |
| SAM3 | Text | 7.352 ms | 5.897 ms | 19.79% |
| SAM3.1 | Text | 6.755 ms | 5.548 ms | 17.86% |

Both columns execute FP16 inference; only parameter residency differs. The
component measurements exclude loading, preprocessing, tokenization, detector,
tracking and postprocessing. They are not complete-model or Turing timings.

With both encoders alive, retained parameter allocation falls from 3.051 to
1.639 GiB for SAM3 and 3.070 to 1.651 GiB for SAM3.1 (savings 1.412/1.419 GiB).
Vision-only forward peak falls by the same amount; its additional activation
peak remains 277,750,272 bytes. Text additional peak falls from 14,424,064 to
6,557,696 bytes. The report records exact current/peak byte counts. An owner
that releases text weights retains only the vision portion of the saving.

`compute-storage-validation.json` records samples, equality checks, symbol and
SDK recovery evidence. The private recipe is
`compute-storage-sdk-overlay/overlays.json`, applied after the weight-CRC layer;
evidence is under `native-foundation/compute-storage-linux`. Existing dependency
and shared-weight layers are reused. Large diagnostic feature arrays are
regenerable from the retained native client, fixtures, hashes and metadata.

This uses portable ATen CUDA conversion and the existing standalone LibTorch
route documented in SDK.md. Windows/Turing execution remains user-owned; no
GitHub Actions are used. Original-source FP16 residuals remain as documented
in VIDEO_COLLECTIVE_REFERENCE.md; native equality is not a broader quality claim.
