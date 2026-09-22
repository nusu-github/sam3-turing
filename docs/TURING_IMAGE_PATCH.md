# SAM3-Turing image patch

Runtime optimizations for single-image SAM 3 inference on CUDA GPUs with limited
VRAM. Start with `apply_turing_patch(processor, compile=False)` for a short startup.
Use `compile=True` for repeated inference, or `compile="max-autotune"` to explore
additional kernel choices. The patch leaves the checkpoint and upstream model
files unchanged, but modifies the loaded model and processor in place.

## Validated hardware and scope

Developed on an RTX 3090 and subsequently tested on an **RTX 2060 Max-Q with
6 GB VRAM**, on Windows using a uv virtual environment. The
[RTX 2060 report](../experiments/results/local_rtx2060/README.md) includes setup,
FP16 and INT8 results, compilation costs, and the observed FP16-reference OOM.
Eager FP16 reduced peak inference allocation from 4.992 to 1.997 GiB and median
latency from 3.752 to 1.585 seconds in that test.

Do not extrapolate these results to every 6 GB GPU. Benchmark FP16, INT8, and
compilation on the target hardware. INT4, packed-mask stress tests, and video
results below were measured on the RTX 3090, not the local RTX 2060.

The image patch supports batch size 1, inference only, text and geometric box
prompts, confidence-threshold changes, and empty outputs. Training, SAM 1-style
instance interaction, and SAM 3.1 video are outside its scope. Apply it once per
model/processor pair; rebuild the model to undo it. Moving a patched model to
another device is unsupported.

## Basic usage

Install dependencies and obtain access to the official SAM 3 checkpoint first.
See the [tested Windows uv setup](../experiments/results/local_rtx2060/README.md#reproduce)
or the upstream installation instructions in the repository README.

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.turing import apply_turing_patch

# One initialization thread avoided CPU initialization crashes in the NGC tests.
torch.set_num_threads(1)
model = build_sam3_image_model()  # Or provide checkpoint_path="path/to/sam3.pt".
torch.set_num_threads(4)
processor = apply_turing_patch(Sam3Processor(model), compile=False)

state = processor.set_image(Image.open("assets/images/truck.jpg").convert("RGB"))
result = processor.set_text_prompt(prompt="truck", state=state)
masks, boxes, scores = result["masks"], result["boxes"], result["scores"]
```

The LRU cache holds up to 16 text prompts by default. Set `text_cache_size=0` to
disable it. For changing GPU prompts, `compile_text=True` also compiles the text
Transformer, excluding string processing; this increases startup cost.

### CPU text encoding

For lower VRAM use with arbitrary prompts, keep `compile_text=False` (the default)
and offload before the first image inference:

```python
from sam3.turing import offload_text_encoder

offload_text_encoder(processor, trim_padding=True)
```

Text runs in CPU FP32 and only its features move to the GPU. Repeated prompts
reuse the LRU cache; new prompts incur CPU latency. The geometric prompt token
`visual` is encoded when needed. CPU text is compatible with image INT8 and
packed masks, but not GPU text INT8 (`text=True`) or `compile_text=True`.
FP32 text weights occupy approximately 1.32 GiB on the CPU.

`trim_padding=True` skips tokens after EOS in the standard causal text encoder,
then restores the original token count before passing features downstream.
Changed CPU operation shapes can introduce small rounding differences. Longer
prompts with less padding offer less opportunity to save work.

For optional CPU weight and latency savings, dynamically quantize the text MLPs:

```python
# Use this instead of the preceding offload call, not in addition to it.
offload_text_encoder(processor, int8_mlp=True, trim_padding=True)
```

Only the 48 text MLP Linear layers use per-tensor dynamic INT8. Attention,
embeddings, and the resizer stay CPU FP32. This differs from GPU text quantization
with `apply_int8_patch(..., text=True)`. Caching and later prompt freezing remain
supported. CPU thread count affects latency; set it after model initialization.
The patch does not change thread counts.

On the RTX 3090 / EPYC 7763 system, public CPU INT8 measured 84.79 ms with cached
text and 100.80 ms without caching. The preceding FP32 uncached control took
155.84 ms. CPU parameter plus quantized weight/bias storage fell from about
1.32 to 0.76 GiB; these are not process RSS measurements. Counts stayed 1/4/6/4/0,
with mean mask IoU 0.997790 and 922 changed pixels versus stock.
[CPU INT8 results](../experiments/results/accepted_cpu_dynamic_text_uncached.json) /
[FP32 comparison](../experiments/round48.json).

The EPYC tests used four threads and PyTorch's x86 quantization backend. A
separate untrimmed INT8 comparison took 105.17 ms at four threads and 87.90 ms at
eight. [Thread comparison](../experiments/round50.json).
Trimmed uncached CPU text measured 87.48 ms / 698 changed pixels with FP32 and
87.64 ms / 904 changed pixels with INT8; counts matched.
[FP32](../experiments/results/accepted_cpu_trimmed_text_fp32.json) /
[INT8](../experiments/results/accepted_cpu_trimmed_text_int8.json).

### Fixed prompt vocabulary

If all prompts are known, precompute their features and release the text encoder.
Omit `compile_text` and register prompts before inference:

```python
from sam3.turing import freeze_text_prompts

freeze_text_prompts(processor, ["truck", "wheel", "person", "visual"])
```

Unregistered prompts then raise an error. Include `visual` for geometric-only
box prompts. CPU offload, unlike freezing, retains arbitrary prompts.

## Optional INT8

Apply after the base patch and before inference. No retraining is required.
INT8 introduces additional output differences, so it is an explicit option.

```python
from sam3.turing_int8 import apply_int8_patch

apply_int8_patch(processor, attention_projections=True, fused_mlp=True)
```

- `vision=True` (default) quantizes ViT MLPs. `attention_projections=True` also
  quantizes QKV and output projections. Attention itself receives FP16 Q/K/V.
- Weights use per-output-channel scales; activations use per-token scales.
  `torch._int_mm` performs integer matrix multiplication with INT32 accumulation
  and the layer returns FP16. Original floating-point weights are released.
- `fused_mlp=True` combines dequantization, ordinary GELU, and requantization in
  an eight-warp Triton kernel, preserving FP16 intermediate rounding. The default
  `False` runs these operations separately.
- `text=True` also targets GPU text MLPs. `vision=False, text=True` selects only
  text MLPs. GPU text INT8 cannot be combined with CPU text offload.
- `asymmetric_gelu=True` requires `fused_mlp=True`. It uses a per-token zero point
  after GELU and corrects the INT32 product using weight row sums. Its four-warp
  kernel offers a different numerical tradeoff.

The RTX 2060 test used image INT8, attention projections, fused MLP, and CPU FP32
text with padding trimmed. It did not quantize CPU text. Against FP16 with the
same CPU-text option, latency fell from 1.589 to 1.365 s and allocation from
1.336 to 0.926 GiB. This measures INT8 and fusion together, not their individual
contributions.

`optimize_weight_scales=True` searches scales once during initialization to reduce
weight reconstruction squared error. It adds no inference operations and needs
no image calibration, but downstream agreement can improve or worsen. For
asymmetric GELU plus projections on the RTX 3090, changed pixels fell from 735
to 697, score error from 0.01172 to 0.00391, and box error from 1.44 to 0.57 px.
The adjacent latency comparison was 85.67 versus 86.91 ms. Improvements were not
consistent for symmetric or weight-only variants; the default is off.

```python
apply_int8_patch(
    processor,
    attention_projections=True,
    fused_mlp=True,
    asymmetric_gelu=True,
    optimize_weight_scales=True,
)
# Optional VRAM saving with arbitrary prompts; compile_text must be False.
offload_text_encoder(processor)
```

This CPU-offloaded RTX 3090 configuration measured 86.30 ms, 0.793 GiB allocated,
2.368 GiB NVML, and mean IoU 0.998065. Repeated prompts hit the text cache.

### INT8 storage with FP16 computation

`weight_only=True` reconstructs INT8 weights as FP16 for each layer and uses a
normal FP16 linear operation. It does not quantize activations and cannot be
combined with `fused_mlp` or `asymmetric_gelu`.

```python
apply_int8_patch(processor, attention_projections=True, weight_only=True)
```

On the RTX 3090 this took 117.29 ms, 1.456 GiB allocated, 2.862 GiB NVML, and mean
mask IoU 0.998763. MLP-only storage took 115.57 ms, 1.574 GiB allocated,
3.030 GiB NVML, mean IoU 0.999030, and 343 changed pixels. This saves weight memory
at roughly FP16 speed. `apply_int8_mlp_patch` remains a compatibility entry point.
Rebuild the model to undo quantization; do not quantize the same layers twice.

## RTX 3090 measurements

![Accepted image configurations](images/accepted_image_benchmarks.png)

These results used an existing RunPod container with NVIDIA PyTorch 2.10.0a0 and
CUDA 13.0, without a separate virtual environment. They are distinct from the
Windows RTX 2060 results linked above.

| Configuration | ms/image | PyTorch allocated GiB | Whole-device NVML GiB | Mean mask IoU vs stock |
|---|---:|---:|---:|---:|
| Stock BF16 | 210.81 | 4.994 | 6.017 | 1.0 |
| Patched FP16, eager (Round 10) | 169.77 | 1.997 | 3.144 | 0.99917 |
| Patched FP16, compiled | **114.31** | **1.862** | **3.095** | **0.99921** |
| FP16, uncached text | 116.32 | 1.862 | 3.095 | 0.99920 |
| Weight-only INT8, MLP + attention projections | 117.29 | 1.456 | 2.862 | 0.99876 |
| INT4 Gaussian group 32 + CPU text + image refinements | 116.36 | 0.590 | 2.403 | 0.99329 |
| INT4 asymmetric group 16 + CPU text + image refinements | 116.75 | 0.668 | 2.229 | 0.99448 |
| Vision MLP INT8 + compilation | **92.26** | **1.625** | **3.099** | **0.99740** |
| INT8 + projections + fused GELU | **85.46** | **1.455** | **2.872** | **0.99742** |
| Asymmetric INT8 MLP + fused GELU | **89.47** | **1.628** | **3.190** | **0.99848** |
| Asymmetric INT8 + projections + fused GELU | 85.01 | 1.456 | 2.851 | 0.99805 |
| Same + weight-scale tuning | 86.91 | 1.456 | 2.851 | 0.99807 |
| Same + tuned weights + CPU text, cached | **86.30** | **0.793** | **2.368** | **0.99806** |
| Same + CPU text MLP INT8, cached | 84.79 | 0.793 | 2.368 | 0.99779 |
| Same + CPU text MLP INT8, uncached | **100.80** | **0.793** | **2.345** | **0.99779** |
| Image refinements + CPU FP32, trimmed padding, uncached | **85.37** | **0.782** | **2.573** | **0.99800** |
| Same + CPU text MLP INT8 | **84.38** | **0.782** | **2.321** | **0.99774** |
| CPU text + FP16, cached | 113.39 | 1.202 | 2.370 | 0.99919 |
| CPU text + INT8 + projections + fused GELU, cached | **84.94** | **0.793** | **2.333** | **0.99742** |
| CPU text + INT8, uncached | 187.33 | 0.793 | 2.333 | 0.99742 |
| Fixed prompts + FP16 | 114.52 | 1.256 | 2.411 | 0.99921 |
| Fixed prompts + INT8 + projections + fused GELU | **84.18** | **0.850** | **2.401** | **0.99742** |
| Vision + text INT8, fused, uncached | 87.64 | 1.265 | 2.849 | 0.99745 |

Compiled FP16 reduced latency by about 46% and whole-device memory by about 49%
versus stock. Compiling grounding as a unit improved earlier timings of 153.53
and 121.30 ms. INT8 with projections and fused GELU reduced latency by about 59%.

### Method and provenance

Latency covers `set_image` + `set_text_prompt` on `truck.jpg` / `truck`, including
preprocessing and transfer but excluding file reads and model construction.
The table uses two warmups and the median of nine repetitions, with TF32 off.
NVML samples whole-device usage every 5 ms. Memory peaks cover inference after
patching; `build_allocated_bytes` separately records model construction.

Stock, FP16, and MLP INT8 are from Round 14; fusion and prompt variants from
Round 18; asymmetric INT8 from Round 32; weight-only INT8 from Round 34; public
CPU-text APIs from Round 42; tuned weights and CPU text from Round 44; and public
INT4 with trimmed CPU text and refinements from Round 67. Eager FP16 uses the
unchanged Round 10 measurement.

GPU uncached-text rows use `text_cache_size=0, compile_text=True`. CPU text uses
`compile_text=False` on an AMD EPYC 7763 with four threads. The untrimmed CPU
uncached prototype measured 146.64 ms, versus 187.33 ms for the public version;
the nine public repetitions ranged from 147 to 211 ms. Prompt workload and CPU
latency therefore matter. Projections without GELU fusion took 90.19 ms and
2.854 GiB NVML.

Output comparisons cover three images and five conditions: truck, paper bag,
child, wheel, and an empty elephant query. Accepted configurations preserved
counts **1 / 4 / 6 / 4 / 0**. Masks are compared after box matching; this measures
agreement with stock, not ground-truth accuracy.

| Configuration | Changed pixels / 18,038,400 | Maximum score difference | Maximum box difference, px |
|---|---:|---:|---:|
| Compiled FP16 | 285 | 0.00391 | 0.49 |
| MLP INT8 | 836 | 0.01367 | 0.66 |
| INT8 + projections + fused GELU | 916 | 0.00781 | 0.49 |
| Asymmetric MLP INT8 | 544 | 0.00586 | 0.47 |
| Asymmetric INT8 + projections | 735 | 0.01172 | 1.44 |
| Weight-only INT8 + projections | 389 | 0.00537 | 0.51 |
| CPU text + image INT8 | 915 | 0.00830 | 0.51 |
| Asymmetric INT8 + tuned weights | 697 | 0.00391 | 0.57 |
| Same + CPU text MLP INT8 | 922 | 0.02051 | 0.52 |

CPU text with FP16 images changed 289 pixels; tuned asymmetric INT8 with CPU
text changed 698. Cached and uncached CPU INT8 had the same five-case comparison
values. Fewer mask differences do not necessarily imply smaller box differences.

First-call compilation depends on cache state: about 20.5 s for FP16 and 22.3 s
for MLP INT8 in the reported run; a new grounding graph took about 48 s in
Round 13. New fused INT8 graphs took 55–57 s, while related cached fixed-prompt
graphs took about 21 s. Asymmetric INT8 took about 63 s for MLP only and 37 s
with projections; weight-only INT8 took 23 s, CPU text 21–23 s, and tuned weights
24 s. The RTX 2060's 138 s first compiled inference is a separate measurement.

### Efficient attention and SM75 checks

Forcing efficient CUDA attention on the RTX 3090 without FlashAttention gave
118.83 ms / 3.081 GiB NVML for FP16 and 90.93 ms / 2.892 GiB for fused INT8 with
projections. Counts matched; the latter had mean IoU 0.997066. With refinements
and trimmed uncached CPU text, image INT8 + CPU FP32 measured 90.60 ms,
0.783 GiB allocated, 2.549 GiB NVML, and IoU 0.998116. CPU text INT8 measured
90.01 ms, 0.782 GiB, 2.351 GiB, and IoU 0.997886. CPU attention was unchanged.
[Comparison](../experiments/round58.json).

Seven public Triton kernels passed offline SM75 compilation with Triton 3.5.0.
An experimental custom INT8 GEMM failed lowering and was not adopted; the public
patch uses `torch._int_mm`. Offline compilation alone is not runtime validation;
the later RTX 2060 report provides actual device measurements for its tested
configurations. [Compile results](../experiments/results/sm75_compile_check.json).

## Lower input resolutions

Set resolution when creating the processor; the patch also adjusts RoPE. Keep
the default 1008 for closer agreement. Round 30 compared the same INT8 +
projections + fused GELU configuration:

| Input resolution | ms | Whole-device NVML GiB | Mean mask IoU | Minimum mask IoU | Maximum box difference, px |
|---|---:|---:|---:|---:|---:|
| 1008 | 86.78 | 3.026 | 0.99742 | 0.99319 | 0.49 |
| 896 | 76.04 | 2.712 | 0.97389 | 0.89985 | 4.32 |
| 840 | 70.47 | 2.714 | 0.97280 | 0.87577 | 7.89 |
| 784 | 63.84 | 2.692 | 0.97082 | 0.88517 | 8.63 |
| 672 | 40.68 | 2.854 | 0.95396 | 0.78244 | 17.62 |
| 560 | 35.23 | 2.577 | 0.93205 | 0.61447 | 26.19 |

Counts remained 1/4/6/4/0, with allocation from 1.298 to 1.455 GiB. Agreement is
against stock 1008-resolution BF16 output; small objects can differ more than the
mean suggests. First inference took about 84–95 s with that cache state.
[Settings](../experiments/round30.json) /
[672 results](../experiments/results/compact_int8_resolution672.json).

```python
processor = apply_turing_patch(Sam3Processor(model, resolution=784), compile=True)
apply_int8_patch(processor, attention_projections=True, fused_mlp=True)
```

## Optional image refinements

`apply_image_refinements` combines FP16 ViT block outputs, FP16 decoder FFNs,
and precomputation of the mask head's final projection. Create the base patch
with `compile=True`, then apply refinements once after any image quantization
and before inference. These options change rounding and remain opt-in.

```python
from sam3.turing_refinements import apply_image_refinements

apply_image_refinements(processor)
```

Public image INT8 + CPU text measured 83.08 ms, 0.837 GiB allocated,
2.382 GiB NVML, mean IoU 0.997999, 673 changed pixels, score error 0.01074, and
box error 0.556 px. GPU text with the same image side took 83.08 ms / 2.813 GiB
NVML with 675 changed pixels. FP16 images with CPU text took 112.09 ms with
293 changed pixels. Counts stayed 1/4/6/4/0.
[CPU text](../experiments/results/accepted_refined_int8_cpu.json) /
[GPU text](../experiments/results/accepted_refined_int8_gpu.json) /
[FP16 images](../experiments/results/refined_fp16_cpu.json).

Preprocessing is unchanged and no additional custom CUDA kernel is used. A
five-change candidate also compiling the neck and normalizing outputs took
83.78 ms, so the three-change variant was selected. Public allocation was
0.837 GiB versus the prototype's 0.782 GiB; the public value is reported above.

For trimmed CPU text, offload before refinements. The following comparison used
asymmetric image INT8 and tuned weight scales:

```python
offload_text_encoder(processor, trim_padding=True)
apply_image_refinements(processor)
```

Uncached CPU FP32 measured 85.37 ms, 0.782 GiB allocated, 2.573 GiB NVML,
674 changed pixels, IoU 0.997997, score error 0.01074, and box error 0.571 px.
Adding `int8_mlp=True` to offload gave 84.38 ms, 0.782 GiB, 2.321 GiB,
822 pixels, IoU 0.997737, score error 0.02783, and box error 0.552 px. Counts
matched. The speed difference was small for these short prompts; CPU FP32
offered closer agreement.
[FP32](../experiments/results/accepted_refined_trimmed_fp32.json) /
[INT8](../experiments/results/accepted_refined_trimmed_int8.json).

### Skip projections of padded window rows

For sizes such as 784, `unpadded_projections=True` moves QKV projection before
window partitioning and output projection after unpadding. Attention retains
padding tokens and their biases; only projections of rows without real pixels
are skipped. Default: `False`. At sizes without padding, such as 1008, it is a
no-op. On EPYC, more CPU threads exposed its benefit at smaller image sizes:

```python
# Apply to a fresh image model; this thread count was tested on EPYC.
torch.set_num_threads(16)
processor = apply_turing_patch(
    Sam3Processor(model, resolution=784), compile=True, text_cache_size=0
)
apply_int8_patch(
    processor, attention_projections=True, fused_mlp=True,
    asymmetric_gelu=True, optimize_weight_scales=True,
)
offload_text_encoder(processor, trim_padding=True)
apply_image_refinements(processor, unpadded_projections=True)
```

Resolution-related mask differences remain. Public 784 results preserved counts
1/4/6/4/0 and matched the prototype comparison metrics:

| CPU text | ms | Allocated GiB | NVML GiB | IoU vs stock | Minimum IoU | Changed pixels |
|---|---:|---:|---:|---:|---:|---:|
| [FP32, 16 threads](../experiments/results/accepted_unpadded784_cpu_fp32_threads16.json) | 61.58 | 0.692 | 2.173 | 0.971324 | 0.889583 | 9352 |
| [INT8 MLP, 8 threads](../experiments/results/accepted_unpadded784_cpu_int8_threads8.json) | 58.51 | 0.725 | 2.169 | 0.971532 | 0.892061 | 9374 |

For the INT8 row, use eight threads and
`offload_text_encoder(processor, int8_mlp=True, trim_padding=True)`. At 1008 the
option preserved all five comparison results. The
[API check](../experiments/results/api_smoke_unpadded784.json) covers all 28 window
blocks at 784, CPU INT8, packed masks, state reuse, boxes, and empty output.

## Compact 4-bit weight storage

`apply_int4_patch` stores ViT MLP and attention projection weights in four bits,
then reconstructs FP16 weights before each Linear operation. Activations and
matrix multiplication remain FP16. It prioritizes memory and was slower than
dynamic INT8, with larger output differences. No dedicated INT4 computation
kernel or retraining is used.

```python
from sam3.turing_int4 import apply_int4_patch

# Apply once to a fresh model, before inference.
processor = apply_turing_patch(Sam3Processor(model), compile=True)
apply_int4_patch(processor, group_size=32)
offload_text_encoder(processor, trim_padding=True)
apply_image_refinements(processor)
```

The default uses 15 symmetric levels derived from Gaussian quantiles and FP16
scales; it is not NF4. Group sizes are 16 or 32. `asymmetric=True` uses 16 uniform
levels and an FP16 offset. Gaussian group 32 saved the most allocated memory in
this comparison; asymmetric group 16 gave smaller mask differences:

```python
# Alternative to the previous INT4 call, not an additional conversion.
apply_int4_patch(processor, group_size=16, asymmetric=True)
```

Reapplying to quantized layers raises an error. `attention_projections=False`
selects MLPs only. CPU text, image refinements, and packed masks are compatible.
INT4 on the RTX 2060 remains unmeasured.

These public API results use CPU FP32 text, four threads, trimmed padding, and
uncached prompts. Counts remain 1/4/6/4/0. Timing covers whole-image inference
returning dense masks:

| 4-bit method | ms | Allocated GiB | NVML GiB | IoU vs stock | Changed pixels |
|---|---:|---:|---:|---:|---:|
| [Gaussian 16](../experiments/results/accepted_int4_gaussian16_cpu.json) | 116.94 | 0.616 | 2.220 | 0.993886 | 1946 |
| [Gaussian 32](../experiments/results/accepted_int4_gaussian32_cpu.json) | 116.36 | 0.590 | 2.403 | 0.993292 | 1995 |
| [Gaussian 32, efficient attention](../experiments/results/accepted_int4_gaussian32_cpu_efficient.json) | 121.37 | 0.589 | 2.173 | 0.993303 | 1993 |
| [Asymmetric 16](../experiments/results/accepted_int4_asymmetric16_cpu.json) | 116.75 | 0.668 | 2.229 | 0.994485 | 1744 |

The matching FP16 control took 113.50 ms, 1.244 GiB allocated, and 2.392 GiB NVML.
Whole-device peaks include compilation and temporary weight reconstruction, so
they do not fall proportionally with allocation. Initial model construction
allocated about 3.330 GiB. The
[INT4 API check](../experiments/results/api_smoke_int4_cpu_trimmed.json) verifies
128 converted layers, CPU text MLP INT8, trimming, refinements, packing, state
reuse, boxes, empty results, and fixed prompts.

Earlier prototypes remain in [Round 38](../experiments/round38.json) and
[Round 62](../experiments/round62.json). They ran at about 114–118 ms, close to
FP16, with roughly 2,000–3,000 changed pixels and unchanged counts. Round 62's
Gaussian group 32 measured 116.81 ms / 1.267 GiB / IoU 0.993293 / 1995 pixels;
asymmetric group 16 measured 116.65 ms / 1.453 GiB / IoU 0.994461 / 1748 pixels.
The later public API is described above.

## Compact binary mask outputs

`packed_masks=True` uses Triton to return one bit per pixel without retaining
dense probability maps. It returns `masks_packed` and `mask_shape` instead of
`masks` and `masks_logits`.

```python
from sam3.turing_masks import unpack_masks

processor = apply_turing_patch(Sam3Processor(model), packed_masks=True)
state = processor.set_text_prompt("truck", processor.set_image(image))
h, w = state["mask_shape"][-2:]
first_mask = unpack_masks(state["masks_packed"][:1], (h, w))
```

The RTX 3090 tests below resize 200 real model-query logits to 4K. They are
**output-component stress tests, not whole-image inference timings**:

- Dense binary output: 24.10 ms / 6.243 GiB. Chunking eight masks and bit-packing:
  28.64 ms / 0.505 GiB, with about 198 MiB of output. Eight sampled masks
  (66,355,200 pixels) matched exactly. Chunk size 1 took 46.06 ms / 0.288 GiB.
  [Original comparison](../experiments/results/packed_masks.json).
- Fusing all four output stages: 8.18 ms / 0.257 GiB, versus 28.72 ms / 0.505 GiB
  for the adjacent chunk-8 control. One of 1,658,880,000 pixels differed near the
  threshold due to rounding. Odd sizes, downsampling, near-threshold values,
  and 1x1 inputs matched exactly. [Fused results](../experiments/results/fused_masks.json).
- Block-size tuning and direct FP16 cutoff comparison reduced latency to
  **6.18 ms**. The cutoff matched PyTorch sigmoid-then-threshold for all 65,536
  FP16 bit patterns, and all 200 masks matched the preceding fused version.
  [Kernel tuning](../experiments/results/mask_pack_tuning.json).

The tuned fused path is the default for FP16/FP32; FP32 still uses sigmoid.
`mask_chunk_size` controls the older path for other dtypes. Explicitly select it
with `resize_and_pack_masks(logits, size, fused=False)`; default chunk size: eight.

## What the base patch changes

- Replaces the BF16-only ViT MLP with Linear + GELU following FP16 autocast.
- Converts Linear, Conv, attention projection weights, and token embeddings to
  FP16, preserving decoder FFNs that explicitly require FP32.
- Disables autocast's weight cache and reduces FPN clones, unused feature levels,
  and position caches.
- Caches text features and applies post-interpolation sigmoid in place.
- Uses known feature sizes to avoid GPU scalar reads in box position bias.
- Optionally compiles the vision encoder and text-prompt grounding as a unit.
  The processor's four output tensors are cloned outside CUDA Graphs to prevent
  overwrites by later inference. Boxes and early query filtering use separately
  compiled stages. Direct model calls retain their original output dictionary.
- Optionally compiles the text Transformer or offloads it to CPU.

Fused floating-point MLPs, GELU approximations, channels-last, in-place
PixelDecoder operations, and early query filtering offered little additional
benefit to the selected base configuration. `early_filter=True` remains optional
and defaults to off. Head reassociation, fixed attention backends, MLP splitting,
arena reuse, and eager real-valued RoPE were not selected as defaults based on
the RTX 3090 comparisons. Candidate code and measurements remain available.

## Reusing the patch and experiments

Apply the [standalone patch](../patches/turing-image.patch) with `git apply` to a
compatible upstream checkout without these modules. Do not apply it again when
using this repository. The source baseline was
`2345a4ad109ac29c569da749c91d84f10dc08c40`; image experiments pin `facebook/sam3`
to revision `3c879f39826c281e95690f02c7821c4de09afae7`.

The original reproduction inputs were the supplied FP16, 20USD, and GPU-validation
archives. RunPod experiments reused the container's Python/CUDA installation,
adding missing helper packages. Windows validation instead used a uv environment.

See the [experiment instructions](../experiments/README.md),
[all-candidate table](../experiments/results/README.md),
[CSV](../experiments/results/summary.csv), and the
[historical experiment notebook (Japanese)](../experiments/NOTES.md).
The development sweep concluded on 2026-09-21: 376 candidates, 384 attempts
including repeats and failures, and 373 valid image measurements. The final
bias-correction comparison increased changed pixels from 697 to 769 for the
selected asymmetric INT8 + tuned-weight configuration, so it was not adopted.
Unexecuted settings are not counted as measured results.

## Additional measurement and API references

The following raw artifacts preserve the detailed comparisons behind this guide.

- [r14 fp16 control](../experiments/results/r14_fp16_control.json)
- [r14 int8 control](../experiments/results/r14_int8_control.json)
- [accepted fused attention](../experiments/results/accepted_fused_attention.json)
- [compact fixed all int8](../experiments/results/compact_fixed_all_int8.json)
- [accepted asymmetric mlp](../experiments/results/accepted_asymmetric_mlp.json)
- [accepted asymmetric attention](../experiments/results/accepted_asymmetric_attention.json)
- [accepted weight only attention](../experiments/results/accepted_weight_only_attention.json)
- [accepted cpu text fp16](../experiments/results/accepted_cpu_text_fp16.json)
- [accepted cpu text int8](../experiments/results/accepted_cpu_text_int8.json)
- [accepted cpu text int8 uncached](../experiments/results/accepted_cpu_text_int8_uncached.json)
- [accepted optimized asymmetric](../experiments/results/accepted_optimized_asymmetric.json)
- [accepted optimized asymmetric cpu](../experiments/results/accepted_optimized_asymmetric_cpu.json)
- [accepted cpu dynamic text](../experiments/results/accepted_cpu_dynamic_text.json)
- [api smoke cpu text](../experiments/results/api_smoke_cpu_text.json)
- [api smoke cpu dynamic text](../experiments/results/api_smoke_cpu_dynamic_text.json)
- [api smoke refined cpu trimmed](../experiments/results/api_smoke_refined_cpu_trimmed.json)
