# Native SAM3 building blocks

This is a Python-independent ATen C++/CUDA library with a connected image
grounding pipeline, **not yet the full SAM3/SAM3.1 runtime**.
It implements little-endian mask packing/unpacking, chunked bilinear resize +
sigmoid + mask packing, stable score-ordered generic NMS with no detection count cap, 8-connected
component labeling/counts, and Euclidean distance transform. CPU and CUDA implementations use the same public C++ API in
`include/sam3/ops.h`. Development-only dispatcher registration also permits
reference tests through `torch.ops.load_library`, without linking Python.

See [the onboarding report](../docs/native/ONBOARDING.md) and
[progress](../docs/native/PROGRESS.md) for constraints and remaining work.

Provide LibTorch (or compatible development PyTorch exposing TorchConfig.cmake):

The text frontend also needs ICU 72–74 (74.2 recommended) and zlib development
libraries. ICU 74.2 is the tested Unicode 15.x profile; newer Unicode lowercasing
and normalization data must be revalidated before changing that range. On this
Ubuntu environment the packages are `libicu-dev` and `zlib1g-dev`. On Windows,
provide matching MSVC ICU/zlib packages through `CMAKE_PREFIX_PATH`; place their
DLLs beside the executable or on PATH along with the LibTorch DLLs. ICU supports
both platforms ([official build instructions](https://unicode-org.github.io/icu/userguide/icu4c/build.html)).

```sh
cmake -S native -B build/native -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -DCMAKE_BUILD_TYPE=Release -DSAM3_WITH_CUDA=ON -DSAM3_TEST_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75;120"
cmake --build build/native --config Release --parallel 2
ctest --test-dir build/native -C Release --output-on-failure
```

Select architectures supported by your CUDA toolkit; 75 is the Turing target,
120 is used on the present Blackwell development GPU. Torch's own CMake package
may add additional architectures. GPU execution tests require real hardware for
the architecture being tested. No Turing runtime validation has been performed.

Use both `-DSAM3_WITH_CUDA=OFF -DSAM3_TEST_CUDA=OFF` for CPU-only builds. Windows
requires matching MSVC and LibTorch Release/Debug configurations. Tests fail on
unavailable requested devices rather than silently falling back. No Python
command is used by this CMake project or the C++ executables.

Development-only extensive reference comparisons (Python is allowed here):

```sh
python native/tests/parity.py build/native/libsam3_native.so --cuda
```

The C++ ABI currently follows LibTorch and is not a stable C ABI. An application
must distribute matching LibTorch/CUDA libraries, and this directory is not yet
a relocatable runtime package. Generic NMS materializes a boolean N-by-N matrix;
this is a correctness baseline with optimization still pending. Resize preserves
ATen's dtype and sigmoid rounding, and bounds temporaries by `chunk_size` without
limiting output count.


Weight sharing inventory (development only, original checkpoints untouched):

```sh
python native/tools/inventory_weights.py --checkpoint sam3=/path/to/sam3.pt --checkpoint sam3.1=/path/to/sam3.1_multiplex.pt --output /private/path/inventory.json
```

Tensor identity includes dtype, shape and SHA256 of the contiguous logical bytes.
This is an inventory, not a deployable weight format. Components use equal-valued
nonzero pixels, 8-connectivity, root-index labels, and 64-bit labels/counts. EDT
returns float32 distances and uses upstream's finite `1e9` distance for masks
without any zero pixel. These operations preserve input shape (EDT: `[B,H,W]`,
components: `[B,H,W]` or `[B,1,H,W]`). GPU calls honor the current stream.

The native weight reader and full token-level VE text encoder are now available.
See [the store format](../docs/native/WEIGHTS.md). After exporting private weights,
a standalone C++ command accepts arbitrary token IDs:

```sh
build/native/sam3_text /private/native-weights-v1 sam3 cuda 49406 4629 49407
```

`sam3::TextEncoder` retains its loaded language module across calls and accepts
batched token tensors with variable sequence length up to the upstream context
of 32. It executes all 24 layers and returns padding mask, resized language
memory and input embeddings. This CLI is a token-level development probe;
`sam3::Tokenizer` below supplies IDs from arbitrary UTF-8 strings.
The Python `text_encode` test operator reloads weights per call; production C++
callers should retain the `TextEncoder` object for the needed lifetime.

Visual modules and RGB tensor preprocessing are available in `vision_encoder.h`
and `preprocess.h`. `VisionEncoder` returns the 32-layer trunk and all requested
SAM3 dual / SAM3.1 tri-neck features. Inputs are normalized `[B,3,1008,1008]`;
`preprocess_rgb` converts arbitrary-size RGB tensors using the upstream resize
and normalization. Move raw pixels to the execution device before preprocessing
to match that device's torchvision rounding.

```sh
build/native/sam3_vision /private/native-weights-v1 sam3.1 cuda fp16
```

This command uses synthetic pixels and prints feature shapes. It is a native
module probe, not complete segmentation. `fp32` and `fp16` use the unfused MLP
path; `bf16_reference` reproduces the original BF16-forcing MLP for comparison
on capable hardware and is not intended for Turing deployment. Vision calls
restore the caller's autocast state. The original model weights are not copied
into separate image/video variants or separate precision variants.

`GeometryEncoder` in `geometry_encoder.h` implements the detector's configured
point/box prompt encoder, including image pooling, label/position embeddings,
right-padded sequence concatenation, CLS and all three transformer layers.
Its input tensors preserve arbitrary prompt counts, positive/negative labels,
per-image padding and empty prompt sequences. This is distinct from the tracker
and interactive mask-prompt path, which remain to be ported.

```sh
build/native/sam3_geometry /private/native-weights-v1 sam3.1 cuda fp16
python native/tests/roi_align_parity.py build/native/libsam3_native.so --report /tmp/roi.json
python native/tests/geometry_parity.py build/native/libsam3_native.so /private/native-weights-v1 --checkpoint sam3=/private/sam3.pt --report /tmp/geometry.json
```

The geometry executable is an empty/mixed-prompt module probe using synthetic
features; applications supply their own feature/prompt tensors through C++.
The native ROIAlign kernel supports CPU/CUDA float32/float64/float16 and follows
torchvision's autocast policy. Its sampling implementation is adapted under
the [torchvision BSD license](third_party/torchvision/LICENSE); neither the
Python torchvision package nor its compiled library is required at runtime.

The image detector's six-layer fusion encoder and six-layer query decoder are
available as `DetectorEncoder` / `DetectorDecoder` in `detector.h`. The decoder
retains all 200 learned queries and the presence token, relative box-position
bias, iterative box refinement, and outputs from all six layers. Fusion accepts
variable prompt embeddings/masks and per-image spatial padding. It returns the
spatial metadata needed by the decoder without modifying caller tensors.

```sh
build/native/sam3_detector_transformer /private/native-weights-v1 sam3.1 cuda fp16
```

This standalone probe chains geometry, fusion and decoding on synthetic image
features. The separate image probe below also connects scoring and mask heads.
An optional decoder trace records layer intermediates for diagnostics; leave
it null in normal inference to avoid retaining large attention-bias tensors.

`GroundingDetector` in `grounding.h` connects geometry, image/prompt fusion,
query decoding, dot-product scoring, box refinement, pixel decoding, instance
masks and semantic masks. It retains all 200 queries and accepts varying text,
point/box and visual prompt sequences, padding, image/text mappings and optional
previous-mask features. Vision and text modules remain separate so callers can
release or reuse them without distributing duplicate model weights.

`postprocess_image` in `image_results.h` applies the original confidence rule
and restores boxes and masks to each image's original dimensions. It returns
probability masks, boolean masks and original query indices with no count cap.
Resize chunking bounds temporary memory; the full requested result is retained.
For original SAM3 image behavior, use `joint_scores=false` in grounding and
`combine_presence=true` in postprocessing. The video/SAM3.1 detector's joint
score path uses the opposite pair, avoiding double presence multiplication.

```sh
# Arbitrary token IDs including start/end, padded by the probe to context 32.
# This example is "truck"; the executable does not hard-code that prompt.
build/native/sam3_image_probe /private/native-weights-v1 sam3 cuda fp16 image.ppm result .5 49406 4629 49407
```

The C++ probe reads 8-bit RGB P6 PPM, runs preprocessing→vision→text→grounding→
postprocessing, and writes `result.json` plus `result.masks.bin`. JSON records
the dimensions, mask row byte count, query IDs, scores and pixel-space boxes.
Masks use one row per detection, row-major pixels packed least-significant-bit
first. The example uses FP16 for the intended Turing path; `bf16_reference` is
only for reference comparisons on supporting hardware. Image codecs,
interactive/video/multiplex session orchestration,
a stable C ABI and a relocatable LibTorch distribution remain outstanding.
Windows/Turing execution is left to the user; no GitHub Actions are used.

Development comparisons include `image_end_to_end_parity.py` (both models and
all three precision modes, including batched visual/geometry/previous-mask
prompts) and `image_probe_parity.py` (a separate C++ process with Python absent
from PATH, checked against saved upstream results). The SAM3.1 tensor test uses
its tri-neck/weights and joint scoring with common detector math; it does not
validate the unported multiplex video scheduler.

`Tokenizer` in `tokenizer.h` accepts arbitrary UTF-8 strings, performs the
source ftfy 6.1.1 cleaning (including mojibake repair), double HTML unescaping,
whitespace/lowercase normalization and VE byte-level BPE. `encode` returns
unframed IDs; `tokenize` accepts a batch, defaults to context 32, and preserves
the original start/end, padding and truncation rules. The same vocabulary is
used by both models; no model weights or shape/precision variants are added.

```sh
# UTF-8 prompt text may contain whitespace/newlines; it is read as one prompt.
build/native/sam3_image_probe /private/native-weights-v1 sam3 cuda fp16 image.ppm result .5 --text-file sam3/assets/bpe_simple_vocab_16e6.txt.gz prompt.txt
# Standalone tokenizer, one UTF-8 prompt per input line:
build/native/sam3_tokenize sam3/assets/bpe_simple_vocab_16e6.txt.gz < prompts.txt
```

The runtime reads the existing gzip vocabulary using zlib; it neither imports
Python nor invokes a vocabulary conversion script. Frozen source character
tables are checked in at `src/tokenizer_tables.h`. Regenerate only during
development using `tools/generate_tokenizer_tables.py` with CPython Unicode
15.0.0, ftfy 6.1.1 and regex 2025.11.3. The positive and negative regex property
branches are frozen separately because case-insensitive Unicode properties are
not simple complements in the source engine. Licensing notices are under
`third_party/ftfy`, `third_party/python` and `third_party/icu`.

`tests/tokenizer_parity.py` compares cleaned text and complete 32-token output
against the original implementation, including Unicode decompositions/case
mappings, combining marks, entities, broken encodings and long input segments.
The native CTest also covers variable context, empty batches, error handling
and reuse. `image_probe_parity.py` now supplies actual text files to the child
process; Python is used only to prepare/check the saved image fixtures.

Interactive neural modules are available in `interactive_prompt.h` and
`interactive_decoder.h`. `InteractivePromptEncoder` produces point/box sparse
embeddings, mask/no-mask dense embeddings and the learned Fourier position grid.
Point counts and batch sizes are variable; padding labels and box-corner labels
follow the source. This module accepts masks at the model's 288×288 prompt size;
the higher-level predictor's input-mask resize is still to be connected.

`InteractiveMaskDecoder` runs both two-way transformer layers and all four mask
tokens. `project_pyramid` projects the two high-resolution feature maps once for
reuse. Decoding supports batched images or repeated prompts for one image,
three-candidate output, and the original stability-based single-mask fallback.
It preserves source object-pointer tokens even when stability chooses a different
mask, and SAM3 versus SAM3.1 IoU activation differences. Raw candidates are also
returned for diagnostics and subsequent tracking integration.

```sh
build/native/sam3_interactive /private/native-weights-v1 sam3.1 cuda fp16 3 9
```

This executable chains prompt encoding and mask decoding on synthetic full-size
features with variable batch/point counts. It verifies native execution without
Python; it does not yet implement the real-image interactive session. Coordinate
transforms, input-mask resizing, no-memory feature injection, object gating and
pointers, original-size mask postprocessing and image/video session control are
the next integration steps. The separate multiplex propagation decoder is still
outstanding. Existing weight shards serve these modules without duplication.
