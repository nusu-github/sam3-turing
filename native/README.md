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
video/multiplex session orchestration,
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
the video heads below resize other mask sizes before prompt encoding.

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
features with variable batch/point counts. Existing weight shards serve these
modules without duplication. Video-specific input-mask resizing, object gating
and pointers are connected by `VideoInteractiveHeads` below. The separate
multiplex propagation decoder remains outstanding.

`InteractiveImageSession` in `interactive_image.h` connects real image features,
pixel/normalized point and box coordinates, no-memory embeddings, original-size
mask postprocessing and repeated mask prompts. It retains projected high-resolution
features and the small interactive modules so the separately owned vision trunk
can be released after `set_image`. `set_images` and `predict_batch` accept image
batches with differing original sizes and differing prompt counts. `set_features`
also accepts a shared vision pyramid, including already projected high-resolution
features. This allows grounding and interaction to reuse the same trunk execution.

Preprocessing follows the canonical SAM3 image processor: resize RGB bytes to
1008, then convert/normalize. `set_image` and `set_images` preserve their respective
source tensor layouts, including a batch containing only one image. This matters
for exact convolution results. The standalone SAM2 float-before-resize transform
is a different preprocessing path; externally prepared features can be supplied
through `set_features`.

```sh
build/native/sam3_interactive_image /private/native-weights-v1 sam3 cuda fp16 image.ppm result 500 600 1 100 100 0
```

This development probe accepts any number of point triples, writes all three
initial candidates, and reuses the cached image features to refine the best-IoU
candidate with its returned low-resolution mask. Outputs are `result.initial.*`
and `result.refined.*`: JSON dimensions/IoU, little-endian packed boolean masks,
and float32 low-resolution logits. The C++ session also accepts boxes, previous
288×288 mask logits, normalized coordinates, and configurable mask thresholds,
hole/sprinkle areas and single/multimask output. Image predictions preserve the
source behavior of not applying the video object-presence gate. SAM3.1 here uses
its interactive modules with the common image host; its multiplex scheduler is
still to be ported.

Development parity scripts are `interactive_image_parity.py` (host transforms
and postprocessing), `interactive_image_probe_parity.py` (real-image standalone
execution, initial and repeated prompts), and `interactive_image_batch_parity.py`
(real batched images). Their Python reference's CPU component labeling requires
`numpy==1.26.4 scikit-image==0.25.2 tifffile==2025.6.11`; these are development
dependencies only. Reports under `docs/native` record exact tensor comparisons.
An intermittent CPU crash in this development PyTorch build also reproduces with
the original Python prompt encoder alone; see
[`CPU_RUNTIME_ISSUE.md`](../docs/native/CPU_RUNTIME_ISSUE.md). A successful parity
run does not establish CPU runtime stability. No GitHub Actions are used.

`VideoInteractiveHeads` in `video_heads.h` implements the original SAM3 video
heads and SAM3.1's per-object interactive video path. It handles missing-point
padding, arbitrary-size mask prompts with antialiased resizing, object-presence
mask gating, best-IoU selection and object-pointer projection. SAM3 uses its
learned absent-object pointer; SAM3.1 uses its learned linear transformation.
`use_mask_as_output` preserves supplied masks and computes pointers using the
original downsample/decoder path, including the second absence transformation.
The caller supplies projected high-resolution features and memory-conditioned
72×72 features (one per object for SAM3, one repeated image for SAM3.1).

```sh
build/native/sam3_video_heads /private/native-weights-v1 sam3.1 cuda fp16 3 9
```

This standalone synthetic-feature probe exercises single/multimask output,
variable batch/point counts and direct-mask output. `video_heads_parity.py`
compares all seven outputs against the original tracker methods: 57 CUDA cases
and 19 completed CPU FP32 cases match exactly. Direct-mask output dimensions
retain the upstream formulas: SAM3 uses `input_size // 14 * 4`, while SAM3.1
uses `input_size // 4`. Supplied mask sizes and model-image sizes are separate.
Frame selection, tracking sessions and multiplex propagation
remain to be implemented; these head tests are not full video tracking tests.

`MaskMemoryEncoder` in `memory_encoder.h` implements the memory downsampler,
two ConvNeXt fusion blocks, output projection and cached positional encoding.
SAM3 produces 64-channel spatial memory per object. SAM3.1 produces 256-channel
memory per multiplex group, with 16 mask and 16 conditioning input channels.
Groups can grow with object count; this is not a 16-object limit. A single image
feature may be shared across all objects/groups and may be staged on CPU before
the module transfers it to its execution device. No image-feature copies or
model-weight variants are required for this sharing.

`forward` exposes the neural module with optional sigmoid skipping.
`encode_frame` adds the shipped trackers' mask transformations, optional
non-overlap constraint, SAM3 point-mask binarization and absent-object spatial
embeddings. For SAM3.1, supply the original-layout selection matrix
`[groups*16, objects]` and optional conditioning object indices. Its selection
uses matmul to preserve the source's autocast rounding; the object management
controller that constructs/updates these matrices remains to be ported.
SAM3.1 ignores point-mask binarization as in its source configuration. The host
also preserves its object-score padding/truncation and unused-slot embeddings.

```sh
build/native/sam3_memory /private/native-weights-v1 sam3.1 cuda fp16 37
```

This synthetic-feature probe encodes 37 objects in three groups while sharing
one image feature. It exercises predicted and point-mask paths and reuses the
loaded module and positional grid. `memory_encoder_parity.py` compares neural
and frame-host outputs against original modules/methods, including shared
images, CPU staging, non-square masks, layouts, overlap, empty/partial conditions,
scores of differing lengths and object counts exceeding one group's capacity.
CUDA comparisons use the original builder's precomputed positional cache and
check its output layout as well as values. CPU reference positions are generated
on CPU because the original constructor hard-codes CUDA for precomputation.
These encoder tests do not cover temporal attention or complete video tracking.

`MemoryAttention` in `memory_attention.h` implements all four temporal attention
layers and final normalization. SAM3 uses one attention head, 64-channel memory
and ReLU; SAM3.1 uses eight heads, separate image/object streams, 256-channel
memory and GELU. Axial complex rotary encoding repeats over spatial memories
and excludes trailing object-pointer tokens. It preserves shared position/image
batches and SAM3.1's image-stream padding for pointer tokens. Inputs and output
are sequence-first tensors; optional layer traces are for development comparisons.

```sh
build/native/sam3_memory_attention /private/native-weights-v1 sam3.1 cuda fp16 1 2 4
# Force standard SDPA math for a full-grid fallback probe:
build/native/sam3_memory_attention /private/native-weights-v1 sam3.1 cuda fp16 1 1 4 math
```

The last three numbers are batch size, spatial memory frame count and pointer
token count. The probe always uses the full 72×72 query grid; memory frames and
pointer tokens are not truncated. The native module lets LibTorch dispatch its
available precompiled SDPA backend and preserves the caller's backend settings.
It does not force SAM3.1's source Flash-only context or import FA3/Triton.

`memory_attention_parity.py` compares every layer and the final output against
the original encoders. CUDA FP16/BF16-reference comparisons retain the source
backend behavior. For CUDA FP32 (which fails in the source Flash-only context
on this development build) and explicit math comparisons, the SAM3.1 reference
removes only that backend context. Math tests also prevent SAM3's forward method
from re-enabling fused backends and verify that only math stays enabled.
CPU rotary caches are recomputed on CPU as on a CPU-only host. Reports explicitly
identify adapted reference cases. These are backend compatibility and tensor
parity checks on available hardware; Turing/Windows execution remains untested
here, and complete video tracking still needs session control.

`Sam3MemoryConditioner` in `temporal_memory.h` connects SAM3 frame selection,
temporal positions, pointer assembly and memory attention. `TemporalState`
retains separate ordered conditioning/tracked frame collections; insertion order
is significant for equal-distance ties. Options preserve the original
conditioning limit/keep-first policy, temporal stride, forward/reverse direction,
score-based memory selection and pointer history. Defaults match the source
configuration, with no additional history or object cap. `assemble` exposes the
selected plan and tensors; `forward` also handles initial/no-memory branches.

Spatial features and positions may reside on CPU; pointers stay on the execution
device as in the source. Effective-IoU scores retain their tensor dtype because
converting them to double before threshold comparison can change half-precision
decisions. `memory_confidence` preserves source operations and broadcasting,
including vector-IoU input behavior. The shared `select_conditioning_frames`
helper preserves ordered ties and keep-first/limit-two slicing behavior.

```sh
build/native/sam3_temporal /private/native-weights-v1 cuda fp16 3
```

This development probe chains synthetic full-grid frames through memory
conditioning → video heads → memory encoding, offloads spatial memory to CPU,
and reuses it on later frames. The frame count is configurable. It runs without
Python, but it is not a real-video predictor or an accuracy benchmark.
`temporal_memory_parity.py` compares selection, assembled memory and positions,
pointer counts and conditioned features against the original SAM3 host. CPU
comparisons redirect only hard-coded `.cuda()` transfers and recreate rotary
caches on CPU. SAM3.1 uses the separate multiplex temporal host described below.

`MultiplexState` and `MultiplexController` in `multiplex.h` implement SAM3.1's
inference object allocation and tensor mux/demux. Physical width defaults to 16;
the number of buckets grows with the object count. Optional external IDs remain
stable while dense internal indices are renumbered after removals. Removed slots
stay occupied until their whole bucket is discarded. `remove_objects` returns
the old bucket indices to retain, for updating associated memory tensors.
Allocation supports reduced per-bucket capacity, CPU-RNG shuffling and explicit
preference for new buckets. Mux/demux retain the source matrix multiplications
and caller autocast behavior, including input rounding.

Invalid additions/removals leave the state unchanged. Removing all objects
invalidates it, clears active counts and releases matrices; create a fresh state
before adding new objects. These error/invalid-state guarantees deliberately do
not reproduce the source's partial failed mutations or stale inactive metadata.
Matrix construction uses CPU 0/1 buffers with one transfer per matrix; both
layouts match separate contiguous source allocations, including singleton shapes.

```sh
build/native/sam3_native_multiplex_test cuda fp16
build/native-cpu/sam3_native_multiplex_test cpu fp32
```

The standalone C++ test covers transactional errors, bucket retention, external
IDs, invalidation, noncontiguous/empty-feature tensors and up to 257 objects.
`multiplex_parity.py` compares allocation, RNG output and repeated additions /
removals against the original inference controller. This is the state/controller
component; full video sessions remain to be connected.

`MultiplexMaskDecoder` and `MultiplexPropagationHeads` in `multiplex_decoder.h`
implement the shipped SAM3.1 propagation path. The trained decoder has 16 slots
per bucket, three distinct mask tokens per slot and separate IoU/object tokens.
It produces all three candidates; the propagation host demuxes them into valid
object order, gates absent-object masks, resizes to 1008px, selects the best IoU
candidate and projects its pointer. The source's optional IoU attenuation by
mask stability is available. Suppression embeddings distinguish live slots from
padding and removed objects. Dense positional encoding, high-resolution feature
projection and the linear absent-pointer transform use the existing weight store.
The number of buckets is determined by `MultiplexState`, with no added object cap.

The low-level decoder supports diagnostic rectangular grids and shared or
per-bucket high-resolution maps. The propagation host uses the original full
72x72 features/288x288 candidate masks and shares high-resolution image features
across buckets. The shipped weights are configured for three candidates only;
they have no extra single-mask token. Interactive image/video heads remain the
separate original interactive decoder.

```sh
build/native/sam3_multiplex_propagation /private/native-weights-v1 cuda fp16 17
# Exercise standard precompiled SDPA math attention:
build/native/sam3_multiplex_propagation /private/native-weights-v1 cuda fp16 17 math
```

This C++ probe uses synthetic conditioned features, decodes all objects, encodes
the selected masks as new multiplex memory, removes/adds an object and repeats.
It does not yet select temporal memories or process a real video.
`multiplex_decoder_parity.py` compares the original decoder and tracker host,
including per-slot outputs, suppression, demux, positions, resized masks and
pointers. `--math` prevents the original attention method from re-enabling
Flash/memory-efficient SDPA to compare the ordinary math fallback. Model tensor
operations remain the same; the native library preserves caller backend settings.

`MultiplexMemoryConditioner` in `multiplex_temporal.h` connects SAM3.1 temporal
selection to its separate image and object memory streams. It shares the ordered
conditioning/stride/score selector with SAM3, while preserving SAM3.1's default
future-conditioning pointers, unsigned pointer times and v2 spatial time
embeddings. Options also expose past-only/signed pointers, dummy pointer times,
v1 spatial time encoding and disabled pointers/memory. Spatial and image tensors
may be offloaded to CPU; pointers stay on the execution device. The current image
is shared across buckets. Stored history must already match the current bucket
allocation; session-level history updates after rebucketing remain separate.

Cleared spatial memory or pointer-only history falls back to the current image
features, as in the original host. Legacy 5D per-slot memory/positions are
demuxed, made contiguous and cached back into state. Initial or explicit
previous-memory bypass frames must use interactive/direct-mask heads; the source
does not support them in this temporal path unless memory is disabled.

```sh
build/native/sam3_multiplex_temporal /private/native-weights-v1 cuda fp16 17 3
build/native/sam3_multiplex_temporal /private/native-weights-v1 cuda fp16 2 3 math
```

The arguments after precision are object and frame counts. This development
probe initializes supplied masks with the interactive heads, stores encoded
memory and image features on CPU, then conditions/propagates/encodes subsequent
synthetic full-grid frames. All objects and three propagation candidates remain
available. It is not yet a real-video session or an accuracy benchmark.
`multiplex_temporal_parity.py` compares original assembled tensors and final
conditioned features, including reverse/strided/score-based selection, cleared
memory and 5D state normalization. CUDA FP32/math reference removes the source
Flash-only context; CPU redirects hard-coded CUDA transfers and recreates rotary
caches on CPU. These backend adaptations are recorded in the reports.

`Sam3TrackingFrame` in `tracking_frame.h` exposes the SAM3 inference frame host
used to build a video session. `TrackingFrameRequest` carries frame identity,
direction, initial/bypass policy, arbitrary point/label tensors, a supplied mask
or previous mask logits. `TrackingFeatures` contains cached low-resolution image
features/positions and projected high-resolution maps. Shared image features can
be expanded to the object batch without recomputing the visual backbone.

The host selects the direct-mask or memory-conditioned interactive path, follows
the source multimask policy, optionally encodes new memory, and returns masks,
object pointers/logits and optional confidence scores. `encode_memory=false`
supports repeated edits before consolidation. `encode_memory` is also exposed
separately for encoding masks after all objects have been consolidated. Optional
CPU offload keeps pointers/object logits on the execution device. History
trimming retains low-resolution masks, pointers and object logits and preserves
the source's score/offload-dependent pruning rules, including reverse tracking.
The caller inserts returned frames into conditioning/tracked history.

```sh
build/native/sam3_tracking_frame /private/native-weights-v1 cuda fp16 2 3
```

This probe previews and refines the first frame using previous logits, then
propagates subsequent synthetic frames while storing outputs/memory on CPU.
`tracking_frame_parity.py` compares the original `Sam3TrackerBase.track_step`,
including direct masks, 17-point prompts, deferred/disabled memory encoding,
overlap constraints, scores, offload and history trimming. Session-level prompt
accumulation, per-object consolidation and real-video evaluation remain separate.
