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

### SAM3 interactive tracking session

`Sam3TrackingSession` in `tracking_session.h` implements the low-level interactive
tracker's state machine. It shares a `Sam3TrackingFrame` instance (and its weights)
between sessions and accepts a frame feature provider. The provider supplies one
image's projected maps; the session caches the most recent frame and expands its
features across objects. A visual backbone/video decoder can be attached through
this provider without creating image/video copies of the model weights.

The API supports arbitrary accumulated points, boxes, supplied masks, preview
edits using previous logits, memory consolidation, forward/reverse propagation,
clearing annotations, object removal/remapping and reset. Normalized coordinates
are multiplied by 1008; `normalized=false` expects model-image coordinates.
The default point cap is disabled. Optional positive `max_points` reproduces the
source's first/last-click policy, but is not an optimization used by this port.
There is no cap on object count. This is the original **low-level** predictor:
new IDs must be introduced before propagation starts. Dynamic object discovery
and text/video association belong to the remaining higher-level tracker.

Pending edits are kept per object. Preflight consolidates their low masks and
pointers, supplies empty-mask pointers for missing objects, applies source
overlap constraints and encodes memory only after consolidation. Supplied brush
masks stay at original video resolution for preview; overlapping later brush
strokes suppress earlier masks. Stored memory uses BF16, including FP16/FP32
execution, and can be offloaded to CPU. Positions are cached once; pointers and
object scores remain on the execution device. BF16 here is a storage format;
FP16 execution casts projection inputs through autocast. FP32 explicitly expands
the compressed values before memory attention; the source's permanent CUDA BF16
context otherwise hides a dtype error in its FP32 path.
CPU offload waits for device-to-host copies before returning: CPU consolidation
can immediately resize/copy the stored masks. Keeping those copies asynchronous
caused intermittent preview corruption despite correct final stored tensors.

Frame dictionaries retain insertion order, and temporary integer-set traversal
matches 64-bit CPython 3.12, including its different preallocation for dictionary
versus key-view updates. This preserves ordered conditioning attention for
sparse/nonmonotonic annotations without linking Python. The compatibility helper
is covered against the original Python containers, including collisions, large
growth and 64-bit keys. Other Python implementations/versions are not a claimed
container-order reference.

`propagate` emits completed frames through a callback. Return false, or call the
thread-safe `cancel()`, to stop after a consistent completed frame. Resuming is
another `propagate` call with the desired start index. Other session methods must
not be called concurrently. Propagation endpoints are inclusive, matching the
source: `max_steps=0` emits the start frame. Clear/reset retains cached visual
features and constant positions; removing the last object resets the session.
Optional output overlap suppression and two-pass connected-component cleanup
run after resizing to the original frame dimensions.

```sh
build/native/sam3_tracking_session /private/native-weights-v1 cuda fp16
build/native-cpu/sam3_tracking_session /private/native-weights-v1 cpu fp32
```

This standalone probe exercises 17-point input, two objects, brush input,
consolidation, cancellation/resume, reverse correction, removal and reset using
synthetic full-grid features. It does not decode a video or measure accuracy.
`tracking_session_parity.py` compares original predictor operations and snapshots
of global/per-object/pending state, not just displayed masks. FP32 reference
comparison explicitly restores compressed memory to float before attention;
CPU reference redirects CUDA transfers and rotary caches. Real-video evaluation,
SAM3.1 session rebucketing, text-driven video tracking and the C ABI remain.

### Real-frame tracking input

`Sam3TrackingVision` shares an existing `VisionEncoder` and `Sam3TrackingFrame`.
`encode_rgb` accepts decoded RGB bytes and follows the original synchronous
JPEG-sequence preprocessing: byte bicubic resize, division by 255, then
normalization by mean/std 0.5. It selects the SAM3 tracking neck, discards the
source's final `scalp=1` level, projects the two high-resolution maps and returns
`TrackingFeatures` for a session provider. The visual trunk runs once per cache
miss and is shared across all objects and repeated edits of that frame.
`encode_preprocessed` accepts normalized F32 `[1,3,1008,1008]` from a caller's
decoder-specific preprocessing, preserving its layout.

Image-mode `preprocess_rgb` is **not** interchangeable with this video JPEG path.
The original uses Pillow's default RGB bicubic, whose fixed-point weights,
byte-rounded intermediate rows and clipping differ from ordinary tensor bicubic.
`resize_tracking_rgb` implements those rules in portable C++ with CPU row
parallelism; it does not depend on Pillow or Python. Coefficient arithmetic
disables FMA contraction for reproducibility; the Pillow permission notice is
included under `third_party/pillow`. The output is contiguous NCHW, matching
the source frame buffer. This currently matches synchronous JPEG loading;
compressed video decode/resizing and asynchronous source-loader precision are
separate integration work.

```sh
build/native/sam3_tracking_video /private/native-weights-v1 cuda fp16 frames.txt commands.txt results
```

This development executable accepts decoded frames as binary P6 PPM files. The
UTF-8 manifest has one path per line, relative to its own directory or absolute;
all frames must have the same dimensions. The command file supplies arbitrary
prompts and operation order. Empty lines and lines starting with `#` are ignored.
Each active line has one of these forms (booleans are `0` or `1`):

```text
points FRAME ID CLEAR_OLD NORMALIZED USE_PREVIOUS_MEMORY [X Y LABEL]...
box FRAME ID NORMALIZED X0 Y0 X1 Y1
mask FRAME ID PPM_PATH
preflight ENCODE_MEMORY
propagate START MAX_STEPS REVERSE ENCODE_MEMORY PREFLIGHT STOP_AFTER USE_CANCEL
clear FRAME ID
remove ID
reset
```

`START=-1` and `MAX_STEPS=-1` use session defaults; `STOP_AFTER=0` runs the whole
requested range. Mask prompts use the PPM red channel divided by 255; the C++
session API also accepts full-precision tensor masks. Paths with spaces in mask
commands can be quoted using standard C++ quoted-string escaping. Output names
are `OPERATION-OUTPUT_INDEX`: JSON metadata, packed positive masks, F32 video
logits, and F32 low logits for propagated frames. Binary F32 files use the host's
native byte order (little endian on the tested/target x86-64 systems). The probe
logs visual-backbone calls to make frame-cache reuse observable.

`tracking_video_parity.py` compares actual video frames through preprocessing,
the full 1008px visual backbone and the session, with points/boxes, forward
tracking, correction, reverse propagation, removal and reset. Its native child
has `PATH=/nonexistent`. PPM fixtures carry identical decoded JPEG pixels; this
does not validate a compressed-video codec. `tracking_preprocess_parity.py`
separately compares byte resizing/normalization against Pillow and source loader
rounding. `compare_tracking_modes.py` measures FP16/FP32 agreement with BF16
artifacts; same-mode port parity is separate from precision-mode quality changes.

### SAM3.1 multiplex frame host

`sam3/multiplex_frame.h` exposes `Sam31TrackingFrame`, joining the interactive
head, propagation decoder, temporal selector and memory encoder. Supply the
shared image's interactive and propagation features (72px image/position and
projected 288px/144px high-resolution maps), a `MultiplexState`, ordered history,
and a frame request. Both necks share the model's visual trunk; this API adds no
weight copies or image/video model variants.

The host supports direct mask initialization, point initialization, refinement
with previous logits, pure propagation, and propagation with corrections to
selected objects. All points are passed to the decoder. The source's multimask
policy chooses the output mode without truncating points. During mixed
propagation/correction, interactive results replace the selected object rows,
including the source's broadcast into three propagation candidates. Pointers
are multiplexed into 16-slot buckets; memory includes conditioning flags.
Requests can defer memory encoding, reverse temporal selection, offload output
to CPU, retain image features and trim old history according to source policy.
Optional IoU stability attenuation is supported by both heads.

History passed to this layer must already have compatible bucket assignments.
The caller inserts the returned frame into conditioning/tracked history. An
interaction-only request with a strict subset needs a matching extracted local
state; demo-session singleton extraction/reintegration is a subsequent layer.
The original ground-truth-driven training correction loop is not an inference
operation and is not implemented here. Offloading follows the source frame
policy, which drops candidate masks and IoU/confidence after producing memory.

```sh
build/native/sam3_multiplex_frame /private/native-weights-v1 cuda fp16 17 3
build/native-cpu/sam3_multiplex_frame /private/native-weights-v1 cpu fp32 2 3
```

This standalone development probe uses full-grid synthetic features, a
17-point preview/refinement, partial correction, propagation and CPU-offloaded
memory. Object/frame counts are probe arguments; the runtime accepts arbitrary
compatible bucket counts. The optional final `math` argument selects ordinary
SDPA math for fallback testing. `multiplex_frame_parity.py` compares the original
`VideoTrackingMultiplex.track_step` across direct masks, points, partial
corrections, temporal direction, memory deferral, scoring, offload and trimming.
These synthetic-feature checks are separate from real-video quality evaluation
and user-owned Turing/Windows hardware verification.

`Sam31TrackingFrame::update_masks` implements the original dynamic model's
current-frame mask insertion and reconditioning methods. `MultiplexMaskUpdate`
selects append versus replacement, immediate versus deferred memory encoding,
and bucket-growth policy. Append allocates consecutive internal indices and
preserves optional caller object IDs. Reconditioning overwrites the requested
indices in caller order. Returned indices identify the affected objects. Mask
resolution adjustments, absent-object pointers, conditioning flags, mux/demux
precision and memory re-encoding follow the source. Existing image features
and optional input-mask storage are retained. Failed updates leave caller state
unchanged; successful updates replace the frame/state arguments.

This operation updates the current frame only. Use the history remapping helper
below after layout changes; full session singleton extraction/reintegration
remains separate work.
Use a frame with compatible devices and high-resolution masks when re-encoding
memory; apply storage offload after updates, as the source session does. The
source update methods leave existing auxiliary candidate tensors and effective
confidence unchanged, even when appending changes the object count; this API
does the same. The surrounding session is responsible for refreshing any
derived fields it subsequently uses. Reconditioning uses masks rather than
point provenance when re-encoding, matching the original method.

```sh
build/native/sam3_multiplex_update /private/native-weights-v1 cuda fp16 17
build/native-cpu/sam3_multiplex_update /private/native-weights-v1 cpu fp32 2
```

This standalone probe initializes three objects, appends the requested number
into new buckets, reconditions selected objects and propagates from that updated
conditioning frame. It also checks that capacity failure leaves state intact
and that an unrelated object's mask survives reconditioning. It does not test
remapping an older history after growth. `multiplex_update_parity.py` compares
the source dynamic methods across capacity/removed-slot policies, new buckets,
resolution changes, optional IDs/input masks, overlap, scores and deferred
encoding. Synthetic full-grid feature parity is separate from real-video quality.

### SAM3.1 history layout changes

`sam3/multiplex_history.h` provides `remap_multiplex_history` for histories whose
frames share a known source layout. Pass source and destination states with
unique global object IDs. Object rows and conditioning indices follow those
IDs; new historical objects receive absent masks/logits, zero pointers/IoUs and
no conditioning flag. Shared image features retain their storage. Obsolete
auxiliary tensors with incompatible row counts are cleared. Effective confidence
is recomputed from the remapped logits and retained IoUs when available.

Pointers are copied by slot without extra floating-point arithmetic. Unchanged
physical buckets retain their entire historical pointers and spatial memory,
including removed slots, matching the original removal policy. A permutation of
internal object indices alone therefore does not require neural re-encoding.
Stored memory keeps its device and dtype, including BF16 CPU storage.

Dense `[buckets,256,72,72]` memory jointly encodes 16 objects; it cannot be demuxed
as though its channel axis were an object axis. Changed bucket membership or
slot placement requires a `MultiplexHistoryRebuilder` callback. The callback
must not mutate its input tensors. It receives the remapped frame and can call
`Sam31TrackingFrame::encode_history`, which uses retained full-resolution masks,
object logits, conditioning indices and shared image features. The helper uses
rebuilt values for changed buckets and preserves unaffected buckets exactly.
It rejects a missing rebuilder rather than silently slicing or dropping joint
memory. Trimmed frames without spatial memory need no rebuild. An error in any
frame leaves the whole input history unchanged.

This requires retaining full-resolution masks and image features for histories
that may later need re-encoding. The source's usual aggressive trimming/offload
policy can remove those inputs; a session must retain them, reload them, or
explicitly recompute them before requesting a layout change. The helper does
not implement that storage policy or a full interactive session. It handles
valid nonempty destination states; removing every object is a session reset.

```sh
build/native/sam3_multiplex_history /private/native-weights-v1 cuda fp16 17
build/native-cpu/sam3_multiplex_history /private/native-weights-v1 cpu fp32 2
```

The probe preserves two prior frames, adds objects in new buckets, re-encodes
history, propagates with it and reconditions the new objects. It checks rollback
after failure on the second frame, unchanged bucket contents, and BF16 CPU
storage. `multiplex_history_parity.py` compares removal tensors against the
original demo and separately checks ID/pointer/memory conservation on growth,
reordering and extraction. `multiplex_history_encode_parity.py` compares neural
history reconstruction with the original memory host. These checks do not
establish full upstream demo-session parity for singleton extraction/merging;
the native dense-memory rebuild policy is explicit rather than inferred from
legacy object-axis handling in those methods.

### SAM3.1 dynamic interactive sessions

`sam3/multiplex_session.h` exposes `Sam31TrackingSession`. It shares a
`Sam31TrackingFrame` core and accepts a frame-feature provider returning both
interactive and propagation necks for one image. The latest frame is cached
across edits and consolidation. The provider may later use a shared visual
backbone; this session API currently operates on projected features.

The session supports arbitrary point sequences, boxes (labels 2/3), individual
and simultaneous mask prompts, fresh and incremental refinement, midstream
object insertion, forward/reverse propagation, clear/remove/reset, callback
stopping, and atomic cancellation between completed frames. It does not truncate
point sequences or cap object counts. `add_masks` accepts `[objects,H,W]` and a
matching vector of unique IDs, decodes the batch together and applies the
source's mutual brush suppression. Repeated individual brush calls instead give
later brushes precedence, as in the source. Coordinates can be normalized or
expressed in model pixels. UI brush previews preserve thresholded original-size
masks; preflight consolidates them at 288px and encodes full 1008px memory.

`preflight` finalizes edited masks and memory. `propagate` takes the same
`TrackingPropagation` controls as the SAM3 session, including an inclusive end
frame (`max_steps=0` yields one frame), optional preflight and deferred encoding.
`cancel()` is the thread-safe stop signal; other session operations must be
serialized by the caller. Outputs include full object IDs, low-resolution masks,
original-size logits and object scores. The native API returns the whole scene
for previews; the original point UI may return only the edited object. Object
IDs and existing slot assignments stay stable across refinement. New point
objects prefer new buckets, while mask objects fill available slots.

The session uses the explicit dense-history reconstruction policy above rather
than the source demo's legacy singleton history extraction/merge. It retains
full masks and shared image features, compresses spatial memory to BF16, and can
store retained data on CPU with `offload_state=true`. It deliberately overrides
the frame core's output trimming/offload flags to retain reconstruction inputs.
This currently costs substantial history RAM on long videos; disk-backed storage
and further retention optimization are still needed. Weights are shared and no
new model variants or weight copies are introduced. Failed edits/preflight
restore the previous session state. If clearing an input leaves only later
non-conditioning annotations, the earliest remaining annotation becomes a
conditioning frame so it remains usable. Removing the last object resets state.
Reset retains the frame-feature cache and shared core.

```sh
build/native/sam3_multiplex_session /private/native-weights-v1 cuda fp16
build/native-cpu/sam3_multiplex_session /private/native-weights-v1 cpu fp32
```

The standalone synthetic-feature probe covers 18 accumulated points, masks,
boxes, midstream insertion, repeated refinement, reverse propagation,
clear/removal, cancel/resume, failed-edit rollback, reset and simultaneous brushes.
It checks cache reuse and unchanged IDs. `multiplex_session_invariants.py`
compares interrupted/resumed execution with uninterrupted outputs and history,
and checks dynamic layout conservation and retained annotations.

`multiplex_session_parity.py` compares original point/box/refinement and one/two
object brush workflows, with explicit reference repairs: staging offloaded
mux/demux tensors to the matrix device, restoring F32 outside AMP, and rebuilding
annotation index sets lost during singleton extraction/merge. CPU reference
session constructors, including nested singleton states, are redirected to CPU;
otherwise their hard-coded CUDA device changes interpolation results. The
neural equations remain unchanged. These comparisons do not establish full
upstream multi-object editing equivalence or real-video quality for the native
stable-slot policy. SAM3.1 visual integration, full text-driven tracking, codecs,
release packaging and long-video storage optimization remain separate work.

### SAM3.1 real-frame tracking

`sam3/multiplex_vision.h` provides `Sam31TrackingVision` for a
`Sam31TrackingSession` feature provider. `encode_rgb` applies the same portable
video bicubic/byte-rounding preprocessing as SAM3; `encode_preprocessed` accepts
normalized F32 `[1,3,1008,1008]`. A single shared `VisionEncoder` evaluates the
full trunk once and produces the interactive and propagation necks. Their own
tracking decoders project the high-resolution maps. The session caches both
necks together for repeated edits. The model store stays unchanged; this adds
no weight copies or shape-specific model exports.

```sh
build/native/sam3_multiplex_video /private/native-weights-v1 cuda fp16 frames.txt commands.txt results
```

The standalone development probe shares the SAM3 PPM manifest, command parser
and output format described above. It additionally accepts simultaneous brush
prompts with `masks FRAME [ID PPM_PATH]...`; all masks in that operation must have
the same dimensions. IDs, prompts and operation counts remain caller supplied.
`remove ID` changes SAM3.1 session state without emitting previews; subsequent
propagation emits the remaining objects. These small probes share one native
library and the existing modular weight store; they are not separate complete
model distributions.

`multiplex_video_parity.py` compares real decoded JPEG pixels through the full
visual backbone and session against the adapted original demo. It covers point
prompts, correction, forward/reverse propagation and two simultaneous brush
prompts. The original staging and annotation-index repairs are documented in
`multiplex_session_parity.py`; FP16/FP32 use an ordinary MLP instead of the
original forced-BF16 fused MLP. The native child runs with `PATH=/nonexistent`.
This does not yet provide JPEG/MP4 decoding, high-level text tracking or a
relocatable runtime package. Broad accuracy evaluation and long-video storage
optimization remain separate work.

### Lossless SAM3.1 history paging

Set `MultiplexSessionOptions::history_directory` to a caller-owned directory to
page frame payloads to disk. The default empty path keeps the resident policy.
The standalone video/session probes accept this optional path as their final
argument. Disk paging does not change precision, object limits, prompts or
memory selection. It keeps low/high masks, image features, encoded memory,
positions, pointers and scores available for future edits and reverse tracking.

The tracking core reads only the spatial and pointer streams selected by the
existing temporal planner. It reads output masks separately. A bucket-layout
change processes one old frame at a time and writes replacement archives before
committing the new history. If reading, reconstruction or writing fails, the
session preserves its prior edit state; completed propagation frames stay
available. Archives retain tensor dtype and strides, including channels-last
and expanded tensors, and check each payload's CRC before use.

`sam3/multiplex_storage.h` provides `load_multiplex_frame` for inspection of
paged state and `load_selected_multiplex_history` for temporal consumers.
Undefined payload fields in a frame with an archive are **not** evidence that
its history was discarded. Small selection metadata stays resident. A state
copy shares immutable archive ownership; replacement/reset deletes old files
only after the last reference releases them. `TensorArchive` reserves unique
subdirectories and uses standard C++ filesystem/streams with existing zlib;
there is no mmap or platform-specific runtime dependency.

These are temporary process-owned caches, not resumable session checkpoints.
Normal teardown reclaims them; a killed process can leave its cache files for
the application to clean up. The caller owns the parent directory. Disk usage
still grows with retained history, and a layout-changing edit temporarily needs
both old and replacement archives. This trades memory for disk traffic rather
than compressing the model or dropping inference features.

Paging bounds retained **frame payloads** in RAM; it is not a bound on total
process memory. The active temporal working set, model, frame cache, allocator
caches, annotation input masks and small per-frame metadata remain separate.
SAM3's non-multiplex session does not yet use this backend. Further position
sharing, annotation paging, bounded I/O caching and performance work remain.

`multiplex_storage_parity.py` compares resident/paged dynamic sessions at zero
tolerance. `multiplex_storage_video.py` compares real-frame standalone outputs
and stresses retention by repeating decoded frames. The archive CTest covers
stride/dtype conservation, independent ownership, CRC/truncation/missing-file
failures, and loading selected frames while an unselected file is unavailable.

The mask-update probe also accepts a final history directory. In that mode it
compares resident versus paged append/recondition results, including CPU-staged
low/high masks, before propagating the updated state. Runtime cache files are
not model distribution artifacts. Peak process RSS measurements exclude the
operating system's reclaimable filesystem cache.

### C ABI for native hosts

`sam3/c_api.h` exposes opaque context, image, video and result handles using
standard C types. Clients can tokenize/encode text, ground images, refine image
masks and drive the implemented SAM3/SAM3.1 interactive video sessions without
Torch headers or C++ source. Sessions share lazily loaded modules within a
context; image/video uses the same visual backbone. The modular weight store
is unchanged and has no new image/video variants.

See [C API ownership, layouts and callbacks](../docs/native/C_API.md) for ABI
version checks, lifetime rules, synchronous host-buffer inputs, result views,
thread-local errors and video callback cancellation. `sam3_c_api_probe` is a
C11 example and validation client. It runs with Python removed from PATH; the
shared library still requires matching LibTorch/CUDA/ICU/zlib dependencies.
This is not yet a relocatable SDK or high-level text-guided video tracker.

### Detection-to-track association

`sam3/association.h` ports the source SAM3/SAM3.1 association policies, including
IoU/IoM autocast arithmetic, resize/sign threshold order, ambiguity suppression,
reconditioning metadata and different empty-input branches. Optional SAM3.1
zero padding is preserved without imposing an object limit. It also provides
source-equivalent box-boundary filtering and device placement plans.

`sam3_association_test` runs without Python or weights. The development parity
script calls the actual Python source methods, including native metadata
realization comparisons. See [video integration contracts](../docs/native/VIDEO_INTEGRATION.md)
for the remaining high-level state policies and integration work. This module
is not yet a complete text-guided video tracker or multi-GPU executor.

### Hotstart state and masklet confirmation

`sam3/hotstart.h` implements separate source host/device lifecycle policies,
immutable state updates, removal compaction, dynamic extension, reorder selection
and confirmation across ID changes. The device path supports CPU/CUDA and avoids
the source `[detections,objects,objects]` temporary using exact FP32 binary-count
matmul in its valid integer range. It falls back to the original reduction beyond
that range without truncating inputs. Persistent pair metadata remains quadratic.

`sam3_hotstart_test` needs no Python or weights. `hotstart_parity.py` compares
actual source methods and extracted original planning blocks; `hotstart_benchmark.py`
measures only the isolated state update. See [video integration](../docs/native/VIDEO_INTEGRATION.md)
for state semantics, numerical bounds, measurements and remaining host integration.

### Recent occlusion and reconditioning

`sam3/occlusion.h` adds source-specific suppression/history updates, reconditioning
gates, video mask cleanup and ordered edit recipes. The SAM3.1 helper connects to
hotstart state and survives compaction/extension. Association metadata preserves
candidate insertion order because the original SAM3.1 gate uses the first pair's
IoU. Mask-to-box extraction avoids full-resolution coordinate temporaries without
changing inclusive extrema or empty-mask behavior.

`sam3_occlusion_test` runs without Python or weights. `occlusion_parity.py` checks
original source methods, extracted gate blocks, edit batches with a recording
tracker and consecutive state transitions. It does not execute neural session
edits or complete the high-level video predictor. See [integration contracts](../docs/native/VIDEO_INTEGRATION.md)
for model-specific score/history rules, source quirks and remaining integration.

### Executing video corrections

`sam3/video_recondition.h` executes prepared corrections through real SAM3/SAM3.1
sessions and memory preflight in model-specific order. SAM3.1's
`recondition_masks` corrects existing IDs/current frames without changing their
bucket layout. Preflight releases consolidated temporary previews, so repeated
correction starts from stored logits. See [neural execution contracts](../docs/native/VIDEO_INTEGRATION.md)
for transaction scope, history storage and the still-missing high-level video
coordinator. The standalone session tools exercise this API without Python.

### Global video memory replacement

`sam3/video_memory.h` prepares globally suppressed memory masks/proxy scores and
maps explicit global IDs to local sessions. Session updates preserve predicted
masks/scores, support SAM3.1's optional no-object pointer projection, and retain
the effective encoder inputs for lossless history rebuilding after layout changes.
`sam3_video_memory_test` needs no Python or weights; the session tools additionally
exercise actual neural memory replacement. See [global memory contracts](../docs/native/VIDEO_INTEGRATION.md)
for source differences, storage behavior, validation scope and remaining coordinator
work. No multi-GPU communication or complete high-level predictor is claimed.

`sam3/video_objects.h` owns collections of tracker sessions for detector births
and removals. SAM3 starts a new state per birth group; SAM3.1 uses stable best-fit
placement without limiting group/object counts. Factories share existing cores
and caches, so new sessions do not duplicate model weights. The API prepares
full-resolution binary masks, runs mask insertion/preflight, and prunes empty
states. SAM3.1 removes multiple objects with one history remap. Placement,
original actual-weight new-state workflows and native best-fit/storage workflows
are tested separately; the complete high-level predictor remains unfinished.

`sam3/video_update.h` composes frame-update planning, ID/score/confirmation metadata,
association/hotstart/reconditioning/occlusion policies, local neural execution and
raw output-mask assembly. The caller provides detector outputs and globally ordered
tracker predictions; the planner preserves inputs, and the executor performs
correction, memory, births and removal in dependency order. Policy arithmetic is
FP32 by default; there is no object cap. Predictor prompt/cache/user-action state,
temporal output filtering, distributed communication and a complete high-level
C ABI remain unfinished. See `docs/native/VIDEO_INTEGRATION.md` for the split between
original policy comparisons and actual-weight synthetic execution probes.

### Shared detector/tracker video frames

`sam3/video_frame.h` evaluates a single full vision trunk for detection and all
required tracker necks. It supports the existing grounding prompt API and
prompt batches, source video joint-presence scoring, and uncapped source-specific
NMS/query filtering. See [video integration](../docs/native/VIDEO_INTEGRATION.md)
for the distinction between SAM3 greedy, SAM3.1 batched and alternate perflib NMS.

A development-only standalone integration probe accepts a UTF-8 prompt and PPM
frame manifest (paths relative to the manifest):

```sh
build/native/sam3_video_pipeline_probe STORE sam3.1 cuda fp16 \
  FRAMES.txt sam3/assets/bpe_simple_vocab_16e6.txt.gz PROMPT.txt OUTPUT_DIRECTORY
```

It runs native text encoding, shared visual features, detection, propagation,
update planning/execution and raw mask output. The probe exposes one text prompt;
the library accepts batches. It is not the complete video predictor: temporal
output buffering, prompt/user-action lifecycle and codec input are still pending.
It writes packed masks, raw tensor files and per-frame JSON for validation, not a
final application output contract. No new weight variants are required.

The high-level `VideoFrameEncoder::encode_rgb` path now matches the source video
image-folder loader's Pillow bilinear/F16 normalization. It is deliberately
separate from the low-level tracker bicubic/F32 preprocessing API. SAM3's probe
also enables source score-based memory selection and preserves auxiliary text
batch slots for numerical comparison. `--trace` optionally writes SAM3.1 current
history tensors; these weight-derived debug outputs belong in private storage.

The coherent source comparison is `native/tests/video_pipeline_parity.py`.
SAM3 BF16-reference matches all raw masks and low tracking values in the current
three-frame fixture. SAM3.1 has a captured divergence after the first global
memory update; it is not yet a passing full-pipeline comparison. See the updated
[video integration notes](../docs/native/VIDEO_INTEGRATION.md) for measured scope,
FP16/BF16 comparisons and the remaining investigation.
