# Native runtime development log

## 2026-09-22 — mask/NMS foundation

User clarified that no Turing hardware will be available. Do not block on its
procurement. Continue independently and optimize after functional completion
until asked to stop. Hardware performance claims must remain Blackwell-only.

Implemented a shared C++/CUDA library, no Python link dependency:

- Mask packing/unpacking, independently padded rows, noncontiguous input support.
- Chunked bilinear resize and sigmoid with original dtype rounding.
- Generic IoU-matrix NMS, stable score ties, strict greater-than suppression,
  no detection count cap. Current CUDA baseline uses one cooperative block for
  greedy suppression and an ATen boolean matrix, leaving optimization scope.
- Development dispatcher registration and standalone C++ CPU/CUDA executables.
- Initially added Windows/Linux CPU CI; subsequently removed per user instruction.

Validation: 4 local CTest cases and 46 CPU + 46 CUDA parity scenarios passed.
Coverage includes empty input, tail bits, noncontiguous layouts, float16/32/64,
FP16 sigmoid threshold rounding, invalid shapes, tied scores, threshold equality,
8,203 detections, and a nondefault CUDA stream. See `ops-validation.json`.
`cuobjdump` found sm_75 and sm_120 cubins (Torch CMake also added intermediate
architectures). C++ CUDA test passed with Python removed from PATH. Shared
library is linked to the development installation's LibTorch, not yet packaged
for standalone relocation. Turing hardware execution has not been tested.

Next work: finish connected components / EDT, capture model reference fixtures,
measure shared checkpoint tensors, implement the common weight store and native
model components. No complete SAM3 native inference, C ABI, Windows CUDA proof,
or model-level numerical comparison exists yet.

## Validation ownership update

User requested no GitHub Actions for now and will handle Turing/Windows testing.
The native workflow run was canceled, the workflow disabled and removed. Linux
CPU CI had completed successfully before cancellation; Windows results are not
a gate or a claimed validation. Future checks run locally; no Actions polling
or new CI runs should be introduced. Provide user-facing reproduction commands
for Windows/Turing without blocking development on those platforms.

## 2026-09-22 — connected components, EDT and sharing inventory

Added CPU/CUDA 8-connected equal-value components with 64-bit labels/counts,
root-index labeling and zero background. CUDA uses atomic union-find and never
merges across image boundaries. Added separable Euclidean distance transform
with float32 output and double-precision envelope intersections; singleton
spatial dimensions and empty batches are supported. Upstream's finite infinity
for all-foreground input is retained.

Validation completed locally:

- 4 CTest CPU/CUDA cases, plus a separate build with custom CUDA kernels disabled
  (2 CTest cases), all passed.
- Independent reference suite: 67 CPU + 67 CUDA scenarios passed, including
  noncontiguous images, multivalued components, diagonal connectivity, large
  cross-block regions and brute-force distance comparisons.
- Direct original-Triton comparison: 6 mask, 4 NMS, 4 component and 2 EDT cases
  passed. `upstream-ops-validation.json` records the environment.
- Compute Sanitizer memcheck and synccheck: zero errors on the standalone CUDA
  primitive test. These checks exercise the probe, not a full model workload.

Exact checkpoint inventory: 3,088 tensor references, 2,998 unique typed/shaped
contents, 6,951,840,400 logical bytes and 6,898,771,600 unique bytes. Exact
sharing saves 53,068,800 bytes; only 33,733,632 unique bytes occur in both model
versions. Thus SAM3 and SAM3.1 encoders must not be assumed to share parameters.
The larger distribution saving will come from reusing each version's modules
across image/video use, not duplicating full checkpoints per execution path.
`native/tools/inventory_weights.py` reproduces the analysis; the detailed
inventory is private and the size summary is `weight-sharing.json`.

All of these primitives are available to C++ callers and development tests;
the original SAM3 Python inference paths have not yet been rewired. The native
library is still not a full model runtime. Next: model reference fixtures,
on-demand shared weight storage, C++ model components and host session control.

## 2026-09-22 — lossless module store and native language encoder

Implemented the v1 weight store, C++ reader and lossless exporter. Metadata is
opened without allocating the full model; individual tensors or module prefixes
can be loaded to CPU/CUDA and released by the caller. Identical typed/shaped
contents are stored once, including cross-model/module references. The complete
private store is 6,900,865,193 bytes across 17 files, including metadata/alignment.
Original source checkpoints are backups outside the deployable store.

All 3,088 named tensor references were read with payload checksum validation by
a standalone C++ executable, with Python removed from PATH. Synthetic exporter
and C++ reader comparisons passed exact bytes for 12 dtypes (including BF16 and
complex), scalar/empty/noncontiguous tensors, sharing and CPU/CUDA placement.
Truncation, inconsistent shape, checksum corruption, invalid shard paths and
stale alias inputs are rejected. Format and commands: `WEIGHTS.md`.

Implemented the complete 24-layer SAM3/SAM3.1 VE encoder in C++/ATen, reading
only the language module from the shared store. The API accepts arbitrary token
batches/lengths within the model's existing 32-token context; it returns all
three outputs used by upstream VETextEncoder. The unused pooled projection is
not executed because upstream VE discards it. Tokenizer is not yet ported.
GPU FP32/TF32-off comparisons against both checkpoint versions passed on
multi-prompt, variable-length and single-token cases; maximum absolute error
was zero in all six cases. See `text-cuda-validation.json`.

Captured unmodified SAM3 image eager reference tensors (precision corrected below) using the truck
asset: three independent text prompts, a positive box, positive+negative boxes,
and a changed confidence threshold. All 200 architectural queries and the
original 1008 input resolution were retained. The private reference directory
contains backbone/text features, raw detector outputs and final results; summary
is `image-reference-summary.json`. These are reference fixtures, not evidence
of a native image implementation. Interactive point/mask, batch and video
fixtures remain outstanding.

Next: native tokenizer, visual backbone/neck and geometry/detector components,
then interactive image and video/session control. Keep user-owned Turing/Windows
validation and no-Actions instruction in force.

CPU text comparisons also passed with zero maximum absolute error for the same
six cases (`text-cpu-validation.json`). The standalone C++ text CLI loaded the
private module store and executed on CUDA with Python removed from PATH. The
custom-CUDA-disabled build passed its 3 CTest cases; CUDA-enabled build passed
6. This proves the module/loader path for those inputs, not tokenizer or whole
model parity. Model modules and session lifetime control remain incomplete.


## 2026-09-22 — reference precision correction

Inspecting the visual MLP exposed an inherited CUDA autocast context from the
tracker constructor (`sam3_tracking_predictor.py`). The earlier image capture
had float32 parameters but BF16 autocast, not pure FP32 execution. Actual stored
outputs confirm BF16 vision features/predicted logits/masks and FP32 boxes.
The historical private directory `reference/image-sam3-fp32-v1` is retained to
avoid breaking references; its metadata and the public summary now explicitly
correct the precision description. Capture tooling now records actual autocast
state, parameter dtype and output dtypes. No tensor data was changed.

This does not affect the prior text encoder parity results, whose reference
construction does not instantiate the tracker and was FP32. Visual validation
will distinguish unchanged upstream BF16 fused-MLP reference from FP32/FP16
Linear/GELU/Linear paths (the latter matches the existing Turing MLP replacement).

## 2026-09-22 — native visual trunk, dual/tri neck and RGB preprocessing

Implemented `sam3::VisionEncoder` with all 32 ViT layers, 16-head local/global
attention, checkpoint complex RoPE, tiled absolute positions and complete necks:
SAM3 `convs`/`sam2_convs` (four scales each), SAM3.1 detection/interactive/
propagation (three scales each). Native returns the full neck output; the
higher-level caller will apply each original backbone wrapper's `scalp` policy.
A single trunk is reused for all requested heads. Selecting a head skips only
unrequested output branches, and never reduces detection queries or input
resolution. Model input remains 1008×1008 after normal preprocessing.

Precision modes are explicit and scoped: `fp32`, `fp16`, and `bf16_reference`.
The BF16 mode reproduces the original fused MLP for reference comparisons;
FP32/FP16 use Linear/GELU/Linear, following the MLP replacement already used by
the Turing patch. BF16 reference execution is not the Turing deployment path.
A native RAII guard restores caller autocast state instead of leaking settings.

GPU comparisons passed for both models in all three modes, plus a two-image
FP16 batch (8 cases in total). All compared trunk, neck and position tensors
were exactly equal in this environment; `vision-cuda-validation.json` contains
per-output shapes/dtypes/errors. Head selection and restoration of an outer
BF16 context are checked separately in `vision-selection-validation.json`.

Added RGB tensor preprocessing matching torchvision v2's dtype scaling,
antialiased bilinear resize, integer rounding and normalization. All 110 CPU/
CUDA cases matched exactly, covering nine dtypes, multiple source sizes,
empty batches, noncontiguous input and channels-last strides. Source CPU and
CUDA resize rounding can differ; process pixels on the target device as the
original processor does. C++ output always has normalized NCHW layout and
1008×1008 spatial dimensions. File decoding remains a separate host concern.

A standalone `sam3_vision` CLI successfully ran preprocessing and all SAM3.1
necks on CUDA with Python removed from PATH. It uses synthetic RGB pixels as a
module probe, not a full image segmentation app. The CUDA-enabled build passed
6 CTests; the custom-CUDA-disabled build passed 3. GitHub Actions were not used.

Outstanding model work still includes native string tokenization, geometry
encoding, image encoder/decoder fusion, mask heads, interactive segmentation,
video tracking/multiplex and full host session/C ABI/package integration.

The complete SAM3 visual trunk/dual-neck FP32 comparison also passed on CPU
with zero maximum error (`vision-cpu-validation.json`). Error-path testing
confirmed that rejecting an unknown neck head restores outer autocast state.
The comparisons use the original full input resolution and all 32 blocks;
no model-size or detection-count reduction was introduced.

## 2026-09-22 — detector geometry encoder and native ROIAlign

Implemented the full configured `SequenceGeometryEncoder` for both SAM3 and
SAM3.1 as `sam3::GeometryEncoder`, reading only the existing geometry module
from the private store. Includes direct coordinate projections, sampled image
features, sine positions, positive/negative labels, variable-length right-padded
concatenation, CLS, post projection and all three self/cross-attention layers.
Points use normalized xy and boxes normalized cxcywh; no prompt count cap or
fixed prompt was introduced. Empty point/box sequences retain the CLS path.
The model's detector geometry configuration has no mask encoder; interactive
mask prompts remain part of the outstanding tracker/interactive implementation.

Added precompiled C++/CUDA inference ROIAlign, including adaptive/fixed sampling,
aligned/unaligned coordinates and torchvision-compatible autocast behavior.
Sampling is adapted from torchvision commit `7a13ad0f`; its BSD license is
retained under `native/third_party/torchvision`. This adds no torchvision native
or Python runtime dependency. Kernels honor the CUDA device/current stream and
include sm_75 code. The CPU version is a correctness baseline that repeats
sampling weights across channels; further performance work remains possible.

Validation on the development environment:

- ROIAlign: 88 CPU/CUDA comparisons, including half/float/double, autocast BF16
  and FP16, noncontiguous input, boundaries, empty ROIs, adaptive/fixed sampling,
  and aligned/unaligned behavior. Maximum absolute error was zero in all cases.
- Geometry: 39 CUDA cases across both checkpoints and FP32/FP16/BF16-reference,
  and 13 CPU FP32 cases. All embeddings and padding masks exactly matched the
  Python source. Covers empty, points, boxes, mixed padding, all padded entries,
  37 points plus 29 boxes, and saved real SAM3 visual features.
- Invalid ROI indices/NaNs, invalid geometry labels/padding/modes were rejected.
  Geometry normal and exception paths restored an outer BF16 autocast context.
- Standalone C++ geometry probes executed on CPU and CUDA with Python absent
  from PATH. Dynamic dependencies contain no libpython or libtorch_python;
  the current development LibTorch installation still supplies ATen libraries.
- CUDA-enabled CTest: 6 passed. Custom-CUDA-disabled CTest: 3 passed. No Actions.

Reports are `roi-align-validation.json`, `geometry-cuda-validation.json` and
`geometry-cpu-validation.json`. Native probes/build logs are saved privately
under `native-foundation/geometry-linux-cuda13`. These are development binaries,
not yet relocatable Windows/Linux release packages. Turing and Windows runtime
validation remain with the user as requested.

Next: image/text/geometry fusion encoder, detector decoder and segmentation
heads. String tokenization, interactive/video/multiplex sessions and a stable
C ABI/full standalone distribution also remain outstanding.

## 2026-09-22 — image/prompt fusion and detection transformer

Added native `DetectorEncoder` and `DetectorDecoder` using the existing modular
weight archive, with no new weight copies. Fusion executes all six layers and
retains prompt padding, image padding and valid-ratio metadata. Decoder executes
all six layers, all 200 learned detection queries, presence token, sine box
positions, learned logarithmic relative-position bias and iterative refinement.
It exposes normalized layer outputs, pre-refinement anchors, presence logits
and presence features for the final score/mask heads.

Attention follows the source's packed/separate projection choices and its
mixture of explicit bmm/softmax and ATen SDPA. The decoder FFN preserves the
source's explicit CUDA FP32 region within mixed precision. The source's
unassigned presence-logit `clamp()` has no effect; native preserves the actual
unclamped decoder output rather than silently changing model behavior.

A parity failure identified a source precision detail: the constructor's
standard 72×72 coordinate cache is built with integer scalars, while other
sizes are built from device tensor scalars. CUDA division uses different
rounding in these two paths. Native now follows both paths. The test retains
the original standard cache (moving it to CPU for CPU comparison), rather than
clearing it and inadvertently changing reference behavior. Tolerances were
not increased to accommodate the failure.

Validation reports:

- `detector-encoder-cuda-validation.json`: 21 cases across both models and
  FP32/FP16/BF16-reference, including full 72×72 image features, varying prompt
  lengths, three-image padded batches, and saved real SAM3 vision/text features.
  Every compared tensor was exactly equal.
- `detector-decoder-cuda-validation.json`: 18 cases across both models and all
  three modes, comparing every layer's 200 query features, anchors, presence
  logits and final presence features. Every compared tensor was exactly equal.
- `detector-encoder-cpu-validation.json`: 7 FP32 cases, exactly equal.
- `detector-decoder-cpu-validation.json`: 6 FP32 cases passed, maximum absolute
  feature error 1.205e-5; maximum anchor error 3.577e-7. The standard source
  coordinate cache is generated on CUDA then moved to CPU; native CPU generates
  coordinates locally. These CPU results are within the original test tolerance,
  not bitwise-identical for the standard grid.

Diagnostic comparisons also exposed that normal PyTorch Parameters versus
inference tensors can select different noncontiguous linear kernels through
`requires_grad` metadata, even during inference. Keep construction context
explicit when extending comparisons; current module reports construct the
reference under inference mode, as did the earlier image reference capture.
End-to-end image/application equivalence remains to be demonstrated.

The new `sam3_detector_transformer` C++ probe chains geometry→fusion→decoder.
Scoring, final box/mask heads, string tokenization, interactive/video/multiplex
sessions and standalone packaging remain outstanding. No GitHub Actions were
used; Windows/Turing runtime validation remains assigned to the user.

The linked geometry→fusion→decoder probes passed on CUDA FP16 (SAM3.1) and CPU
FP32 (SAM3) with `PATH=/nonexistent`, retaining `[6,200,1,256]` query features
and `[6,200,1,4]` anchors. Dynamic dependency inspection found no libpython or
libtorch_python. CUDA-enabled CTest passed 6/6; custom-CUDA-disabled CTest passed
3/3. Binary/log snapshots are stored privately at
`native-foundation/detector-transformer-linux-cuda13`; matching development
LibTorch libraries are still required, and these are not release packages.


## 2026-09-22 — connected image grounding and native executable

Added `DetectionHeads`, `GroundingDetector` and `postprocess_image`. The native
chain now runs RGB preprocessing, all vision/text layers, geometry prompts,
fusion, all six decoder layers and 200 queries, score/box heads, pixel decoding,
instance/semantic masks, confidence selection and original-size restoration.
Text, geometry and optional visual prompt sequences remain variable; batched
image/text indexing, padding and previous-mask features are retained. No full
weight variants or precision-specific weight files were added. Vision and text
objects can be released independently of the retained grounding modules.

Image and video detector scoring differ upstream. The image path multiplies
presence at postprocessing; the joint-score path combines it inside the detector.
Both paths are exposed, so the same probability is not multiplied twice. Mask
resize is chunked without limiting detections. The probability output preserves
the upstream dtype, including CUDA autocast promotion during interpolation.
CPU sigmoid is applied once after assembling chunks: per-chunk SIMD tails can
otherwise introduce an ULP difference from the source processor.

The real-image integration test caught a subtle preprocessing issue missed by
value-only tests. For unbatched interleaved RGB, torchvision inserts the batch
dimension after transforming the 3D tensor. The resulting singleton stride is 3,
not the full image size. Identical values with those different strides selected
numerically different convolution behavior on this GPU. Native preprocessing
now restores the source singleton stride and explicitly tests it. Tolerances
were not raised to hide the discrepancy.

Validation on RTX PRO 4500 Blackwell / installed development LibTorch:

- `detection-heads-cuda-validation.json`: 36 cases, both models and all three
  precision modes, full grid, repeated/remapped image batches, both scoring
  modes; every compared tensor exactly equal.
- `detection-heads-cpu-validation.json`: 12 FP32 cases, exactly equal.
- `text-mixed-cuda-validation.json`: 18 cases including caller autocast
  restoration, exactly equal.
- `grounding-fixture-validation.json`: three saved real-image/text feature
  chains, raw scores, boxes and all 200 masks exactly equal.
- `image-end-to-end-validation.json`: 14 real-image tensor pipeline cases,
  both models, FP32/FP16/BF16-reference, text/geometry and a three-item batch
  with padded visual/geometry prompts and previous-mask features. Every
  compared tensor exactly equal. SAM3.1 uses its tri-neck/weights and joint
  scoring with common detector math, not the unported multiplex scheduler.
- `image-results-validation.json`: 42 saved/synthetic comparisons, original-size
  masks, ragged image sizes, empty results and all 200 detections retained at a
  permissive threshold, different resize chunk sizes; exactly equal.
- `preprocess-validation.json`: 110 CPU/CUDA preprocessing cases passed,
  including the newly asserted singleton stride.
- `image-probe-validation.json`: standalone C++ child with `PATH=/nonexistent`,
  real RGB pixels, arbitrary token IDs for "truck", "wheel" and "a yellow
  butterfly". Counts 1/4/0; boxes, scores and every output mask pixel exactly
  equal to saved upstream results. About 11 seconds per fresh process includes
  loading all modules; this is not a warmed performance benchmark.

`sam3_image_probe` accepts P6 PPM and token IDs, writes JSON plus packed masks,
and needs no Python interpreter. It is a development probe: Unicode/BPE string
processing, image codecs and session APIs remain pending. The SAM3.1 FP16
native process also runs all 200 queries on this GPU. Dynamic dependencies
contain no libpython/libtorch_python. CTest passed 6/6 CUDA-enabled and 3/3
custom-CUDA-disabled tests. Current binaries still depend on the installed
NVIDIA development LibTorch; they are not relocatable release packages.

Code and reports are pushed to `codex/native-onboarding`. Binary/log snapshots
are saved privately under `native-foundation/image-grounding-linux-cuda13`,
with real native results under `reference/image-native-v1`. Native text
normalization/tokenization, interactive point/mask sessions, video tracking and
SAM3.1 multiplex orchestration, stable C ABI and standalone packaging remain.
No GitHub Actions were used. Windows/Turing runtime verification remains with
the user; these measurements do not establish Turing runtime or performance.


## 2026-09-22 — native Unicode cleaning and text-prompt inference

Added `sam3::Tokenizer` and `sam3_tokenize`. The native frontend implements the
configured VE tokenizer: default ftfy 6.1.1 text repair, including repeated
mojibake correction through nine candidate single-byte codecs, lossy UTF-8 and
CESU-8/Java-null recovery, inconsistent embedded encoding repair, HTML handling,
C1 controls, ligatures, widths, quotes, line breaks, surrogate repair, terminal
escapes/control removal and NFC; then double HTML unescape, whitespace/lowercase
cleaning and byte-level BPE. Variable batches/context retain upstream framing,
padding and truncation. Runtime takes UTF-8; malformed byte strings fail clearly.
The optional alternate ftfy/VE cleaning configurations are not model features
and are not exposed; this ports the configuration actually used by both models.

The existing gzip vocabulary is read directly with zlib and shared by SAM3 and
SAM3.1. No conversion script runs at build or inference time, and no model weight
copies were added. Frozen character/entity/codec tables occupy about 254 KB of
source. Their reproducible development generator pins ftfy 6.1.1, CPython's
Unicode 15.0.0 and regex 2025.11.3. Notices/licenses for ftfy, CPython and ICU are
included. ICU supplies native regex, NFC and lowercasing; tested ICU 74.2/zlib
1.3. CMake accepts ICU 72–74 (Unicode 15.x), requiring parity revalidation before
upgrading to a newer Unicode data profile. This avoids silently changing prompt
semantics when a system ICU is upgraded. Windows/Linux builds are supported by
[ICU's official instructions](https://unicode-org.github.io/icu/userguide/icu4c/build.html);
Windows builds still require their matching ICU/zlib DLLs. The selected model
runtime remains the LibTorch C++ path supported by the attached C++ docs.

A broader combining-mark comparison found a real source-engine peculiarity:
with IGNORECASE, U+0345 matches neither the positive letter branch nor the
negative non-letter/number branch. Taking the complement of the letter class,
or applying ICU's case-fold closure to it, changes tokenization. The exporter
now freezes each regex branch separately. The failing case was retained, and
no tolerance or input restriction was introduced to bypass it.

`tokenizer-validation.json`: 21,697 cleaned-text and complete token-array
comparisons passed exactly. This includes all 1,433 changed lowercase mappings,
all 5,857 code points with Unicode decompositions, all 2,450 combining-mark
contexts around Greek sigma, 7,214 direct/nested named entities, numeric entities,
character fixes, multilingual text, random Unicode, multi-layer/lossy/mixed
mojibake, special tokens, long prompts and the million-codepoint segment boundary.
The C++ child ran with `PATH=/nonexistent`; Python only supplied reference data.
CTest also covers empty batches, contexts 1/2/7/77, end-token truncation, malformed
UTF-8, invalid context and reuse of a moved tokenizer.

`sam3_image_probe ... --text-file BPE.gz PROMPT.txt` now reads arbitrary UTF-8
text directly. `image-text-probe-validation.json` verifies complete native
text-file→image-grounding execution for the three saved reference prompts.
Counts remain 1/4/0; boxes/scores and every mask pixel match exactly. Python is
not on the child process PATH and is not linked into the executable. Existing
token-ID input is retained for low-level callers. The model still runs all
200 queries; original confidence filtering is the only result selection.

CUDA-enabled CTest passed 7/7 and custom-CUDA-disabled CTest passed 4/4. Code and
reports are pushed to `codex/native-onboarding`; binaries/logs are saved privately
under `native-foundation/tokenizer-linux-cuda13`, with native text/image outputs
under `reference/image-text-native-v1`. These are development snapshots requiring
compatible LibTorch/ICU/zlib, not complete relocatable release packages.

Next: interactive point/mask prompt modules and image session orchestration,
then video tracking and SAM3.1 multiplex control. Image/video codecs, stable
C ABI and standalone packaging remain. No GitHub Actions were used; Windows
and Turing runtime verification remains with the user.


## 2026-09-22 — interactive prompt encoder and mask decoder

Added `InteractivePromptEncoder` and `InteractiveMaskDecoder` for both models.
SAM3 loads `tracker.sam_*`; SAM3.1 loads its separate
`tracker.model.interactive_sam_*` prefixes from the existing modular archive.
No new weight shards, model variants or copies were created.

Prompt encoding retains variable point counts, labels -1/0/1/2/3, optional box
corners, masks, empty input and the learned no-mask embedding. It reproduces the
pixel-center shift, conditional point padding, Fourier coordinate/grid encoding,
mask downscaling and source LayerNorm2d operation order under each precision mode.
Full configured sizes remain 1008 input / 72 embedding / 288 mask prompt; tests
also exercise non-square grids without changing the deployment model defaults.

The decoder includes both two-way attention layers, all four mask tokens,
object/IoU tokens, high-resolution skip projections, transpose convolutions,
hypernetwork mask generation, mask quality and object logits. It supports the
source single-mask and three-candidate outputs, repeated prompts for one image,
and stability-based fallback. A source detail is preserved: the single-mask
object-pointer token remains token zero even when stability selects another mask.
SAM3 uses sigmoid IoU; SAM3.1's interactive decoder uses unactivated IoU scores.
The separate multiplex propagation decoder has not been ported by this change.

Exact comparisons against original modules:

- `interactive-prompt-cuda-validation.json`: 96 cases across both models,
  FP32/FP16/BF16-reference, full/non-square grids, empty/all prompt combinations,
  noncontiguous inputs and 317 points; all compared tensors exactly equal.
- `interactive-prompt-cpu-validation.json`: 32 FP32 cases, exactly equal.
- `interactive-decoder-cuda-validation.json`: 90 cases, full 72×72 features,
  repeated prompts, channels-last batches, all three precision modes, all raw
  candidate masks/tokens/scores, multi/single output and forced/stable fallback;
  every compared tensor exactly equal.
- `interactive-decoder-cpu-validation.json`: 30 FP32 cases, exactly equal.

The standalone `sam3_interactive` probe chains both modules and high-resolution
projection on synthetic full-size image features. With `PATH=/nonexistent`,
SAM3.1 CUDA FP16 processed three prompts with nine points each and returned
`[3,1,288,288]` single masks and `[3,3,288,288]` multimasks, retaining all four
raw candidates. SAM3 CPU FP32 also passed with an empty point sequence. This is
module execution evidence, not yet a real-image interactive session test.
CTest passed 7/7 CUDA-enabled and 4/4 custom-CUDA-disabled tests. Neither the
library nor the probe links Python. Turing/Windows runtime testing remains
with the user, and no GitHub Actions were used.

Code/reports are pushed to `codex/native-onboarding`; development binaries/logs
are saved privately under `native-foundation/interactive-modules-linux-cuda13`.
Next: connect visual features and interactive host behavior (coordinate/mask
transforms, no-memory embeddings, object gating/pointers, postprocessing and
reusable image sessions), then memory/video tracking and multiplex orchestration.
Standalone packaging and stable C ABI remain outstanding.

## 2026-09-22 — reusable real-image interaction

`InteractiveImageSession` now connects canonical RGB-byte preprocessing, the
selected interactive vision neck, no-memory embedding, prompt transforms,
decoder, component cleanup and original-size masks. It caches projected high
features and can accept an externally shared pyramid. The host may release the
vision trunk after setting the image. Variable point/box counts, normalized or
pixel coordinates, previous-mask prompts, multiple original image sizes and
batched images are supported. No model variants or weight copies were added.

The implementation preserves box corners prepended to the point stream, source
padding, separate single-image/batch tensor layouts, and both component passes
reading the original logits. Returned low-resolution logits are clamped only
after full-size postprocessing, as in the source. Image predictions do not use
the video-only object gate. `sam3_interactive_image` demonstrates initial
three-candidate inference followed by mask refinement using cached features.

Validation against original code:

- `interactive-image-cuda-validation.json`: 57 exactly matching cases across
  both models and three precision modes, including projected feature reuse.
- `interactive-image-cpu-validation.json`: 29 exactly matching completed cases;
  an intermittent original-only CPU crash remains open (see below).
- `interactive-image-probe-validation.json`: 12 exactly matching real-image
  outputs, both models × FP32/FP16/BF16-reference × initial/refined stages.
  All IoU values, low-resolution logits and original-size mask pixels match;
  the standalone C++ child runs with `PATH=/nonexistent`.
- `interactive-image-batch-validation.json`: four real-image batch comparisons
  (both models, FP16, batch size one/two) match in cached embeddings and outputs.
- Existing text-image probe regression still matches all three saved prompts.
  CTest passed 7/7 CUDA-enabled and 4/4 custom-CUDA-disabled tests. The native
  probe has no Python library dependency; CUDA objects include sm_75.

CPU comparison failed intermittently in the original Python prompt encoder's
positional-encoding trigonometric expression. A reproduction without loading any
new native library failed once in five fresh processes. The cause is unresolved;
the successful parity report is not a CPU stability claim. The original-only
diagnostic is checked in, and failure/success/GDB logs are preserved privately.
See `CPU_RUNTIME_ISSUE.md`. Reference-only component-labeling dependencies are
pinned to NumPy 1.26.4 / scikit-image 0.25.2 / tifffile 2025.6.11.

Code/reports are pushed to `codex/native-onboarding`; development binaries and
logs are saved under `native-foundation/interactive-image-linux-cuda13`, and
real-image outputs under `reference/interactive-image-native-v1`. These still
require the development LibTorch/ICU/zlib installation. SAM3.1 comparisons cover
its interactive modules with a common image host, not multiplex scheduling.
Next are video object gating/pointers, memory encoding/attention and tracking
orchestration; codecs, stable C ABI and standalone packaging also remain.
No GitHub Actions were used. Windows/Turing execution remains with the user.

## 2026-09-22 — video interactive heads and object pointers

Added `VideoInteractiveHeads`, loading the existing SAM3 tracker / SAM3.1
interactive tracker weights. This connects missing-point padding, antialiased
mask-prompt resize, interactive decoding, object-presence gating, best-IoU mask
and token selection, and object-pointer projection. SAM3 uses its learned fixed
absent-object pointer; SAM3.1 applies its learned linear absent-object transform.
SAM3 processes a feature batch, while SAM3.1 repeats one image over object prompts.
The separate SAM3.1 multiplex propagation head is not implemented by this change.

Direct supplied masks now follow the original `use_mask_as_output` path,
including learned mask downsampling for the pointer, mask-derived object scores,
and the source's second absent-object pointer transformation. The original
different low-resolution sizing formulas and dtype choices are preserved.
In particular, SAM3 uses `input_size // 14 * 4`, whereas SAM3.1 uses
`input_size // 4`; neither is silently replaced with a fixed 288 output size.

`video-heads-cuda-validation.json` records 57 exact comparisons (both models,
FP32/FP16/BF16-reference). `video-heads-cpu-validation.json` records 19 exact
FP32 comparisons. Tests call the original tracker methods with their original
neural modules and compare every returned mask, IoU, pointer and object logit.
Coverage includes single/three-candidate output, empty/variable point sequences,
full/low/non-square mask prompts, combined prompts, present/absent objects,
direct empty/nonempty masks at two input sizes, and SAM3.1 object thresholds.
The previously documented intermittent development CPU failure remains open;
these successful runs do not establish general CPU stability.

The standalone `sam3_video_heads` probe supports arbitrary positive batch size
and nonnegative point count. It runs without Python on PATH and exercises prompt
and direct-mask paths. No weight variants or duplicate shards were introduced.
These tests establish head parity on synthetic features, not temporal tracking
accuracy. Memory encoding/attention, frame selection and session scheduling,
SAM3.1 propagation/multiplex control, codecs, C ABI and final packaging remain.

CUDA FP16 SAM3.1 (batch 3, nine points) and CPU FP32 SAM3 (batch 1, empty point
sequence) standalone probes passed with `PATH=/nonexistent`. CTest passed 7/7
CUDA-enabled and 4/4 custom-CUDA-disabled checks. The probe links no Python
library, and native CUDA objects retain sm_75 builds. Code/reports are pushed
to `codex/native-onboarding`; binaries/logs are saved privately under
`native-foundation/video-heads-linux-cuda13`. Windows/Turing runtime checks
remain with the user. No GitHub Actions were used.

## 2026-09-22 — mask-memory encoding and frame-memory host

Added `MaskMemoryEncoder` for both models, using the existing modular weights.
The neural path includes the 1152×1152 mask resize, four downsampling stages,
per-pixel LayerNorm2d, image projection, two ConvNeXt fusion blocks and output
projection. A float32 72×72 positional grid is retained per loaded module,
then repeated/cast with the same layout as the original CUDA builder's cache.
SAM3 outputs 64-channel spatial memory; SAM3.1 outputs 256-channel memory.

`encode_frame` reproduces the shipped tracker host behavior: mask sigmoid and
scale/bias, SAM3 binarization for point-origin masks, optional non-overlap
suppression, and absent-object spatial embeddings. SAM3.1 additionally packs
mask and conditioning channels using the caller's selection matrix. Matmul is
retained because autocast rounding is observable before mask downsampling.
Its missing/excess object-score handling and contribution from unused slots
are preserved. SAM3.1 does not binarize point-origin masks in this configuration.
The assignment controller and temporal-memory scheduling are still separate
outstanding work; accepting its matrix does not implement that controller.

One image feature may broadcast across all objects or multiplex groups, and CPU
staging of image features is supported. This preserves the original shared-image
path without requiring object-count-dependent feature copies. SAM3.1's 16 slots
per group are a model architecture dimension, not a total object cap: additional
groups handle more objects without new weight files or model variants.

`memory-encoder-cuda-validation.json`: 49 exactly matching cases for both models
and FP32/FP16/BF16-reference. `memory-encoder-cpu-validation.json`: 15 exactly
matching FP32 cases. Comparisons call the original neural modules and tracker
methods. Coverage includes shared/batched images, CPU-to-GPU staging, contiguous
and channels-last tensors, arbitrary/non-square/full-size masks, overlap,
point-origin masks, absent/present objects, empty/partial conditioning lists,
short/long score arrays, permuted assignments with padding, and 37 objects in
three groups. CUDA positional output strides also match the original cached
positions. CPU reference positions are generated on CPU because the upstream
precomputation constructor hard-codes CUDA; this is explicit in the report.
The earlier intermittent CPU runtime issue remains unresolved and is not claimed
fixed by these successful comparisons.

The `sam3_memory` standalone probe passed with Python absent from PATH for
SAM3.1 CUDA FP16 (37 objects / three groups) and SAM3 CPU FP32 (two objects).
Both use one shared image feature and reuse the loaded encoder for predicted and
point-origin masks. CTest passed 7/7 CUDA-enabled and 4/4 custom-CUDA-disabled
checks. Linked dependencies contain no Python library; custom CUDA objects
include sm_75. Windows/Turing runtime validation remains with the user.

Code/reports are pushed to `codex/native-onboarding`; development binaries/logs
are saved privately under `native-foundation/memory-encoder-linux-cuda13`.
No GitHub Actions were used. Next: temporal attention and frame-memory selection,
followed by tracking sessions and SAM3.1 multiplex propagation/control. Codecs,
stable C ABI and standalone release packaging also remain outstanding.

## 2026-09-22 — temporal memory attention without a Flash-only requirement

Added `MemoryAttention` for the four temporal encoder layers of both models.
SAM3 retains one-head attention, 64-channel memory and ReLU; SAM3.1 retains
eight heads, decoupled image/object query and key projections, 256-channel
memory and GELU. Axial complex rotary encoding is repeated for spatial memory
grids and excludes trailing object-pointer tokens. The source's SAM3.1 image
padding and pointer-position append behavior are preserved, along with shared
or batched image/position streams. Optional traces expose each layer for parity
diagnostics. No memory frame, object pointer or query count is truncated.

The native path calls LibTorch's precompiled SDPA dispatch and preserves caller
backend flags. It does not import FA3/Triton or force the original SAM3.1
Flash-only context. An isolated probe confirmed that this development build's
CUDA FP32 attention fails under that context with `No available kernel`.
For SAM3.1 CUDA FP32 reference comparisons, only the backend context is removed;
original weights and equations remain. Explicit math fallback comparisons also
prevent the original SAM3 forward method from re-enabling fused backends.
Tests verify that Flash, memory-efficient and cuDNN SDPA remain disabled in
those math cases. This verifies a usable fallback on Blackwell, not Turing
execution or performance. Original half-precision reference comparisons retain
their normal source backend choices.

`memory-attention-cuda-validation.json`: 45 exact comparisons across both
models and FP32/FP16/BF16-reference, checking all four layer outputs and final
normalization. Cases cover full 72×72 queries, multiple memory frames, variable
pointer counts, pointer-only memory, batched and shared image streams, shared
positions, pre-padded image pointers, mixed precision streams, and full-grid
math fallback. Smaller diagnostic grids also test rotary-grid recomputation.
`memory-attention-cpu-validation.json`: 13 exact FP32 comparisons. CPU reference
rotary caches are computed on CPU as they would be on a CPU-only host; the source
cache is not a registered buffer and otherwise initially resides on CUDA here.
Reports explicitly mark backend adaptations. The prior intermittent CPU runtime
issue remains open; successful runs do not establish overall CPU stability.

The standalone `sam3_memory_attention` probe passed with Python absent from PATH
for SAM3.1 CUDA FP16 with two memory frames, SAM3.1 CUDA FP16 with math-only
attention, and SAM3 CPU FP32. All use 72×72 queries and four pointer tokens.
CTest passed 7/7 CUDA-enabled and 4/4 custom-CUDA-disabled checks; linked libraries
include no Python library and the custom CUDA objects retain sm_75 builds.

Code/reports are pushed to `codex/native-onboarding`; development binaries,
backend-probe output and validation logs are saved privately under
`native-foundation/memory-attention-linux-cuda13`. Existing weight shards are
reused without new model variants. Next are frame-memory selection and temporal
position assembly, tracking sessions and SAM3.1 multiplex propagation/control.
Codecs, stable C ABI and final distribution packaging remain. No GitHub Actions
were used; Windows/Turing runtime checks remain with the user.

## 2026-09-22 — SAM3 frame-memory selection and conditioning

Added `Sam3MemoryConditioner`, `TemporalState`/`TemporalPlan`, the shared
`select_conditioning_frames` helper, and `memory_confidence`. State preserves
Python dict insertion order. Selection retains closest-before/after handling,
stable equal-distance ties, keep-first behavior, temporal stride, reverse
tracking, score-filtered memory and source pointer-history rules. The
keep-first/limit-two edge case retains Python's negative slice behavior;
no extra cap is imposed. Python floor division for negative boundary indices
is reproduced explicitly in C++.

Assembly loads spatial memory/positions from CPU when needed, adds learned
temporal positions, projects sine-encoded pointer times, and splits each SAM3
256-channel pointer into four 64-channel tokens. It returns assembled tensors
and a selection plan for diagnostics. The connected forward runs native attention
and restores BCHW features. Initial frames, explicit previous-memory bypass and
disabled memory retain their distinct source behaviors.

Effective-IoU scores remain scalar tensors, including when offloaded. Converting
scores to host doubles can change half-precision threshold comparisons. Missing
scores, NaN scores and threshold ties are covered. Confidence calculation retains
the source's broadcasting with both matrix and vector IoUs.

`temporal-memory-cuda-validation.json`: 600 randomized conditioning-selection
comparisons and 42 exact tensor cases across FP32/FP16/BF16-reference.
`temporal-memory-cpu-validation.json`: the same 600 selection cases and 14 exact
FP32 cases. Tests compare assembled memory, positions, pointer-token count and
the final conditioned features. Coverage includes forward/reverse/strided
tracking, score selection in both directions, start/end boundaries, unbounded
conditioning selection, initial/bypass/disabled memory, CPU-offloaded spatial
state and full 72×72 features. CPU reference adapts only hard-coded spatial
`.cuda()` transfers and rotary-cache construction to its execution device.
The earlier intermittent development CPU runtime issue remains unresolved.

`sam3_temporal` chains memory conditioning, video heads and memory encoding over
three synthetic full-grid frames with spatial state stored on CPU between frames.
CUDA FP16 and CPU FP32 passed with Python absent from PATH. Frame zero uses no
previous memory, frame one uses one spatial frame/four pointer tokens, and frame
two uses two spatial frames/eight pointer tokens. This is an integrated module
chain check, not real-video tracking validation. Frame count is a runtime argument;
the module API also supports variable batches and selection settings.

SAM3.1's temporal host remains to be connected after multiplex state/controller
and propagation decoding are ported; the shared selector is ready for it.
Full sessions, real-video parity, codecs, C ABI and final packaging remain.
No model-weight variants or copies were added. Code/reports are pushed to
`codex/native-onboarding`; development binaries/logs are saved privately under
`native-foundation/temporal-memory-linux-cuda13`. No GitHub Actions were used;
Windows/Turing runtime validation remains with the user.
CTest passed 7/7 CUDA-enabled and 4/4 custom-CUDA-disabled checks. The standalone
probe links no Python library; custom CUDA objects still include sm_75.

## 2026-09-22 — SAM3.1 multiplex state and controller

Added native `MultiplexState` and `MultiplexController` for inference. They
preserve dense internal object indices, optional stable external IDs, CPU RNG
permutations, capacity/padding, removed-slot occupancy, retained-bucket indices,
and allocation into existing or explicitly new buckets. The default physical
width is 16 and bucket count grows without an added object cap. Mux/demux retain
the original matrix multiplications and autocast rounding. Matrices are assembled
on CPU with one transfer each, retaining the original contiguous strides even
for singleton dimensions. Unindexed `cuda` resolves to the allocated GPU.

Mutation errors leave native state intact, unlike partially applied failures in
the source. Removing every object explicitly invalidates the state, zeroes active
counts and releases matrices instead of retaining source stale metadata. These
deliberate invalid-state differences are documented; valid inference transitions
are compared directly against the source.

`multiplex-cuda-validation.json` contains 156 exact cases across FP32, FP16 and
BF16-reference. `multiplex-cpu-validation.json` contains 52 exact FP32 cases.
Coverage includes widths 1/4/16, reduced capacity, full/object-only shuffling,
deterministic ordering, external IDs, 14-step add/remove sequences, all-object
removal, noncontiguous input and scalar/empty feature shapes. RNG seeds and matrix
strides match the reference. Each mutation case compares every intermediate state
and its mux/demux outputs. The standalone C++ safety/round-trip test additionally
covers rejected mutations, copied states, tombstones and up to 257 objects.
Exact FP32 round-trip tests disable TF32; the library preserves caller policy.

CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. The C++ test
passed with Python absent from PATH on CUDA FP32/FP16/BF16-reference and CPU FP32.
The executable links no Python library; custom CUDA objects retain sm_75 builds.
The earlier intermittent development CPU prompt-encoder issue remains unresolved.

Code/reports are pushed to `codex/native-onboarding`; development binaries and
logs are saved privately under `native-foundation/multiplex-linux-cuda13`.
No model weights or variants were added. SAM3.1 propagation decoding and temporal
orchestration, complete tracking sessions, real-video comparisons, codecs, C ABI
and final distribution packaging remain. No GitHub Actions were used;
Windows/Turing runtime validation remains with the user.

## 2026-09-22 — SAM3.1 multiplex propagation decoder and heads

Added `MultiplexMaskDecoder` and `MultiplexPropagationHeads`. They use the shipped
SAM3.1 weights: 16 object slots per bucket, three mask tokens per slot, separate
IoU/object tokens, a two-layer two-way transformer and shared high-resolution
image features. The host supplies live/invalid-slot suppression embeddings,
demuxes results into object order, gates absent masks, resizes all candidates to
1008px, selects by IoU and projects the chosen token into an object pointer.
The linear absent-pointer transform and optional stability attenuation are
preserved. Bucket count remains dynamic; no new object cap or weight copies
were introduced. These propagation weights have three candidates only; the
separate interactive heads retain their original single/multimask behavior.

`multiplex-decoder-cuda-validation.json`: 33 exact comparisons across FP32,
FP16 and BF16-reference. `multiplex-decoder-cpu-validation.json`: 11 exact FP32
comparisons. Coverage includes full 72x72 features and rectangular diagnostic
grids, contiguous/channels-last inputs, shared/per-bucket high-resolution maps,
optional per-object embeddings, one/17/20 live objects across up to three
buckets, removal/addition, forced presence/absence, projected high-resolution
features, dense positions and stability-based candidate selection. Every
candidate mask, IoU, token/object score and host output is compared directly.

`multiplex-decoder-math-validation.json` adds 11 exact CUDA FP16 comparisons with
only math SDPA enabled. For these fallback comparisons, the original
`Attention.forward` is prevented from re-enabling Flash/memory-efficient SDPA;
its tensor operations remain unchanged. The native library never overrides
the caller's backend policy. The test checks that only math remains enabled.

`sam3_multiplex_propagation` connects native allocation, propagation heads and
mask-memory encoding using synthetic full-grid features. With Python absent
from PATH, CUDA FP16 passed with 17 objects before/after a removal/addition
(two then three buckets), including a math-only run. CPU FP32 passed with two
objects before/after mutation (one then two buckets). This is an integrated
module probe, not a real-video accuracy test or a temporal-memory scheduler.
CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. No Python
libraries are linked and custom CUDA kernels retain sm_75 builds. The earlier
intermittent CPU prompt-encoder runtime issue remains unresolved.

Code/reports are pushed to `codex/native-onboarding`; development binaries/logs
are saved privately under `native-foundation/multiplex-decoder-linux-cuda13`.
Next is SAM3.1 temporal assembly/conditioning, followed by complete sessions
and real-video parity. Codecs, C ABI and final packaging remain. No GitHub
Actions were used; Windows/Turing runtime validation remains with the user.

## 2026-09-22 — SAM3.1 multiplex temporal memory and conditioning

Added `MultiplexMemoryConditioner`, `MultiplexTemporalState` and options for
SAM3.1's temporal host. Spatial selection reuses the validated ordered
conditioning/stride/score planner, while pointer selection preserves SAM3.1's
different defaults: future conditioning pointers are included and pointer time
distances are unsigned. V2 spatial time embeddings mark out-of-range frames;
past-only/signed pointers, v1 spatial time, dummy pointer time and disabled
pointers/memory are also available.

Assembly keeps image/object memory streams separate, restores CPU-offloaded
spatial/image features, projects sine pointer-time encodings and repeats each
frame's position over its 16 slot pointers. Legacy 5D per-slot features and
positions are demuxed, made contiguous and cached back into state. Missing
pointers are skipped. Cleared spatial memory and pointer-only history fall back
to current image features. Initial/bypass frames use interactive/direct-mask
heads, matching the original temporal method's unsupported initial branch.
History must already be aligned to the current bucket allocation; complete
session-level history changes after object additions/removals remain separate.

`multiplex-temporal-cuda-validation.json` contains 58 exact cases over FP32,
FP16 and BF16-reference, including a full-grid math-only attention comparison.
`multiplex-temporal-cpu-validation.json` contains 19 exact FP32 cases. Coverage
includes forward/reverse, stride, score filtering with missing/NaN/tied scores,
start/end boundaries, unrestricted conditioning selection, past/signed pointer
options, v1/v2/dummy time encoding, disabled/missing pointers, partially cleared
spatial state, empty/zero-batch memory fallback and legacy 5D normalization.
Comparisons inspect assembled object/image memory and positions, pointer counts,
cached-state mutations and final conditioned features. Full 72x72 features are
included. CUDA FP32/math reference removes only the original Flash-only context;
CPU redirects hard-coded CUDA transfers and reconstructs rotary caches on CPU.
Native calls preserve the selected backend policy.

`sam3_multiplex_temporal` chains direct-mask initialization, memory encoding,
temporal conditioning and propagation over synthetic full-grid frames, with
spatial and image history offloaded to CPU. Three frames passed with Python
absent from PATH: CUDA FP16 with 17 objects/two buckets, CPU FP32 with two
objects/one bucket, and CUDA FP16 math-only with two objects. The first frame
stores memory; subsequent frames consume 16 then 32 pointer tokens per bucket.
The source direct-mask path's 252x252 low-resolution output for a 1008px supplied
mask is preserved; propagation produces 288x288 low-resolution masks. Final masks
remain 1008px. This is a synthetic integration probe, not real-video validation.

CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. No Python
libraries are linked; custom CUDA kernels retain sm_75 builds. The earlier CPU
prompt-encoder runtime issue remains unresolved. Code/reports are pushed to
`codex/native-onboarding`; development binaries/logs are saved privately under
`native-foundation/multiplex-temporal-linux-cuda13`. Existing weight shards are
reused. Full tracking sessions, real-video parity, codecs, C ABI and final
packaging remain. No GitHub Actions were used; Windows/Turing execution remains
with the user.

## 2026-09-22 — SAM3 frame inference API for video sessions

Added `Sam3TrackingFrame`, `TrackingFeatures`, `TrackingFrameRequest` and ordered
`TrackingHistory`. The frame host composes memory conditioning, interactive or
direct-mask heads, optional memory encoding, confidence calculation, output
offload and source history trimming. It accepts cached/expanded image features
and projected high-resolution maps so multiple objects share backbone work.
Previous logits remain caller-supplied, supporting repeated edits before memory
consolidation. Memory encoding is also callable separately after per-object
results are consolidated. Prompt metadata remains in the request/session layer.

The source multimask policy is retained without limiting point count; 17-point
input is tested. Disabling memory encoding supports interactive previews.
Offload moves masks/spatial memory and optional IoU/confidence to CPU while
keeping pointers/object logits on the execution device. Trimming removes high
masks, IoU/confidence and spatial memory but preserves low masks, pointers and
object logits. Source score/offload-dependent pruning rules and reverse-tracking
index behavior are preserved. Disabled score selection avoids an unnecessary
GPU score read during trimming.

`tracking-frame-cuda-validation.json` contains 54 exact comparisons against
`Sam3TrackerBase.track_step` across FP32, FP16 and BF16-reference.
`tracking-frame-cpu-validation.json` contains 18 exact FP32 comparisons. Cases
cover one/17/zero points, previous-logit refinement, forward/reverse propagation,
1008/1152px supplied masks, deferred/disabled encoding, memory non-overlap,
offloaded confidence, low/high-score pruning, offload-dependent distant-history
pruning and single/multimask policies. Numerical outputs, tensor devices and
history retention are compared. CPU reference adapts hard-coded CUDA transfers
and rotary caches; no model equations are changed.

`sam3_tracking_frame` passed with Python absent from PATH on CUDA FP16 and CPU
FP32, using two objects over three synthetic frames. The first frame runs a
preview without memory, refines it with additional points/previous logits, then
encodes memory. Later frames propagate from CPU-offloaded history. CTest passed
9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. No Python libraries are
linked and custom CUDA kernels retain sm_75 builds. The earlier intermittent
CPU prompt-encoder issue remains unresolved.

Code/reports are pushed to `codex/native-onboarding`; development binaries/logs
are saved privately under `native-foundation/tracking-frame-linux-cuda13`.
Existing weight shards are reused. Next is the session layer: prompt
accumulation, per-object consolidation, edits/deletions and propagation state.
Real-video parity, SAM3.1 session rebucketing, text-driven tracking, codecs,
C ABI and final packaging remain. No GitHub Actions were used; Windows/Turing
runtime verification remains with the user.

## 2026-09-22 — SAM3 interactive tracking session and edit lifecycle

Added `Sam3TrackingSession` with a shared frame core and an image-feature provider.
Sessions cache one projected feature pyramid, expand it across object batches,
and reuse the existing weight shards. The API manages point/box accumulation,
brush masks, previous-logit refinement, per-object pending edits, consolidation,
forward/reverse propagation, cancellation/resume, clearing annotations, object
removal/remapping and reset. All supplied points are retained by default; the
comparison includes an accumulated 18-point prompt. This is the original
low-level interactive predictor, whose object IDs are registered before tracking;
higher-level dynamic discovery/text video association remains to be implemented.

Consolidation preserves missing-object pointers, overlapping brush strokes,
mandatory non-overlap before memory encoding, BF16 memory storage, optional CPU
offload, constant position caching and original output postprocessing. Storage
and inference dtypes are separate. A source FP32 failure was reproduced because
BF16 stored memory reached FP32 linear weights without autocast. Native memory
attention now restores compressed values to FP32 in FP32 mode. The reference
suite makes that explicit input conversion; FP16/BF16 math is unchanged.

Two integration defects were found and fixed rather than hidden by tolerances:

- Sparse, nonmonotonic annotations change ordered conditioning history when
  Python set iteration is replaced by sorted C++ maps. Annotation dictionaries
  now retain insertion order. A small native compatibility helper preserves
  64-bit CPython 3.12 integer-set traversal, including dictionary versus key-view
  update growth. No interpreter dependency is introduced; the existing PSF
  notice/license is extended for the algorithm.
- Asynchronous GPU-to-CPU mask copies were read by CPU consolidation before
  completion. Preview masks could differ while later stored tensors matched.
  CPU offload now guarantees completed copies before CPU resize/copy or external
  state inspection. The reference waits before CPU consolidation/snapshots too,
  preventing timing from becoming part of the numerical comparison.

`tracking-session-cuda-validation.json` records 9 exact operation sequences
(FP32, FP16, BF16-reference), totaling 87 operations and 4,185 compared tensors.
`tracking-session-cpu-validation.json` records 3 exact FP32 sequences, 29 operations
and 1,395 tensors. Comparisons include displayed/low masks, pointers/logits,
compressed memory/positions, scores, input metadata, per-object and global
history, pending edits, insertion order, consolidated sets and tracking direction.
Both runs also passed 608 frame-order cases and 16 independent output cleanup
cases. The CPU cleanup reference uses the original skimage component path.
Reference CPU transfers/rotary caches are adapted as in preceding milestones.

`sam3_tracking_session` passed with Python absent from PATH on CUDA FP16 and CPU
FP32: two objects, 17-point input, brush input, memory consolidation, eight output
callbacks across cancellation/resume/reverse correction, object removal and
reset. CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. The
shared library still contains sm_75 cubins and has no Python library dependency.
The earlier intermittent CPU prompt-encoder issue remains unresolved.

Code/reports are pushed to `codex/native-onboarding`; development binaries/logs
are saved privately under `native-foundation/tracking-session-linux-cuda13`.
This is a synthetic-feature session milestone, not a real-video accuracy result
or a relocatable release. Remaining work includes the visual/video provider,
SAM3.1 session rebucketing, high-level text-driven tracking, codecs, C ABI,
multi-GPU and final packaging/optimization. No GitHub Actions were used;
Windows/Turing runtime verification remains with the user.

## 2026-09-22 — real-frame SAM3 tracking through the visual backbone

Added `Sam3TrackingVision`, which shares the existing visual encoder and tracking
core. A session provider now accepts decoded RGB frames, runs the full 1008px
visual trunk and tracking neck, drops the original scalp level, projects the
high-resolution maps and supplies cached features for all objects. Repeated
clicks/consolidation reuse the session's most recent frame. No image/video model
variants or duplicated weight shards are introduced. Preprocessed F32 input is
also exposed for decoder-specific frame pipelines.

The original synchronous JPEG loader uses Pillow RGB bicubic and F64 byte / 255
rounded to F32 before normalization. This differs from the image processor's
bilinear path. Generic ATen byte bicubic differed by up to two levels at 4,594
pixels on frame 0 and 11,698 pixels on the truck image. Added a portable C++ CPU
resampler with Pillow-compatible coefficients, 22-bit fixed-point weights,
byte-rounded/clipped intermediate rows and noncontracted coefficient arithmetic.
Normalization uses division rather than a rounded F32 reciprocal multiplication.
The Pillow HPND notice/license is included. There is no Pillow/Python dependency
in the runtime. CPU row parallelism does not depend on a specific SIMD ISA.

Both CUDA-enabled and custom-CUDA-disabled builds passed 48 exact resize/real-RGB
cases plus nine normalization/layout cases against Pillow 12.2.0. Cases cover
single-pixel axes, up/downsampling, unchanged dimensions, odd shapes and real
720x1280/1200x1800 images. CUDA input staging was checked in the CUDA-enabled
build. The result matches the synchronous source frame-buffer layout.
`tracking-preprocess-*-validation.json` records the tests.

`sam3_tracking_video` accepts a manifest of arbitrary P6 RGB frames and an edit
command file. It supports point/box/mask prompts, consolidation, propagation,
clear/remove/reset and callback/cancel stopping, and writes packed masks plus
F32 mask values and metadata. This is a development frame-sequence interface;
it does not yet decode JPEG or compressed video. Paths and commands are provided
by the caller rather than fixed in the executable.

`tracking-video-cuda-validation.json` records 30 exact real-frame output
comparisons: FP16, BF16-reference and FP32 on three 720x1280 frames from the
repository video sequence. Operations include two prompted objects, forward
tracking, a correction after propagation, reverse tracking and removal/reset.
Compared outputs include full-resolution logits, binary masks, low-resolution
logits and object scores. All native child processes ran with PATH=/nonexistent.
The workflow made five visual-backbone calls for ten outputs, preserving cache
reuse for multiple objects, previews and consolidation. Reference FP16/FP32 use
the previously documented ordinary MLP replacement; FP32 compressed-memory and
D2H synchronization adaptations remain explicit.

Same-mode port parity does not prove precision-mode equivalence. Relative to the
original fused BF16 reference on this fixture, the minimum per-object mask IoU
is 0.9565003 for FP16 and 0.9571946 for FP32. `tracking-video-precision.json` keeps
all per-output differences, reproducible with `compare_tracking_modes.py`.
These are precision-agreement measurements, not ground-truth accuracy or a
representative dataset benchmark. Broader quality evaluation remains required.

CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. Python libraries
are absent from the native executable's linkage; sm_75 cubins remain present.
Turing/Windows runtime verification stays with the user; no Actions were used.
Code and reports are pushed to `codex/native-onboarding`. Private development
binaries/logs are under `native-foundation/tracking-vision-linux-cuda13`; decoded
fixtures, outputs and reference tensors are under `reference/tracking-video-v1`.
Remaining work includes codecs (and their decode/resize parity), SAM3.1 sessions
and rebucketing, high-level text/video association, C ABI, multi-GPU, packaging
and further accuracy/performance/size optimization. The earlier CPU source
prompt-encoder issue remains unresolved; this is not a stability claim for it.

## 2026-09-22 — SAM3.1 multiplex inference frame host

Added `Sam31TrackingFrame`, connecting both decoder heads, temporal selection,
object-pointer multiplexing and mask-memory encoding. The frame request selects
direct masks, interactive initialization/refinement, pure propagation, or
propagation plus corrections to selected objects. It retains all input points,
full 1008px output and arbitrary compatible bucket counts. Mixed corrections
preserve the source's indexed replacement/broadcast of candidate masks, IoUs,
pointers and logits. Conditioning object flags, deferred memory, saved image
features, CPU offload and temporal trimming follow the source frame policies.
Added the optional SAM3.1 interactive IoU stability attenuation, also used to
choose its best mask. Default SAM3 behavior is unchanged.

A parity failure exposed another singleton-stride requirement: the SAM3.1
memory encoder consumes the source's sequence-to-BCHW view. Passing the same
channels-last pixels with a different singleton batch stride changed FP16
convolution results despite identical masks. Reproducing that view restores
exact memory equality without a numeric tolerance or precision workaround.

`multiplex-frame-*-validation.json` records 63 CUDA comparisons (21 each FP16,
BF16-reference and FP32), 21 CPU FP32 comparisons, and three CUDA FP16 math-SDPA
fallback comparisons. All 1,099 compared tensor outputs match exactly. Cases
include 17-point initialization, empty points, 1008/1152 mask inputs, previous
logits, 17-object/two-bucket propagation, ordered partial corrections, reverse
selection, optional scores, deferred encoding, overlap handling, offload and
old/distant history trimming. Math tests use three objects to fit the fallback
working set; this is a test-fixture choice, not an API cap. These are synthetic
full-grid features, not a real-video quality benchmark. CUDA FP32 removes the
source Flash-only context; CPU adapts source CUDA transfers and rotary caches.

The standalone `sam3_multiplex_frame` passed with PATH=/nonexistent on CUDA FP16
(17 objects, two buckets, three frames) and CPU FP32 (two objects, three frames).
It chains a 17-point preview/refinement, partial correction and propagation with
CPU-offloaded memory. CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled
checks. Native linkage has no libpython/libtorch_python dependency; sm_75 cubins
remain present. Development binaries still require matching LibTorch/ICU/zlib
and are not a relocatable release. The intermittent original CPU prompt-encoder
issue remains unresolved despite this passing CPU run.

The frame host expects compatible history/bucket assignments. Dynamic object
insertion/reconditioning and demo-session singleton extraction/reintegration
remain to be ported; interaction-only requests with a subset require a matching
extracted state. Ground-truth-driven training correction is outside inference.
Code/reports are pushed to `codex/native-onboarding`; private development builds
and logs are stored under `native-foundation/multiplex-frame-linux-cuda13`.
The larger goal remains active: SAM3.1 sessions/rebucketing, codecs, high-level
text/video association, C ABI, multi-GPU, packaging and quality/performance/size
optimization are unfinished. No GitHub Actions were used; Turing/Windows runtime
verification remains with the user.

## 2026-09-22 — SAM3.1 dynamic current-frame mask updates

Added `Sam31TrackingFrame::update_masks` for the original dynamic model's mask
insertion and reconditioning methods. Appends allocate consecutive internal
indices with optional global object IDs, honor capacity/removed-slot policies,
and optionally grow or prefer new buckets. Reconditioning overwrites selected
objects in caller order. The implementation preserves the source's low-mask
antialiased resize, high-mask resolution adjustment on append, absent-object
pointer transformation, demux/merge/remux order, conditioning flags and memory
re-encoding. It retains optional input-mask storage and saved image features.
Changes are staged so an exception does not partially mutate caller state.

`multiplex-update-*-validation.json` records 45 CUDA cases (15 each FP16, FP32
and BF16-reference) and 15 CPU FP32 cases. All 796 compared tensor outputs match
the original dynamic methods exactly. Cases cover growth from 15 to 18 objects,
new-bucket preference, removed slots, constrained bucket capacity, resolution
changes, empty masks, ordered partial replacement, optional IDs/input masks,
score fields, overlap policy and deferred memory. The reference uses full-grid
synthetic features; this is not a real-video quality result.

The source updates primary masks/logits, selected IoUs, multiplexed pointers
and optional memory, while leaving existing candidate tensors and effective
confidence untouched. The native method preserves this behavior, including
potentially older auxiliary shapes after append. A surrounding session must
refresh the derived fields it uses. Mask reconditioning always re-encodes with
mask provenance, matching the source. Storage offload belongs after the update.

The standalone `sam3_multiplex_update` ran with PATH=/nonexistent on CUDA FP16
(three to 20 objects across three buckets) and CPU FP32 (three to five objects
across two buckets). Both initialized, appended, selectively reconditioned and
propagated from the updated conditioning frame. They also verified capacity
failure leaves caller state intact and reconditioning preserves an unrelated
object's mask. These chains intentionally have a single updated conditioning
frame; they do not prove older-history remapping after bucket growth.

CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. The new native
probe has no libpython/libtorch_python linkage and sm_75 cubins remain present.
The existing intermittent source CPU prompt-encoder issue remains unresolved.
Code/reports are pushed to `codex/native-onboarding`; development binaries and
logs are saved under `native-foundation/multiplex-update-linux-cuda13` in the
private bucket. They require compatible LibTorch/ICU/zlib and are not a portable
release. No Actions were used; Windows/Turing runtime checks remain with the
user. Full session history remapping/singleton reintegration, codecs, high-level
tracking, C ABI, multi-GPU, packaging and optimization remain unfinished.

## 2026-09-22 — SAM3.1 dense-history remapping and reconstruction

Added `remap_multiplex_history` and `Sam31TrackingFrame::encode_history` to carry
older frames across object/bucket layout changes. States carry unique global
IDs; masks, logits, IoUs and conditioning indices follow those IDs. New objects
are absent in older frames, with zero historical pointers/IoUs. Pointer slots
are copied without extra arithmetic. Unchanged physical buckets retain their
full pointer and spatial-memory contents, including historical removed slots,
as in original object removal. Internal index renumbering alone needs no neural
reconstruction. Image features stay shared and memory retains its storage device
and dtype, including BF16 CPU storage.

The shipped dense memory is [buckets,256,72,72], jointly encoding 16 slots. It
cannot safely be demuxed as an object-axis tensor. Changed bucket membership or
slot placement therefore requires a reconstruction callback; the provided
encoder uses retained full-resolution masks, logits, conditioning flags and
image features. Reconstructed values replace changed buckets; unchanged buckets
remain exact. Missing reconstruction inputs produce an error, not silent loss
or a guessed object slice. Changes across all frames commit together. Aggressive
history trimming must retain/reload/recompute the needed inputs before such a
change; full session storage integration remains unfinished.

`multiplex-history-*-validation.json` contains 20 original demo removal core
comparisons across CPU FP32 and CUDA FP32/FP16/BF16-reference, plus 12 independent
layout-conservation checks for ID reordering, bucket growth and singleton
extraction. Removal compares masks, logits, pointers, dense memory/positions,
IoUs, effective confidence and conditioning sets; original positional caching
is used. Input clearing and per-object view rebuilding are excluded from this
component test. Additional 12 neural reconstruction cases compare exactly with
the original memory host (24 tensor outputs), using CPU-stored image sequences
and full-resolution masks, 18-object growth, singleton extraction and slot
reassignment/overlap. These are synthetic feature tests. This does not claim
full parity with upstream demo singleton extraction/merging: the explicit
native dense-memory reconstruction policy replaces reliance on legacy
object-axis assumptions there.

`sam3_multiplex_history` passed with PATH=/nonexistent on CUDA FP16 (three to 20
objects, three buckets) and CPU FP32 (three to five objects, two buckets). Both
retain two earlier frames, reconstruct their changed buckets, propagate using
that history and recondition the new objects. Checks cover an injected failure
on the second frame with no partial history commit, byte-exact preservation of
an unchanged bucket, absent historical masks and BF16 CPU storage. The probes
use shared synthetic image features and the actual weights; real-video session
quality still needs integration/evaluation.

CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. Native linkage
has no libpython/libtorch_python dependency, and sm_75 cubins remain present.
The existing intermittent original CPU prompt-encoder issue remains unresolved.
Code/reports are pushed to `codex/native-onboarding`; private development builds
and logs are stored under `native-foundation/multiplex-history-linux-cuda13`.
They require compatible LibTorch/ICU/zlib and are not a relocatable release.
Full SAM3.1 sessions/retention policy, codecs, high-level tracking, C ABI,
multi-GPU, packaging and further quality/performance/size work remain. No GitHub
Actions were used; Turing/Windows hardware validation remains with the user.

## 2026-09-22 — SAM3.1 dynamic interactive session API

Added `Sam31TrackingSession` with shared core weights and a cached provider for
both projected tracking necks. It connects point/box prompts, individual and
simultaneous masks, fresh/incremental refinement, midstream object insertion,
preflight, forward/reverse propagation, clear/remove/reset, callback stopping
and cancellation. All points are retained; object counts are not capped.
Simultaneous masks use one batched decoder invocation and mutual brush
suppression; repeated individual brushes give the latest brush precedence.
Original-size binary brush previews and 288px consolidation feed full 1008px
memory. Failed edits/preflight roll back state. Remaining annotations stay
usable when the last original conditioning input is cleared.

The native session deliberately keeps existing IDs/slots stable on refinement,
using singleton interactive heads and replacing selected results. It uses the
explicit dense-history reconstruction helper for layout changes instead of
copying the original demo's legacy singleton history extraction/merge. New point
objects prefer new buckets; masks use available slots. Spatial memory is stored
as BF16, with optional CPU retention. Full masks and shared images are retained
for reconstruction; core output trimming is overridden. This currently uses
substantial RAM for long videos. Disk-backed retention, position-cache sharing
and further storage optimization remain necessary. No model variants or extra
weight shards are introduced.

`multiplex-session-*-validation.json` records 12 CUDA workflows (four each FP16,
FP32 and BF16-reference) and four CPU FP32 workflows: 80 operations and 404 exact
output/ID/frame/count tensor comparisons. Workflows include point accumulation
(18 points), boxes, one/two-object brushes, forward tracking, first refinement,
reverse propagation and reset. These compare projected synthetic full-grid
features against the adapted original demo, not a complete real-video system.
The reference adaptations are explicit in the test/report:

- Original singleton merge multiplies a device-resident mux matrix by offloaded
  CPU history and fails. Reference mux/demux stage to the matrix device and
  restore F32 outside AMP; compressed-memory projection also restores F32.
- Removing the sole object during extraction clears consolidated annotation
  indices. Merge restores inputs/history but omits those indices, so the next
  preflight fails its equality assertion. Reference indices are restored from
  the actual merged inputs and frame stores.
- The original nested singleton constructor hard-codes CUDA even in a CPU test.
  This mixed GPU and CPU interpolation, producing up to 0.00003052 output error.
  Redirecting every nested constructor to CPU restored exact equality; no
  tolerance was introduced. CUDA FP32 Flash-only context removal and D2H
  synchronization remain as in earlier comparisons.

`multiplex-session-invariants-*.json` adds CUDA FP16, CUDA BF16-reference and CPU
FP32 checks. Each verifies 44 exact comparisons between interrupted/resumed and
uninterrupted outputs/history, plus an 11-operation dynamic edit sequence with
midstream growth, unchanged old-bucket memory/pointers, historical absence for
new objects, 18 accumulated points, reverse propagation, removal/clear/reset and
remaining-annotation promotion. This validates the explicit native policies;
it does not establish full upstream multi-object editing equivalence or quality
for the stable-slot policy. Broader real-video evaluation remains required.

The standalone `sam3_multiplex_session` passed with PATH=/nonexistent on CUDA
FP16 and CPU FP32, each producing 12 callbacks and exercising masks, boxes,
midstream insertion, repeated refinement, reverse, clear/removal, cancel/resume,
provider-failure rollback, reset and simultaneous overlapping brushes. The
latest-frame cache reused both necks across prompts; each run made nine provider
calls. CTest passed 9/9 CUDA-enabled and 5/5 custom-CUDA-disabled checks. Native
linkage has no libpython/libtorch_python dependency and sm_75 cubins are present.
The separate intermittent original CPU prompt-encoder issue remains unresolved.

Code/reports are pushed to `codex/native-onboarding`. Private development builds,
passing logs and reference-failure diagnostics are saved under
`native-foundation/multiplex-session-linux-cuda13`. They still require matching
LibTorch/ICU/zlib and are not a relocatable release. SAM3.1 visual integration,
long-video retention, high-level text/video association, codecs, C ABI,
multi-GPU, packaging and further quality/performance/size work remain. No GitHub
Actions were used; Turing/Windows runtime verification remains with the user.

## SAM3.1 real-frame visual integration

`Sam31TrackingVision` now connects decoded RGB to the dynamic tracking session.
It applies portable source-compatible video preprocessing, runs one full
1008px visual trunk, computes both tracking necks and projects their high-level
maps with the corresponding decoders. A single cached provider result serves
all objects and repeated prompts. `sam3_multiplex_video` shares the existing
SAM3 command parser/output format and adds batched brush input; it uses the
same modular weight store without additional exported weight variants.

`multiplex-video-cuda-validation.json` records 36 exact output comparisons on
three 720x1280 frames from `assets/videos/0001`: FP16, BF16-reference and FP32,
with a point/refinement/forward/reverse workflow and a two-object simultaneous
brush workflow. Video logits, available low logits, packed positive masks,
IDs and frame indices match the adapted original at zero tolerance. The point
workflow produces eight outputs with five backbone calls; the brush workflow
produces four outputs with three calls. Prompts are fixtures; the executable
accepts arbitrary coordinates, IDs, masks and command sequences. Each native
child ran with PATH=/nonexistent. Reference staging/annotation repairs remain
explicit, and the non-BF16 reference replaces the forced-BF16 fused MLP. This
is numerical parity for the tested workflows, not complete dynamic-edit or
high-level video parity.

`multiplex-video-*-precision.json` measures precision-mode agreement separately.
Minimum FP16-vs-BF16 mask IoU is 0.9993323 for points and 0.9926113 for brushes;
FP32-vs-BF16 is 0.9992885 and 0.9925850, respectively. These are a short fixture's
mode differences, not ground-truth segmentation accuracy or Turing results.

`multiplex-video-regression.json` records 36 byte-identical files from the
existing SAM3 FP16 real-video workflow after sharing the CLI source. The new
SAM3.1 custom-CUDA-disabled executable also completed a CPU FP32 full-backbone
single-frame smoke with PATH=/nonexistent, yielding two finite nonempty masks;
this is not an original CPU numerical comparison. The refactored reference
factory passed its CPU batched-mask comparison (19 exact tensors). CTest passed
9/9 CUDA-enabled and 5/5 custom-CUDA-disabled tests. Linkage has no libpython or
libtorch_python dependency, and sm_75 cubins remain present. The intermittent
original CPU prompt-encoder issue remains unresolved.

Code and reports are pushed to `codex/native-onboarding`. Private development
binaries/logs are saved under `native-foundation/multiplex-vision-linux-cuda13`,
and decoded fixtures/reference outputs under `reference/multiplex-video-v1`.
These builds depend on the current LibTorch/ICU/zlib environment; packaging is
not complete. Compressed codecs, long-video retention, high-level text/video
association, C ABI, multi-GPU and broader quality/performance evaluation remain.
No GitHub Actions were used. Windows/Turing runtime validation remains with the
user.

## Lossless SAM3.1 history paging

SAM3.1 sessions now accept an optional `history_directory`. Frame payloads are
written to immutable temporary archives; existing temporal selection reads only
the needed spatial/image and pointer streams, while preview/output reads only
mask/score fields. Arbitrary edits, reverse propagation and later object growth
retain their original inputs. Dynamic dense-memory remapping loads/rebuilds and
archives one old frame at a time, committing all replacements together. No
object, prompt, frame or detection limit was introduced; no model export or
weight variant was added. The resident policy remains the default.

The generic `TensorArchive` preserves values, dtypes and strides, including
expanded and channels-last views. It verifies per-tensor CRCs and rejects
missing/truncated/corrupted files. Exclusive temporary subdirectories and shared
ownership keep state copies valid; reset/replacement reclaims files only when
no frame references them. It uses C++ filesystem/streams and existing zlib, with
no platform-specific runtime API. Metadata stays in process, so these caches
are not serialized/resumable session checkpoints. A killed process may leave
cache files; the application owns cleanup of its parent cache directory.

Paging is integrated into the frame core as well as the session. Standalone
frame mask updates materialize archived fields and restore the execution
device. Clear/removal now also roll back their edits when archive reads fail.
The session probe temporarily hides a later archive during insertion, clear
and removal: prior IDs/history/dirty state survive, and failed staged archives
are reclaimed. The archive unit test checks that a missing unselected frame is
never read, but a missing selected frame fails.

`multiplex-storage-cuda-validation.json` records six resident-vs-paged workflows
(FP16, BF16-reference and FP32, each with score selection off/on), 16 operations
and 463 exact tensor comparisons per workflow: **2,778 exact comparisons**.
They cover new buckets, same-bucket brush insertion, repeated point edits,
reverse tracking, cancel/resume, removal, clearing, reset and simultaneous
brushes. Retained frame payload tensors are absent from the paged state;
selection metadata remains resident. These are native-policy equivalence tests,
not a new claim of upstream dynamic-layout equivalence.

`multiplex-storage-video-validation.json` records 144 byte-identical files for
the previously validated real-frame point/refinement/reverse and batched-brush
workflows across all three precisions. A longer retention test repeats the
three decoded images for 128 frames, comparing **516 files** between resident
and paged FP16 execution. Retained payloads total 2,262,434,048 bytes. Sampled
peak process host RSS falls from 3,829,436,416 to 1,747,148,800 bytes (about 54%).
Observed process durations are 22.55s and 22.14s. These single runs include model
loading, allocations and file I/O, use the local filesystem cache, and are not a
controlled inference-speed benchmark. RSS excludes reclaimable OS file cache;
this is not a claim of total system memory usage or ground-truth accuracy on
128 distinct frames. Native child processes use PATH=/nonexistent.

This reduces RAM used by retained frame payloads, not every memory category.
The active temporal working set, model, latest features, allocator caches,
annotation masks and per-frame metadata still consume memory. Disk usage grows
with history; layout changes temporarily retain both old and new archives.
Shared position storage, annotation paging, bounded I/O caching, SAM3 non-mux
session adoption and durable session serialization remain separate work.

The final CPU FP32 comparison completed under GDB with **926 exact tensors**
across the two selection policies. One preceding non-GDB rerun exited 139 with
a UCX null-address SIGSEGV; the GDB rerun exited normally and provided no failing
stack. `CPU_RUNTIME_ISSUE.md` records this native-only failure separately from
the earlier original-only reproduction. Passing CPU results establish numerical
parity for those runs, not CPU stability; the cause remains unresolved.

The standalone paged-session probes passed on CUDA FP16 and CPU FP32, including
archive-read rollback on add/clear/removal. Paged mask-update probes compared
12 output tensors exactly for append/reconditioning and then propagated:
CUDA 3-to-20 objects and CPU 3-to-5 objects. All completed probe caches were
empty afterward. CTest passed 11/11 CUDA-enabled and 6/6 custom-CUDA-disabled
checks, including the expanded archive tests. Native linkage still excludes
libpython/libtorch_python, and sm_75 cubins remain present. No GitHub Actions
were used; Turing/Windows runtime validation remains with the user.

Code/reports are pushed to `codex/native-onboarding`. Private binaries, successful
logs and the failed CPU log/GDB rerun are saved in
`native-foundation/multiplex-storage-linux-cuda13`. Real-video comparison data
are in `reference/multiplex-storage-final-v1`; the earlier 32-frame experiment
is kept in `reference/multiplex-storage-v1`. These are development builds, not
a relocatable release. Full high-level text/video association, C ABI, codecs,
multi-GPU, distribution packaging and further quality/performance work remain.

## C ABI for image and interactive video hosts

Implemented ABI 1 in `sam3/c_api.h`: standard C types, opaque context/image/video/
result handles, named tensor views, counted UTF-8 text, image batches and prompts,
grounding and both interactive video backends. There are 40 exported C symbols.
The header needs no Torch or C++ types. C++ exceptions are contained at status
boundaries, including CUDA/host out-of-memory translation. Errors are thread-local;
session reentry/concurrent use returns BUSY, while cancellation remains available
from callbacks. Results can outlive callbacks/sessions and children can outlive
caller-owned context handles. ABI layouts are fixed and checked before use.

Contexts load modules lazily, share immutable modules across sessions and release
unused cache entries on request. Image and video reuse the visual backbone;
interactive image instances share prototype tensor weights. This does not claim
that every small image/video module instance shares allocation. The modular
weight store is unchanged: no full image/video exports or weight variants were
added. Model inputs, all 200 detector queries and original prompt semantics are
retained. SAM3.1 history paging is exposed through the same video API.

`c-api-validation.json` records **654 exact comparisons** on local Blackwell:
36 two-stage image tensors, 258 video tensor/metadata comparisons, 324 grounding/
text comparisons and 36 image-batch tensors. The pure C11 client runs 18 workflows
(two models, FP16/BF16-reference/FP32, image/video/grounding) with PATH=/nonexistent.
Image refinement and real-frame video outputs match saved reference fixtures;
text/geometry/visual-feature and image-batch composition match the previously
original-validated C++ components. This is an API composition check, not new
proof of all high-level original features or unmodified SAM3.1 dynamic semantics.

Coverage includes row-padded RGB, original-resolution results, prior-mask image
refinement, multiple image/prompt batches, embedded-NUL tokenization, repeated
image/reordered text IDs, positive/negative boxes, visual features and previous
mask features with text disabled. The count test retains 200 detections. Video
checks include forward/reverse propagation, edits, callback cancellation/reentry,
provider failure, context/result lifetime and paging cleanup. These workflows
do not exhaust every combination of the exposed options.

An initial test incorrectly required equal images at different batch positions
to produce bit-identical outputs. The corrected test compares each position
independently against its corresponding C++ reference, still at zero tolerance.
Both FP32 models have a maximum low-logit difference of 3.814697265625e-6 between
positions; SAM3.1 also has one differing thresholded mask pixel. The C API and
C++ composition match exactly at each position. This records existing numerical
behavior rather than hiding the difference with a tolerance or mask filter.
FP16/BF16-reference batch positions are identical for this fixture.

CTest passes 13/13 CUDA-enabled and 8/8 custom-CUDA-disabled checks, including
pure C ABI/default/error validation and synchronized thread-local error tests.
The smoke client separately compiles/links with `cc -std=c11 -Wall -Wextra
-Werror` and no Torch include path. Native linkage excludes libpython and
libtorch_python; sm_75 cubins remain present. No GitHub Actions were used.
Windows/Turing execution remains with the user. CPU full-model stability remains
unresolved as documented in `CPU_RUNTIME_ISSUE.md`; this milestone's full-model
C ABI comparisons run on CUDA, not CPU.

Private binaries, public headers, logs and manifests are saved in
`native-foundation/c-api-linux-cuda13`; raw fixtures/results are in
`reference/c-api-v1`. See `C_API.md` for ownership, callbacks, layouts and build
instructions. These are development binaries requiring matching LibTorch/CUDA/
ICU/zlib, not a relocatable SDK. The goal remains active: high-level text-guided
video detection/association and its state policies, codecs, multi-GPU, portable
packaging and further quality/performance work remain. The C ABI exposes the
implemented lower-level backends and does not imply that those remaining
features are complete.

## Native detection-to-track association

Implemented `sam3/association.h` as a reusable high-level video building block.
It ports both source association wrappers and the shared matching arithmetic:
smaller-area bilinear resize before sign thresholding, IoU/IoM, threshold ties,
ambiguity clearing, reconditioning and host metadata realization. Source-specific
empty branches and optional SAM3.1 zero-mask padding are preserved. Padding is
not a cap: no detection/track count limit or dropping policy was introduced.
The module also reproduces normalized boundary filtering and lowest-workload
GPU placement, with multiplex groups kept together. Placement is a plan, not
multi-GPU inference execution.

`association-validation.json` records **792 workflows and 9,336 exact tensor/
metadata comparisons** against the actual repository Python methods across
CPU/CUDA, both models and FP32/FP16/BF16-reference. Another 72 placement and 32
boundary comparisons pass. Cases include empty inputs, false keep entries,
exact threshold boundaries, ambiguous matches, repeated assignments, nonfinite
logits/scores, noncontiguous views, unequal resolutions/equal-area shape ties,
257 detections against 513 tracks, dense full masks, and stored full 200-query
neural masks. The stored detector/tracker tensors come from different fixtures;
they test real-value arithmetic, not coherent-clip tracking quality.

SAM3 marks every supplied detection new when no tracks exist and only nonempty
tracks unmatched when no detections exist. SAM3.1 instead applies a score
threshold without intersecting keep in its no-track branch, and marks all tracks
unmatched in its no-detection branch. Both are deliberate reproductions of
source behavior. The original explicitly disables Hungarian matching; this
module implements its active many-to-one path.

A dense-mask check exposed source FP16 intersection overflow: 82,944 foreground
pixels in a 288x288 full mask exceed finite half range. IoU may become infinite;
IoM's float-to-int conversion behaves differently on the local CPU/CUDA paths.
The CPU FP16 path incorrectly treats two identical full masks as unmatched/new.
Same-mode parity reproduces this, so it must not be advertised as quality proof.
The native default is FP32 count arithmetic, independent of neural precision.
A standalone regression verifies that the default remains FP32 even inside
outer FP16 autocast. `association-dense-mask.json` preserves observed decisions.
The future host must retain this separation and validate end-to-end quality.

CTest passes 15/15 CUDA-enabled and 9/9 custom-CUDA-disabled checks. Standalone
CPU FP32/CUDA FP16 probes also run with PATH=/nonexistent, including the independent
FP32 count regression. Passing these CPU component checks does not resolve the
previously recorded intermittent full-model CPU runtime fault. Linkage still
excludes libpython/libtorch_python; existing sm_75 cubins remain. No GitHub
Actions were used and no Windows/Turing execution is claimed.

`VIDEO_INTEGRATION.md` records the next state-integration contracts. CPU hotstart
and SAM3.1 GPU hotstart differ in keep-alive updates, overlap ordering/ties and
suppression membership; they must be compared separately rather than unified.
Hotstart, confirmation, occlusion suppression, reconditioning, object insertion/
removal and propagation/cache coordination remain to be integrated into the
full text/visual-guided video host. C ABI existence and this association module
do not finish that host. Code/reports are pushed to `codex/native-onboarding`;
private binaries/headers/logs are in `native-foundation/association-linux-cuda13`.
No new weights or full-model distribution variants are needed.

## Native hotstart, state compaction and confirmation

Implemented the ID-indexed source hotstart state machine and the separate
SAM3.1 position-indexed device state machine in `sam3/hotstart.h`. These preserve
accumulated unmatched/overlap histories, keep-alive rules, first-frame ordering,
removal and suppression timing, including their intentional source differences.
State selection/compaction updates both pair-matrix axes and returns retained
indices; extension initializes all new entries. Inputs remain unchanged, while
unchanged output tensors may share immutable storage. Confirmation remaps IDs,
counts consecutive detections, resets counts on a miss and retains confirmed
status. These helpers are not yet connected to a complete high-level video host.

The source device overlap count materializes a float tensor of shape
`[detections,objects,objects]`. The native path uses FP32 binary matmul, preserving
exact integer counts for at most 2^24 detections, independently of neural autocast.
Above that range it retains the original reduction order. No input cap, dropping
policy, model export or weight variant was added. Persistent pair-count storage
is still quadratic; this optimizes the temporary, not all tracking memory.

`hotstart-validation.json` records **44,628 exact comparisons**: 8,064 host
state fields over 1,152 updates against both original classes, 35,568 device
fields/decisions over 1,440 updates across CPU/CUDA and three outer autocast modes,
960 confirmation fields, and 36 large-count boundary checks. The tests call
actual source methods and extract original compaction/extension AST blocks from
the planning phase, recording the source hash. Both 16,777,216 and 16,777,217
detection cases match on CPU/CUDA, verifying optimized/fallback boundaries without
a cap. They cover forward/reverse ordering, zero thresholds, empty states,
additions/removals/reordering and unchanged prior-state tensors. The native unit
test additionally checks 301 overlaps under BF16 neural autocast and the distinct
host/device first-frame-tie behavior.

An isolated Blackwell benchmark uses 200 detections and 512 objects, five warmups
and 30 samples per method in both execution orders. Source median CUDA-event time
is 0.792–0.795 ms; native is 0.377–0.383 ms. Peak additional PyTorch GPU allocation
falls from 217,432,576 to 6,568,448 bytes (about 97%), above the same 10,737,152-byte
baseline. Full state/decision outputs match before timing. These synthetic helper
measurements include stream work/launch gaps and exclude host RSS/reserved GPU
memory; they are not full-model/video speedups or Turing performance claims.
Raw samples are preserved in `hotstart-benchmark.json`.

CTest passes 17/17 CUDA-enabled and 10/10 custom-CUDA-disabled checks. Standalone
CPU/CUDA probes run with PATH=/nonexistent. The earlier intermittent full-model
CPU runtime issue is still open; successful state-component tests do not resolve
it. Existing sm_75 cubins and Python-free linkage remain. No GitHub Actions or
Windows/Turing execution was used.

Code and reports are pushed to `codex/native-onboarding`; binaries, headers and
logs are private in `native-foundation/hotstart-linux-cuda13`. `VIDEO_INTEGRATION.md`
documents the APIs and remaining work: recent-occlusion suppression,
reconditioning, coordinated object insertion/removal, visual prompts/caches,
user actions and full high-level video propagation. Codec, multi-GPU execution,
portable distribution and overall quality/performance validation also remain.

## Native recent occlusion and reconditioning preparation

Added `sam3/occlusion.h`: separate SAM3/SAM3.1 recent-occlusion policies, host and
device history adapters, box-based/periodic reconditioning gates, video-specific
hole/sprinkle cleanup, mask preparation and ordered edit batches. The device
adapter connects to hotstart state, preserving indices through compaction and
extension. Updates leave input state/masks unchanged. Default mask-IoU count
arithmetic remains FP32 under neural autocast; explicit reference modes reproduce
source arithmetic, including dense-mask FP16 overflow.

Preserved source differences: SAM3 existing history wins over removal, whereas
SAM3.1 removal overrides history. SAM3 uses raw track scores >0.8, while SAM3.1
uses sigmoid scores >0.8 and also merges low logits. Source finite removal
sentinels, reverse comparisons, empty-mask handling, cleanup after hole filling
and degenerate-box NaNs remain. Invalid missing global IDs are rejected during
preparation; valid object/prompt/detection counts are not capped.

Fixed association metadata to retain Python candidate insertion order in
addition to its lookup map. This affects SAM3.1's source gate, which tests the
first pair's IoU and any candidate's detection score. The previous map-only
comparison did not check order. The updated report has 852 association workflows,
10,968 exact tensor/metadata comparisons, 72 placement and 32 boundary checks.

`occlusion-validation.json` contains 18,450 exact comparisons across CPU/CUDA
FP32/FP16/BF16-reference modes: actual source methods, original extracted gate
blocks, original reconditioning methods with a recording tracker, and
hotstart -> occlusion -> compaction -> extension transitions. It covers reversed
history, first-pair order, missing/stored history, empty and 201-object batches,
noncontiguous tensors and dense 288x288 masks. Native strided cleanup is compared
to contiguous source values because source CUDA CCL requires contiguous storage.
The recording tracker validates edit batches without executing neural edits.

Mask-to-box extraction now uses axis projections and reuses them for the empty
check, avoiding full-resolution coordinate temporaries. Exact inclusive int32
boxes match source. For 200 synthetic 288x288 masks on Blackwell, five warmups
and 30 samples in both execution orders give median source time 0.929–0.930 ms
and native 0.172–0.175 ms. Peak incremental PyTorch GPU allocation decreases from
132,712,448 to 1,048,064 bytes above the same 16,777,216-byte baseline. These are
isolated helper measurements, not full-video/Turing throughput or total VRAM.

CTest passes 19/19 CUDA-enabled and 11/11 custom-CUDA-disabled checks. CPU/CUDA
standalone probes run with PATH=/nonexistent, and the library links neither
libpython nor libtorch_python. Existing sm_75 cubins remain. The earlier
intermittent full-model CPU failure is still open. No GitHub Actions or
Windows/Turing execution was used.

Code/reports are pushed to `codex/native-onboarding`; binaries, headers and logs
are preserved privately in `native-foundation/occlusion-linux-cuda13`. No new
weights or whole-model variants were created. Full neural edit/preflight
execution, coordinated insertion/removal and caches, text/visual prompt state,
high-level propagation/output handling, codecs, multi-GPU execution and portable
packaging remain unfinished. See `VIDEO_INTEGRATION.md` for the exact contracts.

## Execute neural reconditioning and release consolidated previews

Added `sam3/video_recondition.h` to apply prepared corrections to actual tracker
sessions and preflight their memories. SAM3 follows candidate order and preflights
affected states after each candidate. SAM3.1 batches into the first containing
state, records all IDs in changed states as affected, then preflights states
sharing those IDs in session order. The executor borrows sessions/model cores;
it does not clone models or silently drop valid candidates. Multi-state execution
is not atomic: a later failure does not roll back already committed states.

`Sam31TrackingSession::recondition_masks` now applies the existing dynamic frame
helper to current-frame masks, scores and multiplexed pointers while preserving
bucket layout. It rejects unknown/duplicate IDs and missing frames and retains
individual-edit rollback. A native auxiliary 1008 high-mask grid is reconstructed
after the source-equivalent deferred 1152 brush update; preflight rebuilds memory
from consolidated masks. Resident, offloaded and paged histories are supported.

Repeated corrections exposed an existing session bug: already-consolidated brush
previews remained in `video_edits` and could be reapplied/suppressed by later edits
on the same frame. Preflight now releases these temporary previews. Original
annotations and encoded history are retained. The failing repeated-correction
fixture now matches source, including edits before and after preflight.

Actual-weight neural comparisons use full-grid synthetic cached features, not
a coherent detector/vision/video sequence. SAM3.1 compares actual demo correction,
preflight, subsequent forward/reverse propagation and stored state: 1,905 exact
comparisons in 24 CUDA workflows over three modes, plus 214 in two CPU FP32
workflows. SAM3 compares the original video-base `_recondition_masklets` and full
session snapshots: 1,116 across three CUDA modes plus 372 on CPU FP32. Total:
3,607 exact neural output/state comparisons, with no tolerance adjustment.

The original SAM3.1 offloaded correction scatter fails on CPU/CUDA device mismatch,
so those source comparisons keep state on the compute device. Native resident,
CPU-offloaded and disk-paged workflows compare separately: 1,083 exact tensor
comparisons across three modes, unchanged old conditioning history and temporary
archive cleanup. Existing source adaptations for CPU device transfers and FP32
compressed memory are recorded in the reports; no claim of unmodified upstream
CPU support is made. The older intermittent full-model CPU runtime issue remains
unresolved despite the passing CPU comparisons here.

The dynamic session invariant suite passes, including 44 cancellation/resume
comparisons, 18-point retention, insertion/removal and old-bucket preservation.
Standalone SAM3 and SAM3.1 probes run with PATH=/nonexistent; the latter also
checks paged history, invalid corrections and shared-ID preflight across two
sessions. CTest passes 19/19 CUDA-enabled and 11/11 custom-CUDA-disabled checks.
No GitHub Actions or Windows/Turing execution was used. Python-free linkage and
existing sm_75 cubins remain; portable distribution is still unfinished.

Code/reports are pushed to `codex/native-onboarding`; headers, binaries and logs
are private in `native-foundation/recondition-linux-cuda13`. No weights or full
model variants were added. Global occlusion-adjusted memory updates, coordinated
detector insertion/removal, text/visual prompt state, caches/user actions and the
complete high-level video coordinator remain. Codecs, multi-GPU execution,
portable packaging and end-to-end quality/performance validation also remain.
The next source boundary is `_tracker_update_memories` after correction/occlusion.

## Apply global suppression to tracker memory

Added `sam3/video_memory.h` and session/frame memory-replacement APIs. Global
masks resize directly to the 1152 memory grid; objects retaining less than 0.3 of
their positive area after pixelwise competition are suppressed. Surviving masks
retain their logits and may overlap. Source singleton/warmup differences and
argmax ties are preserved. Boolean winners avoid the source floating overlap
image temporary; no detection/object cap or lower-precision count is introduced.

Memory-only +10/-10 proxy scores do not replace predicted masks/scores. SAM3
updates stored memory and per-object slices. SAM3.1 also saves image features and
optionally projects newly suppressed pointers with the existing no-object weights.
Explicit global-ID mapping preserves local state order, including reversed IDs
and backfilling. Source neural comparisons use sorted coherent rank-zero IDs;
native mapping avoids the source dynamic branch's order/rank-offset assumptions.
The caller must still gather global inputs for multi-GPU use; communication is
not implemented by these helpers.

SAM3.1 retains actual memory input masks/proxy scores separately from predictions,
so later bucket changes re-encode the same suppression decisions. The new fields
participate in offload, paging, global-ID remapping and CRC-protected archives.
Obsolete overrides clear when normal edits invalidate memory or preflight encodes
new consolidated memory. They add runtime history storage, not model variants.
Each session stages updates; the overall list of sessions is not one transaction.

Validation uses actual original policies and high-level `_tracker_update_memories`
methods. There are 388 exact policy tensor comparisons across CPU/CUDA and three
precisions, including 201 objects, plus four mapping and three rejection cases.
Actual-weight session comparisons total 2,390 tensors across CPU FP32 and CUDA
FP32/FP16/BF16-reference, including both pointer settings, subsequent forward/
reverse propagation and repeated-correction regression. Full-grid cached features
are synthetic; this does not prove coherent vision/detector/video accuracy.

Resident/offloaded/paged workflows add 1,350 exact comparisons across three CUDA
modes, including object insertion/history rebuild after memory replacement. The
rebuilt bucket is checked with the actual source neural encoder using effective
masks/proxies. Predicted masks/scores stay unchanged, historical new objects are
absent, preflight clears old overrides and temporary archives are cleaned up.

CTest passes 21/21 CUDA-enabled and 12/12 custom-CUDA-disabled checks. Standalone
actual-weight session tools run with PATH=/nonexistent, including paged state and
reversed global IDs. Python-free linkage and existing sm_75 cubins remain. The
previous intermittent full-model CPU failure is still open. No GitHub Actions or
Windows/Turing execution was used; portable distribution is still unfinished.

Code/reports are pushed to `codex/native-onboarding`; private binaries/headers/logs
are in `native-foundation/video-memory-linux-cuda13`. No weights/model variants
were added. The full coordinator still needs detection insertion/removal,
text/visual prompt and cache/user-action management, phase integration and output
assembly. Codec, multi-GPU and end-to-end quality/performance work also remain.

## Coordinate detector births and object removals

Added `sam3/video_objects.h`: owning collections, shared-core/session factories,
source mask preparation at 1152, SAM3 birth-batch states and SAM3.1 stable best-fit
placement. First-state/new-state grouping policies are also available. Counts
are not capped; standalone validation includes a 17-object group. Empty additions
are no-ops and duplicate/existing new IDs are rejected. Fresh states are published
only after preflight; in-place operations retain per-session rollback boundaries.

SAM3 removal keeps the source ID/state iteration order. SAM3.1 now exposes batch
removal with upfront strict validation, duplicate normalization and one history
remap; empty collection entries are destroyed. Native stable-slot/dense-history
semantics remain explicit, not claimed equivalent to source packed-history bugs.
Per-mask layout updates during insertion remain an optimization opportunity.

Original high-level add/remove calls and forward/reverse tracking match exactly
for 1,048 actual-weight output/state tensors over both models and CPU FP32/CUDA
FP32/FP16/BF16-reference. Best-fit, removal and storage checks add 974 comparisons,
including batch/sequential removal equivalence on CUDA. Policy/preprocessing
fixtures run 489 checks per device. Full-grid projected features are synthetic;
coherent real-video accuracy remains unproven. Earlier CPU instability is open.

CTest passes 23 CUDA-enabled and 13 custom-CUDA-disabled checks. Standalone tools
run with Python absent from PATH, including paged state. Linkage excludes
libpython/libtorch_python and sm_75 kernels are present. No Actions or physical
Windows/Turing checks were used. Development binaries are not a portable SDK.

Code/reports are on `codex/native-onboarding`; binaries/headers/logs are saved in
private `native-foundation/video-objects-linux-cuda13`. No weights/model variants
were added. Next: global ID/score/confirmation metadata and orchestration of these
local-state APIs with detector/association/hotstart/occlusion/reconditioning/
memory phases. Prompt/cache/user-action state, output assembly, codecs, multi-GPU,
portable packaging and end-to-end quality/performance work remain.

## Compose per-frame update planning and execution

Added `sam3/video_update.h`: persistent rank/ID/score/confirmation metadata,
immutable planning across existing association/hotstart/reconditioning/occlusion
components, local neural correction-memory-birth-removal execution, and raw video
mask assembly. New IDs are monotonic and uncapped; removed object scores remain
-10000 while the final per-frame sigmoid write follows source overwrite order.
SAM3.1 device metadata rows explicitly follow rank-concatenated IDs after
compaction/extension, correcting a source multi-rank ordering assumption.

Original planning and raw-output methods match for 44,868 exact numerical
comparisons in 96 workflows / 768 frames: both models, CPU/CUDA, three precisions,
both directions, confirmation, correction, occlusion, boundary/IoM and disabled
warmup. Neural calls in this comparison are recording/no-op fixtures. The source
CPU empty-batch connected-components failure is adapted with empty label/count
outputs only. This is not complete neural-video parity.

Separate actual-weight standalone tools execute the new planner/executor/output
path with Python absent from PATH, controlled projected features/global masks,
real neural memory encoding, births/removals and SAM3.1 paged state. Native unit
checks cover prior-state immutability, two-rank ID/device-row alignment, score
ordering, compaction and overflow rejection. CTest passes 25 CUDA-enabled and
14 custom-CUDA-disabled checks. Python-free linkage and sm_75 cubins remain;
Windows/Turing runtime is untested and no GitHub Actions were used.

Code/reports are on `codex/native-onboarding`; private headers/binaries/logs are
in `native-foundation/video-update-linux-cuda13`. No weights/model variants were
added. Warmup metadata lifecycle, coherent detector/visual caches, prompt and
user-action state, predictor temporal/output filtering, complete high-level C ABI,
codecs, multi-GPU, portable SDK packaging and end-to-end accuracy/performance are
still open. Earlier intermittent CPU instability is unresolved. Continue with
coherent feature/detection/propagation integration rather than claiming the full
predictor is finished.

## Connect real video frames to detection and tracking

Added a shared full-1008 vision frame encoder feeding detection and both tracking
paths, video joint-presence scoring, and source-specific uncapped NMS/query
filtering. The same modules are shared by tracking sessions; no whole-model
variants or extra weight files were created. Both standard SAM3.1 batched NMS
and its alternate perflib quirks are preserved as explicit modes. Policy math
defaults to FP32 independently of neural execution precision.

Source comparison passes 522 detector-filter checks and 12 real-frame feature/
detector cases (both models, two frames, three precisions, two runtime text/geometry
prompts, all 200 queries). Every compared feature and detector value is exact.
A standalone C++ integration probe runs native text/vision/detection/propagation/
update/memory/birth/removal/raw-output assembly on three real frames with Python
absent from PATH. Both models track four persons and use exactly one trunk call
per frame. The probe exposes one text prompt while the detection API accepts
batches. Final temporal filtering and source coherent tracking-mask parity are
not established by this execution check.

CTest passes 25 CUDA-enabled and 14 custom-CUDA-disabled checks. Python-free
linkage and sm_75 compilation evidence remain; no Actions or Windows/Turing
execution was used. A source initialization crash was localized with GDB to MKL
parallel erfinv; one-thread comparison initialization completes, but general CPU
stability is still unresolved. Development binaries are not a portable SDK.

Code/reports are on `codex/native-onboarding`; private artifacts are in
`native-foundation/video-frame-linux-cuda13`, with raw fixture outputs separately
in `video-pipeline-fixture`. Continue with coherent source video comparisons,
warmup/temporal filtering, prompt and user-action management, complete C ABI,
codecs, multi-GPU and portable packaging. The overall goal remains incomplete.

## Validate coherent source video and correct integration inputs

The first source raw-frame comparison found a real integration mismatch: the
upper-level image-folder loader uses bilinear/F16 normalization, while the
low-level tracker uses bicubic/F32. Added a distinct native upper-level preprocessing
API, retained low-level behavior, and matched Pillow's extreme-tall pass ordering.
Eleven upper-level and 63 low-level resize/normalization checks pass. Shared
RGB-to-feature/detector comparison passes all 12 model/precision/frame cases exactly.

The standalone probe now uses SAM3 score-based memory selection and the source
auxiliary text batch slots. SAM3 BF16 raw outputs and low tracking values match
exactly over three actual frames. FP16 vs adapted source has at most three boundary
pixel differences per object and minimum mask IoU 0.9999112; FP16 vs original BF16
is measured separately. This is a small fixture, not a completed quality benchmark.

SAM3.1's matched batch/RoPE diagnostic is exact for the first two frames, then
shows a memory-update divergence. Intermediate masks/scores, image features,
positions and pointers match, but encoded memory differs. Captured source/native
state tensors and standalone memory replay narrow the next investigation. Do not
claim that isolated memory tests resolve this end-to-end difference. Default source
batching/RoPE comparisons and failing-state traces are retained separately.

CTest remains 25 CUDA-enabled / 14 custom-CUDA-disabled. No Actions or physical
Windows/Turing tests were run. Code, reports and private diagnostic artifacts are
being persisted before continuing this investigation and temporal/prompt lifecycle
integration. General CPU stability and portable packaging remain open.

## Fix global memory view and extend continuous-video coverage

Identified and fixed the SAM3.1 global memory-update divergence: identical shared
image values had a different singleton batch stride than the source sequence-to-
BCHW view. A forced-stride replay reproduces every differing value, and the source
view removes the discrepancy. Initial/correction/global update paths now share
the view conversion. No data copy, precision reduction, feature limit or weights
were added.

The strengthened regression fails before the fix and passes afterward with 1,356
exact CUDA comparisons over three precisions and 452 CPU FP32 comparisons using
one thread. The initial four-thread CPU attempt hit the known unresolved runtime
crash. CTest passes 25/14. Three-frame matching-configuration SAM3.1 raw video
outputs/low values now match exactly; original builder defaults and native FP16
quality measurements are retained separately.

An 18-frame original neural comparison matches every raw mask/low value through
frame 16, then fails after periodic reconditioning at frame 17. Preserve this
failure: it is the next integration issue, not a reason to disable reconditioning
or relax the exact gate. The goal remains incomplete; after that transition,
continue predictor temporal/output filtering, prompt/user-action state, codecs,
complete C ABI, multi-GPU, portable packaging and quality/performance validation.

## Resolve periodic correction history mismatch; both raw pipelines pass 34 frames

Frame-16 memory, pointer, image and mask traces were all equal. The divergence
came from history classification: original SAM3.1 keeps corrections to already
tracked frames in non-conditioning history; the standalone probe inherited a
low-level default that promotes them. Configure the SAM3.1 probe explicitly with
`all_edits_conditioning=false`. Original SAM3 promotes corrections, so keep its
True setting. No neural change, weight variant or capability reduction is involved.

The previously failing 18-frame strict comparison passes. Fresh actual-source
runs of both SAM3 and SAM3.1 pass all masks and pre-plan tracker values for 34
consecutive frames, including periodic corrections at 16/32 and their following
frames. Scores tolerate 1e-8 serialization rounding. BF16, no TF32; SAM3.1 uses
matching batch-one grounding/complex RoPE. Each standalone process ran with
`PATH=/nonexistent`, all 200 queries, four people, one trunk evaluation per frame.
Both CUDA-enabled and custom-CUDA-disabled probe targets build. This milestone
does not establish dataset-wide quality, CPU stability, physical Turing/Windows
support or complete predictor parity. No Actions were used.

The exact reports, state diagnosis and historical failure are retained. Private
before/after outputs and source tensors accompany the updated development probe
snapshot. Continue with predictor temporal buffering, confirmation/final output
filtering, prompt/user-action state, codecs, full C ABI, multi-GPU, portable SDK
and quality/performance validation; the overall goal remains active.

## Integrate video output buffering and final masks

Added `video_output.h`: delayed output with removal snapshots, forward/reverse
confirmation lookahead, source batch-emission timing, terminal flush, cancellation
and reset. Final processing preserves source ordering of hidden/empty filtering,
boxes, overlap resolution and optional centers. Cached masks retain empty and
pre-overlap values; frame statistics pass through. There is no object count limit.

The coherent standalone probe now emits final results in addition to raw traces.
Focused original-source generator/postprocessor comparisons pass 96 workflows and
13,144 exact checks each on CUDA/perflib, CPU/perflib and CUDA/torchvision. CTest
passes 27/15, including 257 objects, snapshot mutation isolation and reset/cancel.
Fresh real-neural 34-frame SAM3 and SAM3.1 runs match every final mask, ID, score,
box and emission time, with the previous raw parity retained. BF16/no TF32;
SAM3.1 neural batch-one/complex-RoPE configuration is explicitly recorded, while
output batching uses source default 16. Python-free native processes ran with
all 200 queries and one shared trunk evaluation per frame.

See [video output details](VIDEO_OUTPUT.md) for API behavior, evidence and limits.
No Actions or physical Windows/Turing tests were used. This is not full action
routing/prompt lifecycle, full C ABI, codec/multi-GPU or portable SDK completion.
Those integrations, CPU runtime stability and quality/performance work remain.

## Connect action routing, selected-object propagation and cached outputs

Added `video_interaction.h`: source action-history route selection, refined-ID
recognition, propagation bounds with reverse-start exclusion, displayed-frame
cache snapshots, partial merge/fetch/reset and local neural partial execution.
The executor runs sessions containing selected IDs with memory encoding but only
merges requested IDs into prior cached masks. Existing weights/cores are shared.
Semantic prompt replacement and actual instance edit orchestration remain open.

Actual source routing/range and merge/cache methods match on CPU and CUDA: each
run covers 998 histories, two rejected malformed cancellation sequences and
5,106 exact merge/output/score checks across three precisions and up to257objects.
CTest passes28/16. Source quirks, including missing SAM3.1 cache behavior and raw
partial-path scores, are documented in [VIDEO_INTERACTION.md](VIDEO_INTERACTION.md).

Both real-neural coherent probes retain exact34-frame forward raw/final outputs,
then match the original high-level predictor on reverse partial frames17,16,15
for one requested ID while preserving three other cached objects. BF16/noTF32;
SAM3.1 batch-one grounding/complexRoPE, outputbatch16. Native processes run with
PATH=/nonexistent and no Python linkage. This regression records a refine action
but injects no new point/mask edit; it does not prove full interactivity complete.
Code/reports and private runtime/reference snapshots are persisted. No Actions or
physical Windows/Turing tests. Next integrate actual point/mask edits and semantic
text/geometry/visual prompt lifecycle, then full C ABI, codecs, multi-GPU, portable
SDK, broad quality/performance and unresolved CPU runtime stability.

## Integrate actual SAM3 point/mask instance edits

Added `video_edit.h` high-level SAM3 point/mask edit and user removal helpers.
They locate/create shared-core sessions, update IDs/scores/actions/cache, clear
point-nearby detector conditioning, retain authoritative mask observations and
support stateless first refinement. Basic validation precedes stateless removal.
The regression confirms invalid frame input leaves IDs/actions unchanged.

After34real-neural frames, the original high-level SAM3 predictor and standalone
C++ match all10output sets from existing-point edit, new/existing exact masks,
five-frame partial propagation, user removal/fetch and stateless point refinement.
All34raw/final pre-edit frames remain exact. BF16/noTF32; native PATH=/nonexistent,
all200queries and no object limit. Latest rerun also matches saved actual reference
tensors; cached comparison now supports final/partial/edit outputs. CTest28/16.

See [VIDEO_EDIT.md](VIDEO_EDIT.md). SAM3.1 needs separate singleton extraction and
history/input consolidation work; these helpers do not claim that integration.
Semantic prompt lifecycle, full C ABI, codecs, multi-GPU, portable SDK, CPU stability
and broad quality/performance remain. No Actions or physical Windows/Turing tests.

## Integrate SAM3.1 first point refinement and preserve dense history

Added grouped-object singleton extraction, mask-only input cleanup without history
loss, high-level point edits and user removal with updated bucket workloads. Cores
and features remain shared. Resident/offloaded/paged extraction preserves masks,
slot pointers and source-bucket memories;105exact tensor comparisons pass.

The real34-frame/four-person fixture exposed an original extraction bug: slot
demux on dense spatial memory throws and leaves historical memories unset. Native
re-encodes changed buckets. Point preview/frame18 match the unmodified source;
frames19/20 differ by140,939/129,160 mask pixels (IDs/probabilities still match).
An explicitly labeled test-only original-encoder adapter reconstructs34singleton
memories; all four point/propagation output sets then match exactly. Reports keep
unmodified and corrected references separate. This is not a quality-improvement
claim. Pre-edit raw/final34frames remain exact, as do all10SAM3 edit regressions.

CTest28/16 passes; native PATH=/nonexistent, no Python linkage, sm75cubins present.
See [VIDEO_EDIT.md](VIDEO_EDIT.md). No Actions or Windows/Turing hardware tests.
Further SAM3.1 editing combinations, semantic prompt lifecycle, full C ABI,
codecs/multi-GPU, portable SDK, CPU stability and broad quality/performance remain.

## Preserve point conditioning during repeated SAM3.1 edits

Point edits now establish/refresh conditioning even when periodic detector
corrections stay non-conditioning. Unknown/already removed SAM3.1 user IDs are
accepted while cached masks are forgotten and removal actions are recorded.
Dedicated full-grid point/memory invariants and repeated-removal cache tests
cover both changes;153adapted original low-level tensor comparisons stay exact.

Extended the real34-frame regression to15edit/propagation/fetch checkpoints,
including repeated points, reverse propagation, new ID, removal/repeated removal,
and stateless refinement. Original default repeated clicks assert; its stateless
removal call omits a required argument. Explicit reference options record those
repairs, dense extraction reconstruction, stale edit-memory replacement and
preservation of untouched singleton history. No native predictions replace the
original neural outputs. Full34raw/final outputs remain exact;14of15post-edit
output sets match, while reverse frame17still differs by120binary-mask pixels.
IDs, probabilities and boxes remain exact. This discrepancy is unresolved and
reports retain exact=false. See [VIDEO_EDIT_SEQUENCE.md](VIDEO_EDIT_SEQUENCE.md).
No Actions/physical Windows/Turing tests. Full predictor integration/packaging and
quality/performance work remain; these results do not establish completion.

## Own semantic prompt lifecycle in the C++ video API

Added `VideoPredictor`, which owns shared neural modules, frame features, prompts,
metadata, sessions, action routing, output caching and scheduling for one video.
Semantic replacement/reset retains shared model modules; point/mask/removal/fetch
helpers are integrated. Point-only SAM3.1 initializes a cache explicitly. The
standalone probe also checks callback cancellation, requested center shapes and
invalid semantic input preserving state. Prompt/object/query limits are unchanged.

Revisiting the initial SAM3.1 prompt exposed loss of the sole conditioning frame
on detector correction. Mask edits/reconditioning now retain an existing condition,
while new tracked-frame corrections stay non-conditioning. Extended full-grid
neural invariants pass; CTest28/16 passes with and without custom CUDA operators.

SAM3 matches all11actual original semantic lifecycle outputs; latest standalone
results remain exact against saved references. SAM3.1 executes but is not exact:
original bounded batched propagation first raises an endpoint IndexError. An
explicit test-only forward-bound adapter permits comparison. Ten output sets keep
identical IDs/probabilities/boxes but differ in masks; the remaining box-track
frame displays no native object versus original ID0. These discrepancies and the
prior120-pixel reverse-edit difference remain unresolved. See
[VIDEO_PREDICTOR.md](VIDEO_PREDICTOR.md) and the non-exact JSON reports.

This remains a development API. Full video C ABI, SAM3.1 high-level mask editing,
image-only fallback, codecs, multi-GPU transport, portable SDK/CPU stability and
broad quality/performance remain. No Actions or Windows/Turing physical tests.
The previous34-frame SAM3.1 raw/final baseline remains exact after this change.

## Correct SAM3.1 predictor visibility and reference precision

The owning predictor and coherent probe now filter only the original published
host suppression IDs. SAM3.1 GPU hotstart computes an additional suppression
candidate but its original planning path does not publish/use it for display.
Applying it in native orchestration wrongly hid a tracked box object. The fixed
frame retains ID0 with23058mask pixels even with a native keep-alive counter of0.

The new semantic reference also now disables TF32 AFTER construction, records
both effective flags and asserts them before frame inference. The original wrapper
had unconditionally re-enabled TF32 after the old test's initial configuration.
Decoder traces localized the first divergence to the FP32 FFN. The previous claim
that both sides had TF32 disabled was incorrect for this SAM3.1 semantic test;
historical reports now carry a precision audit. Earlier pipeline tests already
configured TF32 after construction and are unaffected by this test-driver error.

All11SAM3.1 semantic lifecycle outputs now match exactly, including emission
timing, versus the recorded forward-bound-adapted source with TF32 disabled.
Primary comparison does not repair source batched geometry. Native honors stored
geometry while source drops it on that path; counters differ despite identical
outputs in this short fixture. An optional explicit geometry reference adapter
is retained for investigation. Full state/long box-trajectory equivalence and the
prior120-pixel reverse discrepancy remain unproven. The overall goal is not complete.

The extended box fixture (six propagated frames,15total checkpoints) also matches
exactly with both endpoint and stored-geometry source adapters explicitly enabled.
Controlled replay with TF32 enabled matches84recorded decoder-layer tensors;
disabling it reproduces the earlier FFN divergence. CTest28/16 and the existing
34-frame raw/final SAM3.1 baseline pass after the output policy correction.

## Explicit image mode and SAM3.1 high-level mask integration

The C++ owner now distinguishes an image from a one-frame video. SAM3.1 uses its
source image birth threshold of 0.5 (configurable), retaining 0.65 for video.
Actual food/image detection returns four objects with scores between 0.515 and
0.617; the same image as a one-frame video returns none. Both cases and the person
image match unmodified original previews exactly in IDs, scores, boxes and masks.
All 200 queries and full model dimensions execute with shared modular weights.

The owner retains image mask baselines across point edits, restores them on empty
points, and supports authoritative SAM3.1 mask creation/replacement. Removal and
reset clear retained baselines. Stateless initial restoration, repeated clearing,
mask-point-mask restoration and semantic reset pass without Python on PATH. The
restored input matches the source detector input exactly. Only one trunk encode is
needed before reset. Display still applies source overlap arbitration.

The original empty-point image route raises AttributeError because its tracker
has no add_new_mask method, both directly and after a real click. This is recorded
without a reference adapter; native restoration is an invariant-checked repair of
intended behavior, not parity with a nonexistent original output.

Existing SAM3/SAM3.1 semantic video cases remain exact (11 each against retained
references, with the earlier source configuration/adapters still recorded). New
mask-only objects propagate through three frames in both models. CTest 28/16
passes. See IMAGE_PREDICTOR.md and image-predictor-validation.json. Owning C ABI,
codecs, multi-GPU transport, portable SDK/CPU stability and broad quality/performance
remain; the 120-pixel reverse-edit discrepancy is unresolved. No Actions or physical
Windows/Turing tests were used. The overall goal remains active.
