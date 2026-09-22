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
