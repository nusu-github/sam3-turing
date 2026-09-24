# Native components

Design notes for the building blocks under `native/include/sam3/`, from primitives
up to the SAM3/SAM3.1 tracking sessions. The high-level video predictor that
composes them is described in [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md) and
[VIDEO_PREDICTOR.md](VIDEO_PREDICTOR.md); the development probes that exercise each
component are listed in [native/README.md](../../native/README.md#development-probes).

Common rules: every component accepts runtime prompt, object, point and batch
counts without caps; model modules are loaded from the shared weight store
without per-use copies; CPU and CUDA paths share one C++ API and honor the
caller's current CUDA stream and autocast state.

## Primitives (`ops.h`, `roi_align.h`, `rotary.h`, `rotary_pair.h`)

- Little-endian mask packing/unpacking with independently padded rows and
  noncontiguous input; chunked bilinear resize + sigmoid (+ packing) that keeps
  ATen's dtype and sigmoid rounding while `chunk_size` bounds temporaries.
- Stable score-ordered generic NMS with no detection cap (materializes an N×N
  boolean matrix).
- 8-connected component labeling and sizes over equal-valued nonzero pixels,
  root-index labels, 64-bit labels/counts (`[B,H,W]` or `[B,1,H,W]`).
- Euclidean distance transform returning float32 distances, with the source's
  finite `1e9` for masks without a zero pixel.
- ROIAlign for CPU/CUDA float32/float64/float16 following torchvision's autocast
  policy; the sampling code is adapted under the torchvision BSD license
  (`native/third_party/torchvision`), without needing torchvision at runtime.
- Rotary embeddings with a fused CUDA path ([PERFORMANCE.md](PERFORMANCE.md#rotary-embedding)).

A development-only dispatcher registration lets the parity tests call these
through `torch.ops.load_library`; the library itself has no Python ABI.

## Weights and archives (`weights.h`, `tensor_archive.h`)

`WeightStore` reads the modular v1 store ([WEIGHTS.md](WEIGHTS.md)).
`TensorArchive` stores tensors with dtype, shape, strides (including
channels-last and expanded tensors) and CRC, using C++17 filesystem/streams and
zlib; it backs history paging and the displayed-mask cache.

## Text (`tokenizer.h`, `text_encoder.h`)

`Tokenizer` accepts arbitrary UTF-8, applies the source ftfy 6.1.1 cleaning
(including mojibake repair), double HTML unescaping, whitespace/lowercase
normalization and the VE byte-level BPE. `encode` returns unframed IDs;
`tokenize` batches with context 32 and the original start/end, padding and
truncation rules. Both models share the gzip vocabulary, read with zlib.

The frozen character tables in `src/tokenizer_tables.h` come from
`tools/generate_tokenizer_tables.py` (CPython Unicode 15.0.0, ftfy 6.1.1, regex
2025.11.3), run only during development. Positive and negative regex property
branches are frozen separately because case-insensitive Unicode properties are
not simple complements in the source engine. ICU must be 72–74 (74.2 validated,
Unicode 15.x); revalidate normalization and lowercasing before changing it.
Notices: `third_party/ftfy`, `third_party/python`, `third_party/icu`.

`TextEncoder` keeps its language module loaded, runs all 24 layers on batched
token tensors up to context 32, and returns the padding mask, resized language
memory and input embeddings.

## Vision (`vision_encoder.h`, `preprocess.h`, `vision_fusion.h`, `vision_position.h`)

`VisionEncoder` runs the 32-block trunk and any requested SAM3 dual /
SAM3.1 tri-neck features (`convs`, `sam2_convs`, `interactive_convs`,
`propagation_convs`). Inputs are normalized `[B,3,1008,1008]`; `preprocess_rgb`
applies the image-processor resize and normalization (move raw pixels to the
execution device first to match that device's torchvision rounding). Modes:
`fp32`, `fp16`, and `bf16_reference`, which reproduces the original BF16-forcing
MLP for comparisons on capable hardware and is not a Turing mode. Vision calls
restore the caller's autocast state. An overload selects which position maps to
generate; Half compute storage is described in [PERFORMANCE.md](PERFORMANCE.md).

The private files `src/vision_experiments.h`, `vision_quantization.h`,
`vision_calibration.h` and `vision_qkv_observer.h` hold the opt-in INT8/INT4
research paths selected by environment variables
([native/README.md](../../native/README.md#experiment-switches)); unset variables
keep the exact FP16 arithmetic.

## Image grounding (`geometry_encoder.h`, `detector.h`, `detection_heads.h`, `grounding.h`, `image_results.h`)

- `GeometryEncoder`: point/box prompt encoding with image pooling, label and
  position embeddings, right-padded concatenation, CLS and all three layers;
  arbitrary prompt counts, labels, padding and empty sequences.
- `DetectorEncoder` / `DetectorDecoder`: the six-layer fusion encoder and
  six-layer decoder with all 200 queries, the presence token, relative box bias,
  iterative box refinement and all layer outputs. An optional trace records
  intermediates; leave it null in inference.
- `GroundingDetector` connects geometry, fusion, decoding, dot-product scoring,
  box refinement, pixel decoding, instance and semantic masks for varying text,
  point/box and visual prompts, image/text mappings and previous-mask features.
  Vision and text modules stay separate so callers can release or share them.
- `postprocess_image` applies the confidence rule and restores boxes and masks to
  each image's original size, returning probabilities, binary masks and original
  query indices with no count cap. For original SAM3 image behavior use
  `joint_scores=false` in grounding and `combine_presence=true` in
  postprocessing; the video/SAM3.1 path uses the opposite pair to avoid double
  presence multiplication.

## Interactive image (`interactive_prompt.h`, `interactive_decoder.h`, `interactive_image.h`)

`InteractivePromptEncoder` produces point/box sparse embeddings, mask or no-mask
dense embeddings and the learned Fourier position grid; masks are expected at the
288×288 prompt size. `InteractiveMaskDecoder` runs both two-way layers and all
four mask tokens, returns three candidates and the stability-based single-mask
fallback, keeps the source object-pointer token even when stability selects a
different mask, and keeps SAM3/SAM3.1 IoU activation differences.
`project_pyramid` projects the two high-resolution maps once for reuse.

`InteractiveImageSession` connects real image features, pixel or normalized
points and boxes, repeated prompts, previous 288×288 mask logits and
original-size postprocessing (thresholds, hole/sprinkle areas, single/multimask).
It keeps projected high-resolution features so the vision trunk can be released
after `set_image`. `set_image` and `set_images` keep the source's different
single-image and batch tensor layouts (this matters for exact convolutions);
`set_features` accepts a shared vision pyramid so grounding and interaction reuse
one trunk evaluation. Image predictions do not apply the video object-presence
gate, as in the source.

## SAM3 tracking

- **`VideoInteractiveHeads`** (`video_heads.h`): SAM3 video heads and SAM3.1's
  per-object interactive path: missing-point padding, antialiased resizing of
  arbitrary-size mask prompts, object-presence gating, best-IoU selection and
  object-pointer projection (SAM3 learned absent pointer, SAM3.1 linear
  transform). `use_mask_as_output` keeps supplied masks and computes pointers via
  the original downsample/decoder path. Direct-mask output sizes follow the
  source formulas (SAM3 `input_size // 14 * 4`, SAM3.1 `input_size // 4`).
- **`MaskMemoryEncoder`** (`memory_encoder.h`): memory downsampler, two ConvNeXt
  fusion blocks, projection and cached positions. SAM3 produces 64-channel memory
  per object; SAM3.1 256-channel memory per multiplex group of 16 mask and 16
  conditioning channels (groups grow with the object count). `encode_frame` adds
  the tracker's mask transforms, optional non-overlap constraint, SAM3 point-mask
  binarization and absent-object embeddings; SAM3.1 selection uses a matmul to
  keep the source's autocast rounding. One image feature can be shared across
  objects and staged on CPU.
- **`MemoryAttention`** (`memory_attention.h`): four temporal layers and final
  norm. SAM3 uses one head, 64-channel memory and ReLU; SAM3.1 eight heads,
  separate image/object streams, 256-channel memory and GELU. Axial complex RoPE
  repeats over spatial memories and excludes trailing pointer tokens. It lets
  LibTorch choose any precompiled SDPA backend (no Flash-only requirement) and
  keeps the caller's backend settings.
- **`Sam3MemoryConditioner`** (`temporal_memory.h`): frame selection, temporal
  positions, pointer assembly and memory attention. `TemporalState` keeps ordered
  conditioning and tracked collections (insertion order breaks equal-distance
  ties). Options keep the source conditioning limit/keep-first policy, stride,
  direction, score-based selection and pointer history. Effective-IoU scores keep
  their tensor dtype, because converting to double before thresholding can change
  half-precision decisions.
- **`Sam3TrackingFrame`** (`tracking_frame.h`): the frame host used by sessions.
  It chooses the direct-mask or memory-conditioned path, applies the source
  multimask policy, optionally encodes memory (`encode_memory=false` supports
  repeated edits before consolidation), offloads to CPU while keeping pointers
  and object logits on the device, and trims history by the source's
  score/offload rules, including reverse tracking.
- **`Sam3TrackingSession`** (`tracking_session.h`): the low-level interactive
  tracker state machine. Arbitrary accumulated points, boxes and masks, preview
  edits with previous logits, consolidation, forward/reverse propagation,
  clear/remove/reset. Normalized coordinates are multiplied by 1008. New IDs must
  be introduced before propagation starts (source low-level behavior); the
  high-level predictor handles dynamic objects. Pending edits are consolidated at
  preflight with source overlap constraints; memory is stored as BF16 (FP32
  execution expands it before attention). CPU offload waits for device-to-host
  copies before returning, because asynchronous copies corrupted previews.
  Frame dictionaries reproduce CPython 3.12's insertion order and integer-set
  traversal, which fixes conditioning order for sparse annotations without
  linking Python. `propagate` emits frames through a callback; returning false or
  calling the thread-safe `cancel()` stops after a consistent frame. Other methods
  must not run concurrently. Endpoints are inclusive (`max_steps=0` emits the
  start frame).
- **`Sam3TrackingVision`** (`tracking_vision.h`): real-frame input for sessions.
  `encode_rgb` reproduces the original synchronous JPEG-sequence preprocessing
  (Pillow byte bicubic, /255, mean/std 0.5) with `resize_tracking_rgb`, a portable
  C++ implementation of Pillow's fixed-point bicubic with FMA contraction
  disabled (notice in `third_party/pillow`). This differs from image-mode
  `preprocess_rgb` and from the high-level video loader
  ([VIDEO_PIPELINE.md](VIDEO_PIPELINE.md)). The trunk runs once per cached frame
  and is shared by all objects and edits.

## SAM3.1 multiplex tracking

- **`MultiplexState` / `MultiplexController`** (`multiplex.h`): object allocation
  into 16-wide buckets that grow with the object count, stable optional external
  IDs, renumbered dense internal indices, removed slots kept until their bucket is
  discarded, CPU-RNG shuffling and new-bucket preference. Mux/demux keep the
  source matmuls and autocast rounding. Invalid additions/removals leave the state
  unchanged (unlike the source's partial mutations); removing all objects
  invalidates the state.
- **`MultiplexMaskDecoder` / `MultiplexPropagationHeads`** (`multiplex_decoder.h`):
  16 slots per bucket with three mask tokens each; demux into object order,
  absent-object gating, resize to 1008, best-IoU selection and pointer projection,
  optional IoU attenuation by stability, suppression embeddings for live, padded
  and removed slots.
- **`MultiplexMemoryConditioner`** (`multiplex_temporal.h`): SAM3.1 temporal
  selection with separate image and object streams, future-conditioning pointers,
  unsigned pointer times and v2 spatial time embeddings (options expose the
  alternatives). Cleared memory falls back to current image features; legacy 5D
  per-slot memory is demuxed and cached back.
- **`Sam31TrackingFrame`** (`multiplex_frame.h`): joins the interactive head,
  propagation decoder, temporal selector and memory encoder for direct masks,
  point initialization, refinement, pure propagation and propagation with partial
  corrections (interactive rows replace the selected objects, broadcast into three
  candidates as in the source). `update_masks` implements current-frame mask
  insertion and reconditioning (append versus replace, deferred encoding, bucket
  growth); failed updates leave caller state unchanged.
- **`remap_multiplex_history`** (`multiplex_history.h`): remaps histories between
  layouts by global object ID. Unchanged buckets keep their pointers and memory
  exactly; dense `[buckets,256,72,72]` memory jointly encodes 16 objects and cannot
  be demuxed per object, so changed buckets require a `MultiplexHistoryRebuilder`
  that re-encodes from retained full-resolution masks, logits and image features.
  A missing rebuilder is rejected rather than dropping joint memory; an error in
  any frame leaves the whole history unchanged.
- **`Sam31TrackingSession`** (`multiplex_session.h`): arbitrary points, boxes
  (labels 2/3), individual and simultaneous mask prompts (`add_masks` applies the
  source's mutual brush suppression), midstream insertion, forward/reverse
  propagation, clear/remove/reset and atomic cancellation. IDs and slots stay
  stable across refinement; new point objects prefer new buckets while mask
  objects fill free slots. It uses the explicit dense-history rebuild above
  instead of the source demo's singleton extraction/merge, keeps full masks and
  image features for rebuilds, compresses spatial memory to BF16 and can offload
  state to CPU. Failed edits restore the previous state.
- **`Sam31TrackingVision`** (`multiplex_vision.h`): one trunk evaluation produces
  both the interactive and propagation necks, cached together per frame.

### Lossless history paging (`multiplex_storage.h`)

Setting `MultiplexSessionOptions::history_directory` pages frame payloads (masks,
image features, encoded memory, positions, pointers, scores) to a caller-owned
directory. Precision, object limits, prompts and memory selection are unchanged,
and everything stays available for later edits and reverse tracking. The
tracking core reads only the spatial and pointer streams chosen by the temporal
planner; a layout change processes one old frame at a time and writes replacement
archives before committing. Failures keep the prior edit state. Archives are
temporary process-owned caches (not resumable checkpoints), shared by state
copies and deleted after the last reference. Paging bounds retained frame payload
in RAM, not total process memory. `load_multiplex_frame` and
`load_selected_multiplex_history` inspect paged state; undefined payload fields
in a paged frame do not mean the history was discarded. SAM3's non-multiplex
session does not page.

## C ABI (`c_api.h`)

Opaque context, image, video, predictor and result handles over standard C types;
see [C_API.md](C_API.md) and [PREDICTOR_C_API.md](PREDICTOR_C_API.md).
