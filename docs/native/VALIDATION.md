# Validation, source parity and known differences

How the native runtime is checked against the original Python implementation,
what currently matches, where it deliberately differs, and what is still open.

## How the runtime is checked

- **CTest** (`native/tests/*.cpp`, registered in `native/cmake/Development.cmake`)
  runs without weights or Python: operators, rotary/fusion kernels on default and
  nondefault streams, weights and archives, tokenizer, C ABI, multiplex state,
  video policies, output and cache components, experiment settings. The Windows
  RTX 2060 build passes 62/62.
- **Source parity scripts** (`native/tests/*.py`, development only) load the
  original modules or call the original methods with actual weights and compare
  every output tensor at zero tolerance unless stated. Each writes a JSON report;
  the recorded reports are in [evidence/](evidence/).
- **Standalone probes** (`native/tools/*`) run the C/C++ library with Python
  removed from `PATH`; the parity scripts compare their outputs.
- **Native regressions**: every optimization and refactor compares the new build
  byte-for-byte with the previous one on actual-weight feature tensors and the
  owning C/C++ lifecycle cases ([PERFORMANCE.md](PERFORMANCE.md)).

Exact comparisons against the source use a matching configuration: BF16-reference
mode, TF32 disabled **after** model construction (the SAM3.1 predictor
constructor enables TF32), and for SAM3.1 grounding batch 1 with complex RoPE
rather than the builder default batch 16 / real RoPE. FP16 comparisons replace the
source's vision MLP helper, which casts to BF16 (`sam3/perflib/fused.py`), with a
test-only FP16 adapter. CPU references redirect hard-coded `.cuda()` transfers and
recompute rotary caches on the CPU; CUDA FP32 and explicit-math references remove
SAM3.1's Flash-only attention context. Each report records the adaptations it
used.

| Area | Scripts |
|---|---|
| Primitives | `parity.py`, `upstream_parity.py`, `roi_align_parity.py`, `weight_parity.py`, `tokenizer_parity.py` |
| Image modules | `vision_parity.py`, `text_parity.py`, `geometry_parity.py`, `detector_encoder_parity.py`, `detector_decoder_parity.py`, `detection_heads_parity.py`, `image_results_parity.py`, `image_end_to_end_parity.py`, `image_probe_parity.py`, `grounding_fixture_parity.py` |
| Interactive image | `interactive_prompt_parity.py`, `interactive_decoder_parity.py`, `interactive_image_parity.py`, `interactive_image_probe_parity.py`, `interactive_image_batch_parity.py` |
| SAM3 tracking | `video_heads_parity.py`, `memory_encoder_parity.py`, `memory_attention_parity.py`, `temporal_memory_parity.py`, `tracking_frame_parity.py`, `tracking_session_parity.py`, `tracking_preprocess_parity.py`, `tracking_video_parity.py` |
| SAM3.1 multiplex | `multiplex_parity.py`, `multiplex_decoder_parity.py`, `multiplex_temporal_parity.py`, `multiplex_frame_parity.py`, `multiplex_update_parity.py`, `multiplex_history_parity.py`, `multiplex_history_encode_parity.py`, `multiplex_session_parity.py`, `multiplex_session_invariants.py`, `multiplex_storage_parity.py`, `multiplex_storage_video.py`, `multiplex_video_parity.py` |
| Video pipeline | see [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md#validation) |
| Owner and C ABI | `video_predictor_parity.py`, `image_predictor_parity.py`, `predictor_c_parity.py`, `predictor_c_sharing.py`, `c_api_parity.py`, `c_api_grounding_parity.py`, `c_api_batch_parity.py` |
| Media and preprocessing | `media_parity.py`, `media_seek.py`, `preprocess_parity.py`, `video_preprocess_parity.py`, `video_loader_preprocess_parity.py`, `video_file_preprocess_parity.py` |
| Multiple ranks | `video_collective_reference.py`, `test_video_collective_reference.py` |

## Current status

| Scope | Result |
|---|---|
| Image modules and grounding, both models, FP32/FP16/BF16 | Exact against the source modules |
| Interactive image and low-level SAM3/SAM3.1 tracking sessions | Exact (synthetic full-grid features and real frames) |
| Coherent video pipeline, 34 real frames, raw + final outputs, partial propagation | Exact for SAM3 and SAM3.1 (BF16 matching configuration) |
| SAM3 instance edits (10 outputs after 34 frames) | Exact |
| SAM3.1 edits and edit sequences | Exact against the source with the adapters below |
| Owning predictor semantic lifecycle (11 outputs per model; SAM3.1 box track 15) | Exact; SAM3.1 with the grounding-bound adapter |
| Image mode previews (3 cases incl. one-frame video) | Exact against the unmodified source |
| Native FP16 versus source FP16 | **Not exact** (below) |
| COCO-slice quality, native FP16 vs BF16 | Mask AP −0.17 (SAM3) / −0.47 (SAM3.1) points ([IMAGE_PRECISION_AUDIT.md](IMAGE_PRECISION_AUDIT.md)) |

These are finite fixtures on a Blackwell GPU. They do not establish dataset-wide
quality, every interactive sequence or arbitrary trajectories.

### FP16 difference from the source

On the four-frame semantic fixture (11 outputs), native FP16 differs from the
source's FP16 by 30 mask pixels (single rank) and from the official Python 2.10.0
wheel by 32 pixels; IDs, probabilities, boxes and emission timing agree. BF16 is
exact. The source and native runs use different binaries (the NVIDIA development
build or the official wheel versus standalone LibTorch, whose CPU/CUDA libraries
hash differently), and substituting the standalone core into Python failed to
import, so a same-binary comparison was not possible. No tolerance turns this
into a pass; the cause is open.

### Logical ranks

Native one-rank and two-rank runs differ slightly (BF16 133 pixels, FP16 26 on
the four-frame fixture). Replaying the original predictor with its neural
sessions partitioned into the same two logical ranks reproduces the BF16
two-rank outputs exactly (all 11 outputs); adding only an empty peer to the
original collection does not. Session partitioning and global object order change
together here, so their individual effects are not separated. The replay adapter
(`video_collective_reference.py`) executes the original propagation, births,
removals and memory updates per rank on one GPU; process dispatch and transport
are adapted, no neural phase is mocked. Run it through
`video_predictor_parity.py --collective-partition-ranks 2` (or
`--collective-empty-peers 1` for the collection-only replay) with
`--require-exact`. Evidence:
[video-collective-reference-validation.json](evidence/video-collective-reference-validation.json).

## Deliberate differences from the source

| Behavior | Source | Native |
|---|---|---|
| SAM3.1 object extraction to a singleton | Demuxes dense bucket memory as slot data, fails, and sets it to `None` (frames 0/16/32 lose spatial memory on the real fixture) | Re-encodes the singleton memory from retained inputs ([VIDEO_EDIT.md](VIDEO_EDIT.md)) |
| SAM3.1 dense history after layout changes | Legacy object-axis slicing | Explicit per-bucket rebuild with retained masks, logits and image features |
| SAM3.1 image empty-point restoration | Calls a missing `tracker.add_new_mask` (`AttributeError`) | Re-feeds the retained detector mask ([VIDEO_PREDICTOR.md](VIDEO_PREDICTOR.md#image-mode)) |
| Point-only SAM3.1 object without a cached frame | Merge drops it | Seeds an empty cache entry so it is returned |
| Association/occlusion arithmetic under FP16 autocast | Overflows for full 288×288 masks | FP32 by default; source modes available for comparisons |
| Invalid inputs (malformed histories, unknown candidate IDs, empty perflib NMS) | Undefined or failing behavior | Descriptive errors or empty results |

### Reference adapters

The original high-level SAM3.1 sequence cannot run every operation unmodified.
The comparison driver keeps each repair opt-in, records it in the report and
never substitutes native tensors:

| Flag | Repair |
|---|---|
| `--sam31-inclusive-grounding-bound` | The generator includes `start+max_steps` but the batched detector excludes it and raises `IndexError`; add one to the detector bound only |
| `--sam31-preserve-batched-geometry` | Pass stored box geometry to the batched prompt builder (used for the extended box-track fixture) |
| `--sam31-rebuild-extracted-memory` | Rebuild singleton memory lost by extraction with the original encoder |
| `--sam31-enable-repeat-refinement` | Set `iter_use_prev_mask_pred=true`; the default fails an assertion on the second edit of a frame |
| `--sam31-default-remove-frame` | Supply the `frame_idx=None` argument the internal stateless removal call omits |
| `--sam31-refresh-refined-memory` | Keep the newly encoded memory when an older conditioning entry and a new non-conditioning edit collide |
| `--sam31-preserve-singleton-history` | Avoid re-muxing untouched singleton history (which also rounds F32 positions to BF16) |
| `--sam31-refresh-refined-pointer` | Keep `obj_ptr` from the newer non-conditioning entry (see below) |

## Resolved investigations

- **SAM3.1 memory stride (fixed in `6b72b1e`).** The global memory update passed
  raw BCHW image features whose singleton batch stride differed from the source's
  sequence-to-BCHW view (`[1327104,1,18432,256]` versus `[256,1,18432,256]`).
  Values were identical, but the encoder produced 387,108 differing memory values.
  All three memory paths now share the source view.
- **Correction history category (fixed in `bb86f98`).** After frame-16
  reconditioning, all tensors matched but the native probe had promoted the
  correction into conditioning history. SAM3.1 defaults
  `add_all_frames_to_correct_as_cond=False`, SAM3 sets it True; the probe now
  selects the SAM3.1 setting, and the 18- and 34-frame comparisons are exact.
- **TF32 in SAM3.1 references (`552658b`).** The source predictor enables TF32 in
  its constructor, so earlier reports that disabled it before construction were
  not equal-precision comparisons. Configuring TF32 after construction made ten
  differing output sets exact; historical reports carry a `precision_audit` field.
- **Output visibility (`552658b`).** The source computes a GPU suppression
  candidate but hides objects using the published host set; the native owner now
  follows the published set.
- **Reverse point edit (`641d234`).** The remaining 120-pixel difference at
  reverse frame 17 came from the adapted source: its consolidation picked the
  older conditioning entry's `obj_ptr` while taking the newer click's masks.
  Replaying the native temporal conditioner with only that pointer changed
  reproduces the old reference exactly; with `--sam31-refresh-refined-pointer`
  all 51 compared tensors and all 15 outputs match. Traces come from
  `sam3_video_pipeline_probe --edit-sequence-trace` and
  `video_pipeline_parity.py --trace-edit-state`; the comparators are
  `native/tests/reverse_edit_trace_parity.py` and
  `reverse_edit_pointer_replay.py`. Evidence:
  [reverse-pointer-provenance.json](evidence/reverse-pointer-provenance.json),
  [reverse-pointer-replay.json](evidence/reverse-pointer-replay.json).

## Open issues

- The native-versus-source FP16 difference above.
- Physical multi-GPU execution, video on Windows/Turing, codec coverage beyond the
  tested cases, and dataset-scale quality and performance evaluation.
- CPU stability in the original development environment. With the NVIDIA
  development PyTorch build (`2.10.0a0+b558c986e8.nv25.11`), CPU comparisons
  crashed intermittently: in the original `PromptEncoder.get_dense_pe()` without
  the native library loaded (1 of 5 runs of
  `native/tests/diagnose_cpu_prompt_encoder.py`), in a native-only CPU history
  comparison, and in MKL `erfinv` during source weight initialization with four
  threads. One CPU thread avoided the failures in those tests. The official
  standalone LibTorch CPU build passes all CPU CTests and an installed C client,
  but long-running CPU stability has not been established.
