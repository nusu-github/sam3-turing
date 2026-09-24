# Video detection–tracking pipeline

How the native runtime reproduces the original SAM3/SAM3.1 high-level video
predictor (`sam3_video_base.py`, `sam3_multiplex_base.py` and their predictor
classes) around the low-level tracking sessions described in
[COMPONENTS.md](COMPONENTS.md). The owning `VideoPredictor`
([VIDEO_PREDICTOR.md](VIDEO_PREDICTOR.md)) runs these stages for every frame;
the headers below can also be composed directly.

Many source behaviors are observable quirks rather than obvious designs. They are
reproduced deliberately and must not be "simplified" merely because some clips
give the same masks: each is covered by comparisons that call the actual source
methods at zero tolerance.

## Per-frame stages

| Stage | Header | Source counterpart |
|---|---|---|
| Shared frame features and video detection | `video_frame.h` | detector call, NMS and query filtering |
| Tracker propagation and collection | `tracking_session.h`, `multiplex_session.h`, `video_collective.h` | tracker step, rank gather |
| Detection–track association | `association.h` | `_associate_det_trk` |
| Hotstart state and confirmation | `hotstart.h` | `_process_hotstart`, `_process_hotstart_gpu` |
| Recent occlusion, reconditioning preparation | `occlusion.h` | planning phase |
| Update planning and local execution | `video_update.h` | `run_tracker_update_planning_phase` / execution |
| Neural reconditioning | `video_recondition.h` | `_recondition_masklets` |
| Globally suppressed memory | `video_memory.h` | `_tracker_update_memories` |
| Object births and removals | `video_objects.h` | `_tracker_add_new_objects`, `_tracker_remove_objects` |
| Output buffering and final postprocessing | `video_output.h` | output generators and postprocessors |
| Action routing, partial propagation, cache | `video_interaction.h` | interactive predictor history |
| Point/mask edits | `video_edit.h` | high-level add_prompt/add_new_mask ([VIDEO_EDIT.md](VIDEO_EDIT.md)) |

Policy arithmetic (association and occlusion mask counts) defaults to FP32,
independently of the neural precision; see below.

## Shared frame features and video detection

`video_frame.h` evaluates one full 1008-pixel vision trunk per frame and derives
the three detection pyramid levels and projected tracker features from it. SAM3
shares its tracker neck for interactive and propagation calls; SAM3.1 produces
both necks from the same trunk. `detect` accepts the unrestricted
`GroundingPrompt` API (text, geometry, visual and previous-mask features) and
enables joint presence scoring for both video models (SAM3 image inference uses a
different scoring default).

`encode_rgb` reproduces the upper-level image-folder loader: Pillow bilinear
resize, F32 division, F16 storage and F16 normalization, widened losslessly to
F32. This is deliberately different from the low-level tracker's bicubic/F32 JPEG
path; both follow Pillow 12.2's vertical-first ordering for extremely tall
shrinking images. Other source transforms are selectable
([VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md)). Text is batched with the source's
auxiliary `visual` (plus `geometric` for SAM3.1) slots to keep its numerical batch
shape; the runtime text is not restricted.

`postprocess_video_detections` handles every prompt independently. SAM3 compacts
retained queries after NMS; SAM3.1 keeps every slot and sorts the keep flags.
Three source NMS policies are distinguished:

- SAM3 greedy suppression, with stable score ordering on CUDA;
- SAM3.1 standard batched greedy suppression, keeping ATen's tie order;
- SAM3.1 alternate single-frame perflib behavior, where rejected earlier rows can
  still suppress later rows and the self-IoM denominator uses the row area (the
  source's missing transpose), preserved as is.

NMS uses ATen and precompiled kernels without Triton, and there is no object or
query cap. An empty perflib input returns empty keep flags (the source fails in
a reshape).

## Association

`associate_tracking` implements `Sam3VideoBase._associate_det_trk`,
`Sam3MultiplexBase._associate_det_trk` and `_associate_det_trk_compilable`:

- The smaller spatial area is bilinearly resized to the other before logits are
  thresholded at zero; on equal areas with different shapes, detections are
  resized to the tracks. Inclusive/strict comparisons follow the source.
- IoM converts the matmul intersection to int64 and uses integer areas with a
  float epsilon; IoU uses float areas and a clamped union. `use_iom` changes the
  whole association metric, like `use_iom_recondition`.
- Ambiguous rows/columns are zeroed before reconditioning and match metadata.
  Ties select the first track; repeated reconditioning assignments keep the last
  detection in iteration order. Metadata keeps track-ID order and kept detections
  with empty match lists, plus `recondition_order` (Python dicts keep the first
  insertion position of an overwritten key; a sorted C++ map would not).
- Empty inputs keep two source policies. SAM3 with no tracks marks every supplied
  detection new (its keep mask must be all true); with no detections only
  nonempty tracks are unmatched. SAM3.1 with no tracks applies the new-detection
  score threshold without intersecting `det_keep`; with no detections all tracks,
  including empty ones, are unmatched.
- `pad_tracks_to` reproduces SAM3.1's optional compile padding with zero masks
  (padding matters at zero thresholds); it never truncates.
- The working many-to-one path is implemented; upstream asserts that
  one-to-one/Hungarian matching is disabled, so no Hungarian fallback is invented.

`realize_association` transfers the decision fields needed by host planning.
`assign_detection_devices` keeps lowest-workload/lowest-index placement (one
object at a time for SAM3, capacity-sized groups for multiplex).
`detection_boundary_keep` keeps the strict normalized box-center margins.

**Association precision is separate from neural precision.** A 288×288 full mask
has 82,944 foreground pixels; the source's float matmul under FP16 autocast
returns an infinite intersection, making IoU infinite and the IoM float-to-int64
cast device dependent (on CPU, two identical full masks become an unmatched new
detection). The native default is FP32 even inside an FP16 neural context;
explicit same-mode reference modes reproduce the overflow for comparisons only.
[association-dense-mask.json](evidence/association-dense-mask.json) records the
observed decisions.

## Hotstart state and confirmation

The source has two observable hotstart policies that must stay distinct:

- **Host (`_process_hotstart`, SAM3 and SAM3.1 CPU)**: ID-indexed first-frame
  indices, cumulative unmatched-frame lists, keep-alive counters, overlap-pair
  frame lists and per-frame suppressed/removed sets; duplicates follow first
  appearance with input order for ties. `update_host_hotstart` copies the state,
  returns newly removed IDs and leaves the input untouched on validation failure.
- **Device (`_process_hotstart_gpu`, SAM3.1)**: position-indexed tensors. Every
  track not matched by the matrix is decremented, with an optional second
  decrement for empty tracks; overlap accumulation uses upper-triangular pairs
  with strict first-frame ordering; suppression is computed before overlap
  removal, so remove/suppress masks are not forced disjoint.
  `update_device_hotstart` implements this on CPU or CUDA;
  `compact_device_hotstart`, `select_device_hotstart` and
  `extend_device_hotstart` follow removals, reorders and additions. Inputs are not
  modified; unchanged fields may share storage, so treat state tensors as
  immutable.

`update_confirmation` follows ID remapping across additions, removals and
reorders: new objects start unconfirmed, a match increments the consecutive
count, a miss resets it, and confirmed objects stay confirmed (source status
values 1/2).

The source device policy materializes `[Ndet,Nobj,Nobj]` float outer products.
For at most 2^24 detections the native FP32 `Aᵀ·A` gives the same counts exactly
(binary inputs, every partial sum exactly representable) with autocast disabled;
beyond that the original outer-product order is used. Both sides of the boundary
match the original at 16,777,216 / 16,777,217 detections. The persistent pair
matrix is still quadratic in objects. Isolated update at 200 detections / 512
objects on Blackwell: 0.79 → 0.38 ms and 217 MB → 6.6 MB extra peak allocation
([hotstart-benchmark.json](evidence/hotstart-benchmark.json)).

## Recent occlusion and reconditioning

`update_occlusion` keeps the source's forward/reverse comparisons, inclusive IoU
threshold, strict history ordering, ties and the finite removal sentinel 100000
(kept even for reverse traversal and frame indices above 100000). Empty and
suppressed masks take the current frame as their history; only suppressed masks
get logits −10. SAM3 uses the removal sentinel only when a history key is absent
(a stored value, even −1, wins; `known` distinguishes absent from −1); SAM3.1
always overrides removed objects and supports `allow_unoccluded_to_suppress`.
`update_host_occlusion` adapts the SAM3 dictionary; `update_device_occlusion`
replaces the SAM3.1 hotstart state's occlusion tensor and needs the updated
hotstart state in the same index space. Mask IoU counts default to FP32.

Reconditioning gates differ by model. SAM3.1 planning tests only the **first**
candidate's box IoU combined with **any** candidate's qualifying detection score;
SAM3 tests each paired IoU/score, skips missing IDs and empty masks, and records
geometry-triggering IDs. A periodic or geometry trigger reconditions all
candidates; the geometry ID set is used for output overrides, not as a filter.

`prepare_recondition_masks` keeps the source bilinear/sign order. SAM3 uses raw
track scores > 0.8 and leaves global low-resolution logits unchanged. SAM3.1 uses
sigmoid(track score) > 0.8, keeps old low logits where their signs agree with the
detection, then applies video hole/sprinkle cleanup: holes become +0.1, sprinkle
removal sees the filled holes, and its area is bounded by half the foreground.
The source ignores `sprinkle_removal_area` and uses `fill_hole_area` for both
passes (nonpositive disables both). Candidate IDs must exist in the global track
IDs (SAM3.1's source argmax would silently pick index zero).
`recondition_batches` keeps SAM3's per-candidate/per-state order; SAM3.1 assigns
each target to its first containing state and groups targets in first-encounter
order.

`tracking_mask_boxes` returns the source's inclusive int32 extrema (zero boxes
for empty masks) by reducing per-axis projections instead of full-resolution
coordinate temporaries (200 masks at 288×288: 0.93 → 0.17 ms, 133 MB → 1 MB,
[occlusion-benchmark.json](evidence/occlusion-benchmark.json)).
`diagonal_box_iou` keeps NaN for zero-union boxes.

### Neural execution

`execute_reconditioning` applies prepared masks through the real sessions. SAM3
applies each candidate to every containing state and preflights those states
before the next candidate. SAM3.1 batches candidates into their first containing
state and afterwards preflights, in session order, every state sharing an
affected ID (affected IDs include all objects of an edited state). Execution
mutates sessions and is not an all-session transaction.

`Sam31TrackingSession::recondition_masks` updates existing IDs on an existing
frame without changing bucket assignments: it demultiplexes pointers, applies the
mask/pointer/score changes, remultiplexes, keeps conditioning membership, updates
pending brush previews and marks the frame for preflight. Unknown/duplicate IDs
fail without changing metadata; the edit rolls back on exceptions, including
paged-state read failures. Native dense history keeps a 1008-resolution auxiliary
mask grid while the source accepts 1152 brush logits, so the grid is rebuilt from
the updated low logits and regenerated with source non-overlap rules during
preflight. Preflight clears already consolidated ±1024 brush previews so a later
edit on the same frame starts from stored logits.

### Globally suppressed memory

`prepare_video_memory` resizes all object masks to the memory encoder's 1152×1152
grid, computes pixel winners and suppresses whole masks that keep less than 0.3 of
their positive area. Surviving masks keep their logits and may still overlap
(this is not pixelwise clipping). SAM3's warmup flag can skip suppression; SAM3.1
always suppresses but returns singletons unchanged, while SAM3 may clamp a
singleton empty mask to at most −10. Memory-only proxy logits are +10 for any
positive pixel after suppression and −10 otherwise; they do not replace predicted
scores or masks.

`video_memory_rows` maps explicit global IDs to each state's row order and
rejects missing/duplicate IDs (the source SAM3 host uses contiguous per-rank
slices; the dynamic SAM3.1 host assumes matching orders). Session
`update_memory` stores BF16 memory and positions in the matching conditioning or
tracked output without changing predictions or pointers; SAM3.1 can reapply the
no-object pointer projection to newly suppressed pointers. Each session stages
and rolls back its own update.

History retains the effective encoder inputs (`memory_masks`,
`memory_object_logits`) separately from the predictions, so a later bucket-layout
rebuild re-encodes the suppressed inputs rather than the predicted logits. They
are paged and remapped with the history and cleared when memory is invalidated or
re-encoded. All three SAM3.1 memory paths use the source sequence-to-BCHW view of
the image features (see the stride fix in [VALIDATION.md](VALIDATION.md)).

### Object births and removals

`prepare_video_object_masks` resizes detector logits to the 1152 grid without
antialiasing, then thresholds. `add_video_objects` runs the mask-input API and
preflight for all supplied IDs and returns the destination state; duplicates are
rejected before editing. SAM3 creates one state per birth batch; SAM3.1 defaults
to best fit (the existing state with the fewest free slots that can hold the whole
batch, ties by order), with `FirstState` and `NewState` grouping matching the
other source branches. Factories return empty sessions sharing the frame core
and feature cache. Removal ignores unknown IDs and drops empty sessions; SAM3.1
removes all requested IDs of a state together with one history remap
(`Sam31TrackingSession::remove_objects`).

## Update planning and execution

`VideoMetadata` holds rank-ordered IDs, bucket workloads, monotonically assigned
IDs, object and per-frame scores, occlusion history and the model-specific
hotstart state (`initialize_video_metadata`). `plan_video_update` is pure with
respect to its inputs and follows the source order:

1. Associate detector and tracker masks (SAM3's optional boundary filter applies
   only to new detections; SAM3.1 consumes the detector keep mask).
2. Allocate IDs and plan least-workload placement from object counts (SAM3) or
   bucket counts (SAM3.1); update hotstart.
3. Evaluate periodic/geometric correction and prepare mask edits. SAM3.1 updates
   global low logits by sign agreement and cleanup before occlusion.
4. Apply recent-occlusion suppression, append new IDs, remove deleted IDs,
   initialize detection scores (removed object scores stay −10000) and update
   confirmation by ID.
5. Compact and extend the SAM3.1 device metadata, explicitly reordered to the
   final rank-concatenated ID order (the source appends new rows globally, which
   can differ with several ranks).

Warmup-disabled planning keeps the source skip behavior; metadata produced during
warmup must be discarded before normal hotstart. `execute_video_update` applies
correction/preflight, memory updates, births, then removals, and refreshes
SAM3.1 bucket workloads. `finalize_video_scores` performs the source's final
sigmoid write, which overwrites a removed object's **frame** score while its
persistent **object** score stays −10000. `build_video_outputs` resizes masks to
video resolution, thresholds at zero and applies SAM3's geometry-triggered output
overrides; removed IDs stay in this raw mapping for later filtering. There is no
object cap or ID truncation (integer ID overflow is rejected).

## Output buffering and final postprocessing

`video_output.h` implements the predictor output stage:

- Masks are ordered by sorted object ID. Empty masks and suppressed, removed or
  unconfirmed IDs are excluded before boxes are computed; boxes use inclusive
  pixel maxima, so a single-pixel mask has zero extent.
- Non-overlap resolution runs after boxes using tracker probabilities (missing
  scores are zero); equal positive scores choose the first sorted ID, several
  masks with nonpositive scores lose their pixels, and a singleton bypasses
  resolution. Rows are not filtered again afterwards.
- Cached masks exclude hidden IDs but keep empty masks and the pre-overlap masks.
- The hotstart delay buffers outputs; removed IDs accumulate only when the delay
  is enabled. Confirmation looks forward/backward by threshold − 1, clamped to the
  video; unobserved future status is not invented. Each frame entering a
  postprocessing batch snapshots the removal set; confirmation is read when the
  batch is processed, so the batch size affects emission timing only.
- The end frame drains the buffers; `cancel()` drops pending output and closes
  the buffer; enqueued tensors are cloned so later mutation cannot change delayed
  results.

Source defaults: delay 15, confirmation threshold 3, batch size 1 (SAM3) or 16
(SAM3.1). Nonemptiness is reduced with one CPU transfer per frame instead of one
synchronization per object. The SAM3.1 predictor hides objects using the
published host set `suppressed_obj_ids`, not the GPU `to_suppress_mask`
candidate that its planning never publishes.

## Action routing, partial propagation and the cache

`VideoAction` records add/remove/refine and full/partial/fetch/cancel events.
`route_video_actions` follows the source history heuristic: an initial full pass,
IDs selected since the previous propagation, boundary/two-pass fetch, and SAM3.1
cancellation recovery (SAM3's history does not accept cancel events). SAM3's
forced-tracker extension selects every active ID, or full propagation if none.
`video_processing_range` resolves the earliest initialized frame when no start is
given; forward ranges include the start, reverse ranges begin one frame before
it, end bounds are inclusive and large step counts do not overflow.

`VideoInteractionState` owns the action history and the filtered pre-overlap
masks from the output stage. Partial merge upsamples the selected low masks in
FP32, replaces only those IDs, applies suppression and the final output stage and
updates the cache; other IDs keep their saved masks. Per-frame tracker scores
stay raw on this path (no extra sigmoid), FP32 for SAM3 and in the incoming
precision for SAM3.1. Missing-cache behavior differs by model: SAM3 fetch returns
an empty result and merges refined masks into an absent frame, while SAM3.1 fetch
requires a cached frame and its `_build_sam2_output` returns empty before merging.
`propagate_video_refinements` runs every local session containing a requested ID
with memory encoding and returns only the requested IDs.

## Validation

Policy comparisons call the actual source methods (or AST-extracted planning
blocks where the logic is inline) at zero tolerance on CPU and CUDA in FP32, FP16
and BF16-reference, with neural phases replaced by recording fixtures:

| Script | Scope |
|---|---|
| `association_parity.py` | Association, placement, boundary filtering; 852 workflows |
| `hotstart_parity.py` | Host/device hotstart, compaction/extension, confirmation; 44,628 comparisons |
| `occlusion_parity.py` | Occlusion, cleanup, gates, edit batches; 18,450 comparisons |
| `video_update_parity.py` | Planning and `build_outputs`, 96 workflows / 768 frames |
| `video_detection_parity.py` | NMS policies and query ordering |
| `video_output_parity.py` | Output generators and postprocessors, 96 workflows per run |
| `video_interaction_parity.py` | Routing, ranges, cache merge and postprocessing |
| `video_objects_parity.py`, `video_memory_parity.py` | Births/removals and memory policy with actual weights |

Neural comparisons with actual weights (`recondition_storage.py`,
`video_memory_storage.py` and the session parity scripts) check resident,
offloaded and paged history. The coherent comparison
`video_pipeline_parity.py` runs the original predictor on real frames:

- SAM3 and SAM3.1 match every raw mask and low tracking value for 34 frames with
  four people and corrections at frames 16 and 32; final outputs and emission
  timing match with `--final-output`, and reverse partial propagation over frames
  17–15 matches with `--partial-output`.
- The matching configuration is BF16-reference, TF32 off, and for SAM3.1
  batch-one grounding with complex RoPE (the builder default batch 16 / real RoPE
  is recorded separately). Scores allow only 1e-8 JSON rounding.
- Native FP16 versus these BF16 references is a separate precision measurement
  (for example minimum IoU 0.9990 on the three-frame SAM3.1 fixture), not parity.

Evidence: `evidence/association-*.json`, `hotstart-*.json`, `occlusion-*.json`,
`recondition-*.json`, `video-memory-*.json`, `video-objects-*.json`,
`video-update-planning-validation.json`, `video-detection-validation.json`,
`video-frame*-validation.json`, `video-output-*.json`,
`video-interaction-*.json`, `video-pipeline-*.json` and
`video-recondition-history-*.json`.
