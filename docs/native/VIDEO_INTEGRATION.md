# High-level video integration

The native image and interactive video backends exist, but the high-level
text/visual-guided video detector/tracker pipeline remains incomplete. C ABI
availability is not evidence that this pipeline is finished. This document
records the integration contracts against the repository's actual source.

## Detection/track association

`sam3/association.h` provides the native implementation of
`Sam3VideoBase._associate_det_trk`, `Sam3MultiplexBase._associate_det_trk` and
`_associate_det_trk_compilable`. It accepts floating mask logits and scores,
selects the smaller spatial area before bilinear resizing, then thresholds
logits at zero. When areas tie and dimensions differ, detections are resized
to the track dimensions, as in the source. All comparisons retain their source
inclusive/strict boundaries.

Both IoU and IoM preserve the source arithmetic and autocast mode. IoM converts
the matmul intersection to int64 and uses integer areas with a float epsilon;
IoU uses float areas and a clamped union. The `use_iom` option affects the whole
association metric, just as `use_iom_recondition` does in the source. Ambiguous
rows/columns are zeroed before reconditioning/match metadata is calculated.
Ties select the first track; repeated reconditioning assignments retain the
last detection in iteration order. Output metadata preserves track-ID order
and kept detections with empty match lists.

Empty cases deliberately preserve two different source policies:

- SAM3 with no tracks marks every supplied detection new; its caller has
  already filtered detections. A supplied keep mask must therefore be all true.
  With no detections, only nonempty tracks are unmatched.
- SAM3.1 with no tracks applies the new-detection score threshold but does not
  intersect it with `det_keep`. With no detections, all tracks are unmatched,
  including empty tracks. These are observable source behaviors, not a new
  interpretation of what matching should mean.

`pad_tracks_to` reproduces optional SAM3.1 compile padding with zero masks. It
never truncates or drops tracks, and zero disables padding. Padding matters at
zero thresholds, so it cannot simply be erased when replicating a configured
source run. The native function has no maximum detection/track count. It does
not adopt the source planning phase's object-dropping limit. Upstream explicitly
asserts that one-to-one/Hungarian matching is disabled; this module implements
its working many-to-one path, not an invented Hungarian fallback.

`associate_tracking` retains tensor decisions on the input device.
`realize_association` creates the host metadata needed by state planning. The
latter transfers the same decision fields used by the original lazy result;
it is not advertised as a synchronization-free planner.

`assign_detection_devices` preserves lowest-workload/lowest-index tie breaking,
one object at a time for SAM3 or capacity-sized groups for multiplex tracking.
It returns a placement plan only. Multi-GPU inference/communication is still
unimplemented. `detection_boundary_keep` preserves the source's strict normalized
box-center margin checks.

The standalone test requires no weights/Python. `association_parity.py` calls
the actual source methods and compares every decision tensor and realized
metadata at zero tolerance on CPU/CUDA. It includes missing detections/tracks,
false keep entries, exact thresholds, ambiguity, NaNs/infinities, strided masks,
resize direction/area ties, 257 detections against 513 tracks, and stored full
200-query neural masks. Those stored image/video masks are from different
fixtures: this checks real tensor arithmetic, not tracking accuracy of a
coherent detector/tracker sequence. Placement and boundary tests are separate.

## Association precision is separate from neural precision

The default association mode is FP32, including when a caller has an active
FP16 neural autocast context. Preserve that separation in the high-level host.
A 288x288 full mask contains 82,944 foreground pixels; the source's float matmul
under FP16 autocast returns an infinite intersection. IoU can become infinite,
and the IoM float-to-int64 cast has device-dependent behavior (the local CPU
FP16 path treats two identical full masks as a new unmatched detection). The
native reference modes reproduce these behaviors; passing same-mode parity
is not proof of sensible association decisions or tracking quality.

The dense-mask fixture is in the actual-source comparison suite. A standalone
native regression also wraps the default FP32 call inside an outer FP16 scope
and verifies the correct full-mask match. `association-dense-mask.json` records
the observed CPU/CUDA decisions across modes. BF16-reference is for capable
reference hardware, not Turing execution. End-to-end quality evaluation remains
required for the intended FP16 neural / FP32 association combination.

## State policies that must remain distinct

The next integration steps are hotstart state updates, confirmation, occlusion
suppression, reconditioning, object insertion/removal and propagation/cache
coordination. Relevant sources are `sam3_video_base.py` and
`sam3_multiplex_base.py`, including their planning and execution phases.

The CPU `_process_hotstart` implementations in those files have the same state
logic (their logging differs). They retain first-frame indices, accumulated
unmatched frame lists, keep-alive counters, overlap-pair frame lists and per-frame
suppressed/removed ID sets. Duplicate ordering follows first appearance and
input track order for ties. These lists are cumulative, not consecutive streaks.

SAM3.1's `_process_hotstart_gpu` has different observable behavior:

- Unmatched keep-alive updates apply to every track not matched by the matrix;
  optional empty-track decrement can apply a second decrement. The CPU path
  instead uses its separate unmatched-ID list.
- GPU overlap accumulation uses upper-triangular position pairs and strict
  first-frame ordering; the CPU path chooses one earliest matching ID, with
  input-order tie breaking. Equal-first-frame behavior therefore differs.
- GPU suppression and the CPU iteration over accumulated unmatched-ID entries
  are not interchangeable. Position-indexed GPU metadata must be compacted
  after removals and extended after additions as in the planning phase.

Do not unify these policies merely because some happy-path clips produce the
same masks. Reference tests must cover each path's persistent metadata and
forward/reverse ordering before connecting it to the native sessions. Full
end-to-end comparisons are still required after that connection.

## Native hotstart and confirmation helpers

`sam3/hotstart.h` now implements the state helpers described above. They are
building blocks for the pending high-level host, not a completed video predictor.

`update_host_hotstart` copies the ID-indexed host state and returns updated state
plus newly removed IDs. It keeps accumulated unmatched/overlap frame lists,
first-appearance tie order, keep-alive clamping, per-frame suppressed sets and
persistent removal sets. Validation failure leaves the caller's state untouched.

`update_device_hotstart` implements the separate SAM3.1 position-indexed policy
on CPU or CUDA. It preserves the source's unconditional decrement for unmatched
matrix columns, optional second decrement for empty tracks, cumulative unmatched
counts, upper-triangle pair counts, strict first-frame ordering and separate
remove/suppress masks. Suppression is computed before overlap removal, as in the
source, so those masks are not forcibly made disjoint.

`compact_device_hotstart` returns both filtered state and retained indices, for
consistent filtering of external IDs/masks. `select_device_hotstart` supports
explicit order changes and selects both axes of pair counts. `extend_device_hotstart`
appends first-frame/counter/removal/occlusion values and grows the pair matrix.
These functions do not modify input tensors; unchanged fields can share storage,
so callers must treat retained state tensors as immutable.

`update_confirmation` follows source ID remapping across additions/removals/order
changes. New objects start unconfirmed; matching increments the consecutive count,
a missed detection resets that count, and an already confirmed object stays
confirmed. The status values are the source's 1/2 convention. This does not yet
wire user-action confirmation into a high-level host.

### Exact overlap-count optimization

The source device policy materializes `[Ndet,Nobj,Nobj]` float outer products
before summing them. For at most 2^24 detections, native FP32 `A.transpose(0,1) @ A`
produces the same counts without this cubic temporary: inputs/products are binary
and every partial integer sum is exactly representable in FP32. Neural autocast
is disabled for this calculation. It neither rounds counts to BF16 nor allows
FP16 overflow. Beyond that exact-integer range, the original outer-product/sum
order remains the fallback. That boundary selects an arithmetic implementation;
it does not drop detections or impose a count limit. Both sides of the boundary
are checked against the original at 16,777,216/16,777,217 detections on CPU/CUDA.

The persistent pair-count state is still quadratic in object count. This change
reduces the intermediate tensor, not every memory category or overall model size.
At 200 detections/512 objects, the source outer product alone is 209,715,200 bytes;
the native float pair matrix is 1,048,576 bytes, plus other working tensors.

`hotstart-benchmark.json` measures the isolated full state update on local
Blackwell with identical outputs. Across two measurement orders, median CUDA
event time is 0.792–0.795 ms for source and 0.377–0.383 ms for native. Peak PyTorch
allocated memory above the same 10,737,152-byte baseline is 217,432,576 versus
6,568,448 bytes (about 97% less). Each case has five warmups and 30 samples. These
are synthetic state-update measurements; they include stream/launch gaps and
exclude host RSS/total reserved device memory. They are not end-to-end video
throughput measurements or Turing performance results.

### Validation scope

`hotstart_parity.py` calls the actual CPU methods in both original classes,
SAM3.1's device method and the original confirmation methods. Compaction/extension
are inline in the original planning phase, so the test extracts their original
AST blocks rather than reimplementing their reference algorithms. The report
records the source-file hash. It checks forward/reverse history, cumulative
counters, zero thresholds, additions/removals/reordering, empty states, state
immutability and all persistent metadata fields at zero tolerance. The standalone
C++ test also distinguishes CPU first-frame ties from GPU strict ordering and
checks a 301-count update under outer BF16 autocast.

`hotstart-validation.json` contains 8,064 host state-field comparisons, 35,568
device tensor/scalar comparisons, 960 confirmation comparisons and 36 large-count
boundary comparisons: 44,628 total. CTest passes 17 CUDA-enabled and 10 custom-CUDA-
disabled checks. Standalone CPU/CUDA probes run with PATH=/nonexistent. This does
not resolve the separately recorded intermittent development full-model CPU fault.
No GitHub Actions or Windows/Turing runtime validation was performed.

At this milestone, remaining integration included recent-occlusion suppression,
reconditioning, coordinated detector/tracker insertion/removal, visual prompt/cache handling,
user-action state and full high-level propagation/output behavior. The new
helpers must be connected and compared in real neural video workflows before
claiming the high-level predictor is complete.

## Recent occlusion and reconditioning preparation

`sam3/occlusion.h` provides recent-occlusion suppression, source-specific
reconditioning gates, mask preparation and ordered edit batches. These are
C++/ATen CPU/CUDA helpers; the complete high-level neural video host remains
unfinished. The new helpers require no weights, Python, Triton or extra model
variant at runtime.

### Occlusion history

`update_occlusion` preserves the source forward/reverse comparisons, inclusive
IoU threshold, strict history ordering, ties and the finite removal sentinel
100000. Empty masks and suppressed masks get the current frame as their new
occlusion history; only suppressed masks have logits replaced by -10. The
finite sentinel is retained even for reverse traversal and frame indices above
100000. SAM3 only uses the removal sentinel when a history key is absent;
a stored history value, including -1, wins over removal. `known` distinguishes
an absent key from a stored -1. SAM3.1 always overrides the history of removed
objects and separately supports `allow_unoccluded_to_suppress`.

`update_host_occlusion` adapts the SAM3 ID-indexed history dictionary.
`update_device_occlusion` replaces the SAM3.1 hotstart state's occlusion tensor;
it must receive the updated hotstart state, with masks/removal flags in the same
index space. Inputs remain unchanged and unchanged state fields can share
immutable tensor storage. An empty host batch retains existing history, as in
the source. The high-level host remains responsible for enabling the policy and
for applying suppression before memory encoding.

Mask IoU counts default to FP32, independently of neural autocast. Explicit
FP16/BF16 reference modes reproduce source autocast behavior, including FP16
intersection overflow for full 288x288 masks; matching that behavior is not an
accuracy guarantee. The standalone test checks that the default still suppresses
a dense overlapping pair correctly under an outer FP16 autocast scope.

### Candidate order, gates and edits

Association metadata now carries `recondition_order`. Python dictionaries keep
the first insertion position when a value is overwritten; a sorted C++ map does
not. `recondition_candidates` uses the explicit order (with sorted-map fallback
for hand-constructed legacy metadata). This matters because the actual SAM3.1
planning gate tests only the **first** candidate's box IoU, combined with **any**
candidate's qualifying detection score. SAM3 instead tests each paired IoU/score,
skips missing IDs and empty masks, and records geometry-triggering IDs. A periodic
or geometry trigger asks for all candidates to be reconditioned; the geometry-ID
set is not an execution filter.

`prepare_recondition_masks` retains source bilinear/sign order and accepts
arbitrary valid candidate counts. SAM3 uses raw track scores >0.8 and leaves
global low-resolution logits unchanged. SAM3.1 uses sigmoid(track score) >0.8,
keeps old low logits where their signs agree with the detection, then performs
video-specific hole/sprinkle cleanup. Holes become +0.1; sprinkle removal sees
those already filled holes, and its area threshold is bounded by half of total
foreground area. The source currently ignores `sprinkle_removal_area` and uses
`fill_hole_area` for both passes; a nonpositive area disables both.

Preparation requires candidate IDs to exist in global track IDs. SAM3's source
lookup also fails for absent IDs; SAM3.1's source argmax lookup would silently
select index zero. Native preparation rejects this inconsistent state explicitly.
SAM3.1 preparation also requires matching low-mask resolutions/dtypes, as needed
by its source scatter. The SAM3.1 gate rejects missing candidate IDs rather than
allowing mismatched paired-box arrays. These validations do not limit valid
prompts, objects or detections.

`recondition_batches` preserves SAM3's per-candidate/per-containing-state order.
SAM3.1 assigns each target to its first containing state and, when multiplexing,
groups targets in first-encounter state order. These are **edit recipes**; preparation itself does not run neural updates.
The executor described below consumes them and performs model-specific neural
edits/preflight. Caller masks and state are not modified during preparation.

### Geometry optimization and validation

`tracking_mask_boxes` returns the source's inclusive int32 extrema and zero boxes
for empty masks. It projects the binary mask onto each axis and reduces these
projections, avoiding source full-resolution coordinate-selection temporaries.
It reuses the row projection for the empty check. `diagonal_box_iou` preserves
source arithmetic, including NaN for degenerate zero-union boxes; no epsilon or
coordinate correction is introduced.

`occlusion_parity.py` compares actual source occlusion/cleanup methods, original
AST-extracted gate blocks, and original reconditioning methods with a recording
tracker. The recording tracker compares mask edit batches without executing their
neural updates. Hotstart -> occlusion -> compaction -> extension is also compared
against original state updates. Coverage includes forward/reverse history,
missing versus stored -1 history, removed objects, first-pair order, empty and
201-object batches, 288x288 dense masks, noncontiguous inputs and three precision
modes on CPU/CUDA. The source CUDA connected-components path requires contiguous
storage; strided native cleanup is compared with the same source values made
contiguous. Source-file hashes are recorded.

The report has 18,450 exact comparisons, including unchanged input state checks.
Updated association validation adds order comparisons and the deliberately
permuted matching fixture: 852 workflows, 10,968 exact tensor/metadata comparisons,
72 placement checks and 32 boundary checks. This verifies candidate-order
preservation that the earlier map-only comparison did not test.

CTest passes 19 CUDA-enabled and 11 custom-CUDA-disabled checks. Standalone
CPU/CUDA probes run with PATH=/nonexistent. These component checks do not resolve
the recorded intermittent full-model CPU runtime issue. No GitHub Actions or
Windows/Turing execution was performed. Portable packaging, complete neural video
integration, codecs, multi-GPU execution and overall quality validation remain.

`occlusion-benchmark.json` measures isolated box extraction for 200 synthetic
288x288 masks on Blackwell, with five warmups and 30 samples in both measurement
orders. Median CUDA-event time is 0.929–0.930 ms for source and 0.172–0.175 ms for
native. Peak additional PyTorch GPU allocation is 132,712,448 versus 1,048,064
bytes (about 99.2% less), above the same 16,777,216-byte baseline. Event timing
includes stream work/launch gaps, and allocation excludes host RSS/reserved GPU
memory. This is not full-video throughput or Turing performance.

## Neural execution of reconditioning

`sam3/video_recondition.h` connects prepared masks to real tracker sessions.
`execute_reconditioning` borrows distinct session pointers, preserving shared
model-core/visual-cache ownership. It accepts `ReconditionMasks` produced by the
preparation helper; the caller still chooses whether periodic/geometry gates
trigger execution and still owns the updated global low logits.

For SAM3, each candidate is applied to every containing state, followed by
preflight of those states before advancing to the next candidate. For SAM3.1,
candidates are batched into their first containing state. After all batches,
every state sharing any affected ID is preflighted in session order. Affected
IDs include all objects in an edited state, rather than only the target masks.
The returned edited/preflight state lists describe execution, and are distinct
from SAM3's geometry-triggered ID set used by output assembly. An empty prepared
list is a no-op. Execution mutates sessions and is not an all-session transaction:
a later failure does not undo an earlier state's completed edit/preflight.

`Sam31TrackingSession::recondition_masks` specifically updates an existing frame
and existing IDs through the tested dynamic frame helper. It does not use the
ordinary new-object brush path or change bucket assignments. It demultiplexes
existing pointers, applies selected mask/pointer/score changes, remultiplexes
pointers, retains conditioning membership, updates pending brush previews and
marks the frame for preflight. Unknown/duplicate IDs and absent frames fail
without changing session metadata. The individual edit rolls back on exceptions,
including failures while reading paged state. Preflight encodes the consolidated
masks and subsequent propagation uses the new memory.

Native dense history retains an auxiliary 1008-resolution mask grid, while the
demo's deferred update omits high masks and accepts 1152 brush logits. The session
therefore reconstructs its auxiliary grid from updated low logits after the
frame helper; preflight then regenerates the grid with source non-overlap rules
before memory encoding. This avoids assigning a 1152 brush into a 1008 buffer
without altering the actual source mask/pointer equations.

Repeated correction testing exposed a pre-existing session bug: `video_edits`
kept already-consolidated +/-1024 brush previews. A later edit on the same frame
could suppress/reapply those obsolete previews instead of using stored logits.
Preflight now clears these temporary previews after successful consolidation.
Original point/mask annotations and the resulting history remain available for
future edits, clearing, removal and propagation.

Source neural comparisons use full 72/144/288 projected features and actual
weights; they do not include the vision encoder/detector or a coherent real
video sequence. SAM3 compares the original `_recondition_masklets` with full
session snapshots. SAM3.1 compares actual demo correction/preflight and stored
masks, memory, pointers, logits, positions and conditioning sets, including
repeated/reordered edits on the same frame. Its source reconditioning comparisons
keep state on the compute device because unmodified offloaded scatter raises a
CPU/CUDA device mismatch. Separate native comparisons check resident, CPU-offloaded
and disk-paged execution, including archive cleanup and unchanged older history.
Existing source adaptations for CPU device transfers/FP32 compressed memory are
recorded in the reports. They are not claims of unmodified upstream CPU support.

The high-level detector/tracker coordinator is still incomplete. In particular,
this executor does not yet apply global occlusion-adjusted tracking masks to each
state's current-frame memory, manage detector insertion/removal, visual/text
prompt caches and user actions, or assemble complete high-level video outputs.
The source `_tracker_update_memories` phase after reconditioning/occlusion is the
next integration boundary. Codec support, multi-GPU execution, portable packages
and end-to-end quality/performance validation also remain.

The reconditioning milestone reports 3,607 exact neural output/state comparisons:
SAM3.1 has 1,905 across 24 CUDA workflows (all existing demo comparisons plus the
new correction cases, three modes) and 214 in two CPU FP32 correction workflows;
SAM3 has 1,116 across three CUDA modes and 372 on CPU FP32. Storage equivalence
adds 1,083 exact comparisons across three CUDA modes. The existing dynamic
session invariant suite also passes, including 44 cancellation/resume comparisons,
18 retained points, insertion/removal and old-bucket preservation. Standalone
SAM3/SAM3.1 session probes run with PATH=/nonexistent; the multiplex probe includes
paged history, invalid-correction rejection and shared-ID preflight across two
sessions. CTest passes 19 CUDA-enabled/11 custom-CUDA-disabled checks. The earlier
intermittent full-model CPU runtime failure remains open; these passing runs do
not establish its resolution. No GitHub Actions or Windows/Turing execution was
used. No new model weights or model variants were produced.

## Globally adjusted tracking memory

`sam3/video_memory.h` now connects global mask suppression to actual session
memory replacement. `prepare_video_memory` resizes the complete object mask set
directly to the source memory encoder's 1152x1152 grid. It computes pixel winners
and suppresses whole masks retaining less than 0.3 of their original positive
area. Masks that pass retain their original logits and may still overlap; this
is not unconditional pixelwise clipping. The native area-after calculation uses
boolean winners, avoiding the source full floating non-overlap mask temporary.
No object/detection limit or precision truncation is introduced.

The source SAM3 warmup flag can skip suppression. SAM3.1 always suppresses but
returns singleton masks unchanged, whereas SAM3 can clamp a singleton empty mask
to at most -10. The implementation preserves these differences, argmax tie order,
positive-area tests and nonfinite behavior. Memory-only proxy logits are +10 for
any positive pixels after suppression and -10 otherwise. They do not replace
predicted object scores, masks or confidence fields.

`video_memory_rows` maps explicit global IDs into each local state's actual row
order. The source SAM3 host uses contiguous per-rank slices; the dynamic SAM3.1
host constructs assignments by sorting local IDs and assumes matching state/global
order (its dynamic branch does not add the computed rank offset). The native
coordinator requires the caller's explicit global ID order, supports arbitrary
state order/backfilling, and rejects missing/duplicate IDs rather than relying
on these source assumptions. Source neural comparisons use coherent sorted IDs
on rank zero; separate mapping tests and standalone reversed-global-ID tests cover
the broader native contract. This is not a claim of multi-GPU communication:
callers must provide gathered global masks/IDs when needed, and only supplied local
sessions are updated.

`Sam3TrackingSession::update_memory` stores BF16 memory and cached positions in
both matching conditioning/tracked outputs and refreshes per-object slices,
without changing predictions/pointers. Missing SAM3 frames are a no-op, matching
the original output-store loop. SAM3.1 requires a current frame to obtain its
conditioning membership. `Sam31TrackingFrame::update_memory` updates encoded
memory and saved image features/positions; an optional flag reapplies the loaded
no-object linear projection to newly suppressed pointers before remultiplexing.
The source's original predicted-score threshold controls which pointers qualify.
Each session stages changes and rolls back on failure. Updating a list of sessions
is not an all-session transaction.

### Preserving the inputs to rebuilt memory

Dynamic bucket changes can require re-encoding older SAM3.1 memory. A global
memory rewrite may have used different masks and proxy scores from the stored
predictions, so history now retains `memory_masks` and `memory_object_logits`
separately. They are cloned from the actual encoder inputs, offloaded/paged with
the history, remapped by global IDs and used by `encode_history`. Newly introduced
objects get the same absent -1024 history values as the native dense-history
policy. Using predicted logits during this rebuild would lose suppression.

The retained inputs add runtime history storage; they are not another model or
weight variant. Selected temporal reads still fetch only needed memory/pointer
fields. Ordinary edits that invalidate memory, and preflight that encodes a fresh
consolidated frame, clear obsolete override inputs. Deferred reconditioning keeps
them with the still-existing memory until preflight. Temporary archives preserve
these new fields using the existing dtype/shape/stride/CRC mechanism.

### Evidence and remaining scope

`video-memory-policy-validation.json` records 384 exact source tensor comparisons
across CPU/CUDA FP32/FP16/BF16 and four additional comparisons with 201 objects on
CPU/CUDA. It includes singleton empty masks, warmup, ties, nonfinite and strided
inputs, plus four explicit-ID mapping cases and three rejection checks.

Actual-weight session comparisons include 1,092 SAM3 CUDA and 364 CPU FP32 tensor
comparisons, and 768 SAM3.1 CUDA plus 166 CPU FP32 comparisons (2,390 total,
including repeated-correction regression). They call the original high-level
`_tracker_update_memories` methods and compare subsequent forward/reverse
propagation and stored session fields; SAM3.1 tests both pointer-reapplication
settings. Features are full-grid synthetic projected tensors, not real-video
vision/detector outputs. Existing documented source CPU/FP32 staging adaptations
remain; this is not unmodified upstream CPU validation.

`video-memory-storage-validation.json` adds 1,350 exact comparisons across three
CUDA modes for resident/offloaded/paged execution, memory rewrite -> object
insertion -> history rebuild -> propagation -> correction/preflight. The rebuilt
bucket is also compared against the actual source neural memory encoder using
the retained effective masks/proxies. Original predicted masks/scores remain
unchanged, new historical rows are absent, obsolete overrides clear after
preflight and temporary archives are removed on session destruction.

CTest passes 21 CUDA-enabled and 12 custom-CUDA-disabled checks. Standalone
SAM3/SAM3.1 session tools exercise actual-weight memory replacement with Python
removed from PATH, including reversed global ID order and paged state. The earlier
intermittent full-model CPU runtime fault remains unresolved. No GitHub Actions
or Windows/Turing execution was used; existing sm_75 cubins are compile evidence
only. Portable SDK packaging is still pending.

A complete text/visual-guided video predictor still needs coordinated detection
insertion/removal, prompt/cache/user-action state and output assembly. These
memory helpers accept the already prepared global masks; they do not yet connect
all association/hotstart/occlusion/correction phases into the full coordinator.
Codecs, multi-GPU execution and end-to-end quality/performance validation remain.

## Collections of detected tracking objects

`sam3/video_objects.h` now connects detector object births/removals to owning
collections of native tracker sessions. This implements the local-state boundary
of `Sam3VideoBase._tracker_add_new_objects` / `_tracker_remove_objects` and the
corresponding `Sam3MultiplexBase` methods. It does not yet implement the complete
high-level video predictor or rank communication.

`prepare_video_object_masks` bilinearly resizes floating detector logits to the
1152 input grid without antialiasing, then thresholds at zero. It preserves the
source order (resize before threshold), including strided FP32/FP16/BF16 inputs.
`add_video_objects` accepts all supplied IDs/masks, runs the relevant mask-input
API and preflight, and returns the destination state index. Empty additions are
explicit no-ops. Duplicate IDs, including IDs already present in the collection,
are rejected before editing. The source host calls this boundary only for new
objects; existing-object correction uses the reconditioning API instead.

SAM3 creates one new state per birth batch. SAM3.1 defaults to best-fit placement:
choose the existing state with the fewest available slots that can hold the
whole batch, breaking ties by collection order. Otherwise create another state.
The entire group can span multiple multiplex buckets; there is no detection or
object limit. `FirstState` and `NewState` grouping are also exposed, corresponding
to the alternative source host placement branches. They select native session
groups; they do not change the neural model into a different inference backend.

Factories return empty sessions sharing the caller's frame core and feature
cache. The collection owns sessions via `unique_ptr`; session state owns no model
weights. A fresh session is appended only after successful preflight. In-place
addition to an existing state follows the session's per-operation rollback;
addition plus preflight and the whole collection are not a single transaction.
The caller must coordinate access and supply any global/rank metadata.

Removal ignores unknown IDs and drops empty sessions while retaining collection
order. SAM3 follows the source ID-then-state loop. SAM3.1 removes IDs together per
state, with the new `Sam31TrackingSession::remove_objects` API. This validates
strict requests before mutation, deduplicates requested IDs and remaps affected
history once. Previously repeated single removals rebuilt history repeatedly.
No speedup is claimed without an isolated measurement. Paged state archives are
released through their existing reference-counted lifetime; retained snapshots
can deliberately keep their archives alive.

SAM3.1 continues to use the previously documented native stable-slot/dense-history
policy, including re-encoding changed joint memory. It is not a literal port of
the upstream packed-history slicing bugs. Existing per-mask insertion still
performs successive layout updates inside `add_masks`; batching those updates
is a remaining optimization, with history equivalence to be checked.

Validation in `video_objects_parity.py`:

- 489 policy/preprocessing checks per CPU/CUDA run, covering three precisions
  and 240 placements. Recording tracker fixtures execute the actual original
  host method, including stable ties, first-state and new-state branches.
- Actual-weight original high-level add/remove methods, two birth groups,
  shared projected frame inputs, forward/reverse propagation and whole-state
  removals: 131 exact output/state tensors per model/mode, 1,048 total over both
  models, CUDA FP32/FP16/BF16-reference and CPU FP32.
- Native SAM3.1 best-fit insertion into an existing tracked state, multiple-ID
  removal, duplicate/missing removal IDs, reverse propagation and empty-state
  cleanup: 248 exact comparisons per CUDA mode, including batch-versus-sequential
  removal equivalence, plus 230 CPU FP32 resident/offloaded/paged comparisons.
  These 974 checks exercise the native history policy independently from source
  new-state grouping comparisons. The CPU report predates the additional 18
  sequential-removal checks, which passed in all three CUDA modes.

Neural comparisons use synthetic full-resolution projected features and the
existing documented original-runtime adaptations. They do not prove coherent
real-video end-to-end accuracy. All inference math uses actual model weights.
The earlier intermittent CPU full-model instability remains unresolved.

Standalone actual-weight session probes run with `PATH=/nonexistent`; they now
also exercise birth groups, best-fit reuse, 17-object groups, strict batch-removal
rejection, partial removal and empty-state deletion, including paged history.
CTest passes 23/23 CUDA-enabled and 13/13 custom-CUDA-disabled tests. The latter
still links this machine's GPU-enabled PyTorch distribution. No libpython or
libtorch_python is linked; four existing sm_75 cubins remain compilation evidence.
No GitHub Actions or Windows/Turing execution was used. Binaries are development
artifacts, not a relocatable Windows/Linux SDK.

Remaining integration includes global ID/score/confirmation metadata, phase
ordering across detector/association/hotstart/reconditioning/memory updates,
text/visual prompt and cache/user-action state, and output assembly. Codecs,
actual multi-GPU execution, portable packaging and end-to-end quality/performance
validation are still open.

## Integrated frame-update planning and local execution

`sam3/video_update.h` now composes association, hotstart, reconditioning preparation,
occlusion, global ID/score/confirmation updates, local neural execution and raw
frame-output assembly. It consumes detector results and globally ordered tracker
predictions; it does not yet run the visual detector, maintain text/visual prompt
state, collect remote ranks or implement the complete predictor service.

`VideoMetadata` retains rank-ordered IDs, actual bucket workloads, monotonically
assigned IDs, object scores, per-frame scores, occlusion history and the distinct
SAM3 host/SAM3.1 device hotstart state. `initialize_video_metadata` creates this
state for a caller-specified rank count and device. There is no maximum-object
setting or detection dropping; integer ID overflow is rejected.

`plan_video_update` is pure with respect to previous metadata and input tensors.
It preserves the original ordering and model-specific policy choices:

1. Associate detector and tracker masks, with SAM3's optional boundary filter
   applied only to new detections. SAM3.1 consumes the caller's detector keep mask.
2. Allocate monotonically increasing IDs and plan least-workload placement from
   previous object counts (SAM3) or bucket counts (SAM3.1). Hotstart uses the
   existing model-specific host/device implementations.
3. Evaluate periodic/geometric correction and prepare eligible mask edits.
   SAM3.1 updates global low logits using sign agreement and cleanup before
   occlusion; SAM3 leaves its propagated global logits unchanged here.
4. Apply recent-occlusion suppression, append new rank IDs, remove deleted IDs,
   initialize detection scores and preserve removed object-score entries at
   -10000. Update confirmation state by object ID.
5. Compact and extend SAM3.1 device metadata for the next frame. Native code
   explicitly reorders these rows to the final rank-concatenated ID order. The
   original appends new device rows globally, which can differ from this order
   with multiple ranks. A two-rank CPU/CUDA test checks the ID/first-frame mapping;
   this is bookkeeping validation, not distributed execution.

Both association and recent-occlusion counting default to FP32 independently of
neural mode; `policy_mode` allows explicit same-mode reference comparisons.
Warmup-disabled planning preserves the source skip behavior. As in the source,
metadata produced during warmup must be discarded/reset before enabling normal
hotstart; the SAM3.1 device metadata count is not extended during warmup. The
higher-level warmup lifecycle remains to be integrated. Confirmation configuration
must remain consistent with the metadata carried between frames.

`execute_video_update` applies prepared correction/preflight, globally adjusted
memory updates, local births, then removals. SAM3.1 records all IDs affected by
local correction and refreshes the local actual bucket workload. Factories and
session ownership use `video_objects.h`; shared neural cores do not copy weights.
The caller supplies global masks in `previous.object_ids()` order, executes each
rank once, and commits the resulting metadata after successful execution. There
is no cross-session rollback or cross-rank communication in this function.
Planning is separate from execution, but the neural operations preserve the
source dependency order; none of these policies depends on newly computed memory.

`finalize_video_scores` performs the source's final sigmoid-score write after
planning/execution. This deliberately overwrites a removed object's **frame**
score while its persistent **object** score remains -10000. Scalar scores are
represented as tensors, retaining their original numeric precision rather than
converting SAM3 values through Python lists.

`build_video_outputs` resizes propagated masks and cleaned new detector masks to
video resolution, thresholds at zero, and applies SAM3's geometry-triggered
output overrides. Removed IDs remain in this raw mapping, just as in the original
method; later predictor filtering uses score/confirmation/suppression metadata.
This API requires coherent mask/ID counts rather than silently padding/truncating
inconsistent SAM3.1 input metadata. It introduces no cap or prompt restriction.

Validation:

- `video_update_parity.py` runs the actual original planning and `build_outputs`
  methods for 96 workflows / 768 frames across CPU/CUDA and FP32/FP16/BF16-reference,
  both models, forward/reverse, geometry/periodic correction, confirmation,
  boundary filtering, IoM, occlusion and warmup-disabled cases. 44,868 planned/
  final metadata, correction-mask and output comparisons are exact numerically.
- These policy comparisons replace only neural correction/memory execution with
  recording/no-op fixtures. They do not prove combined neural output parity.
  Original CPU connected-components fails on an empty batch (`stack([])`); the
  reference adapter returns empty label/count images only for that case, retaining
  all nonempty source computation. CPU CUDA-transfer adaptations are reused.
- Standalone actual-weight SAM3/SAM3.1 probes with `PATH=/nonexistent` run the new
  planner/executor/output path, real neural memory encoding, births and forced
  unmatched-removal inputs, including paged history. Features and detector inputs
  are controlled/synthetic, so this is execution evidence rather than real-video
  accuracy. Final score ordering and empty-state deletion are asserted.
- CPU/CUDA native unit checks cover immutable previous maps/tensors, multi-rank
  ID-to-device-row alignment, confirmation, compaction, unfiltered raw outputs,
  final score overwrite and ID overflow rejection. CTest passes 25 CUDA-enabled
  and 14 custom-CUDA-disabled checks. No libpython/libtorch_python is linked;
  existing sm_75 cubins remain compilation-only evidence.

No GitHub Actions or Windows/Turing execution was used. The earlier intermittent
full-model CPU failure is still open. Development binaries are not a portable
SDK. Remaining work includes coherent detector/tracker feature-cache integration,
text/visual prompt and user-action state, predictor-level temporal/output filtering,
C ABI exposure of the complete high-level lifecycle, codecs, actual multi-GPU
execution, portable packaging and end-to-end quality/performance validation.

## Shared real-frame features and video detector filtering

`sam3/video_frame.h` connects one full 1008-pixel vision trunk evaluation to the
three detection pyramid levels and projected tracker features. SAM3 shares its
tracker neck for interactive and propagation calls; SAM3.1 produces both necks
from the same trunk. Modules are shared with tracking sessions. No additional
weight variants or model copies are introduced. `encode_rgb` uses the video
Pillow-compatible preprocessing path; `encode_preprocessed` accepts a frame
batch. `detect` accepts the existing unrestricted `GroundingPrompt` API,
including visual/previous-feature fields, and enables joint presence scoring
for both video models. SAM3 image inference has a different scoring default.

`postprocess_video_detections` handles every prompt independently. SAM3 compacts
retained queries after NMS; SAM3.1 retains every slot and sorts the keep flags as
in the source. The implementation distinguishes three source NMS policies:

- SAM3 greedy suppression, with stable score ordering on the CUDA path.
- SAM3.1 standard batched greedy suppression, preserving ATen's source tie order.
- SAM3.1 alternate single-frame perflib behavior, where even rejected earlier
  rows can suppress later rows. Its self-IoM denominator uses the row area due to
  the source's missing transpose; this behavior is deliberately preserved here.

NMS uses native ATen and precompiled generic-NMS CUDA kernels, without Triton.
Policy arithmetic defaults to FP32 separately from neural precision. Explicit
same-mode comparison reproduces the source's FP16 full-grid overflow behavior;
that comparison is not a recommendation to use overflowing policy arithmetic.
An empty perflib input fails in the original reshape; the native path returns
empty keep flags. No object/query cap is added.

`sam3_video_pipeline_probe` is a development integration probe. It accepts a
runtime UTF-8 text file and PPM frame manifest, runs native tokenization/text,
shared vision, video detection, propagation, update planning, correction/memory,
object birth/removal and raw mask assembly. A one-frame cache supplies tracking
sessions; ordinary forward processing asserts exactly one trunk call per frame.
Output includes packed masks and detector/tracker tensors for subsequent source
comparison. This executable currently exposes one text prompt per invocation;
the reusable detection API supports prompt batches. It is not the finished
predictor interface and does not implement prompt editing, temporal output
buffering, warmup lifecycle or codecs. Raw outputs must not be presented as final
predictor results or as proof of complete video-quality parity.

Validation in this development environment:

- `video_detection_parity.py`: 522 exact comparisons across CPU/CUDA and three
  arithmetic modes, including suppression chains, ties, empty input, overflowing
  FP16 full-grid counts and 257-query/two-prompt fixtures.
- `video_frame_parity.py`: both models, two real decoded video frames, FP32/FP16/
  BF16-reference, two text/geometry prompts and all 200 queries. All 12 feature
  tensors and seven detector outputs matched exactly in each of 12 cases. The
  reference MLP uses the existing same-mode adaptation for FP32/FP16.
- Standalone probe: both models, FP16, three real frames, runtime text `person`,
  Python absent from PATH. Each frame produces four nonempty raw masks and finite
  logits, with one trunk evaluation per frame. This is execution evidence;
  coherent source tracking-mask and final predictor parity remain to be tested.
- Existing CTest suites pass 25 CUDA-enabled / 14 custom-CUDA-disabled checks.
  Linkage excludes libpython/libtorch_python; sm_75 cubins are present. Those are
  build facts, not Windows/Turing runtime validation or a portable SDK.

Reports are `video-detection-validation.json`, `video-frame-validation.json` and
`video-pipeline-probe-validation.json`. Outputs and binaries are retained in the
private bucket. Earlier CPU instability remains open; the feature comparison uses
one CPU thread after a captured source MKL `erfinv` initialization crash.

## Correct upper-level video inputs and compare coherent source tracking

A subsequent coherent comparison exposed a preprocessing mismatch in the first
integration probe. The low-level tracker JPEG loader uses Pillow bicubic and
F32 normalization. The upper-level `io_utils` image-folder loader instead uses
Pillow bilinear, divides in F32, stores F16, and performs each normalization step
in F16. `preprocess_video_rgb` now reproduces the latter and losslessly widens
its normalized values to F32 for the shared neural encoder. The low-level API
retains its separate behavior. Codec-specific resize paths and image-only video
sessions still need explicit integration policies.

Both resize paths also follow Pillow 12.2's vertical-first ordering for extremely
tall shrinking images (see upstream [Image.resize](https://github.com/python-pillow/Pillow/blob/12.2.0/src/PIL/Image.py)).
Eleven upper-level preprocessing cases match exactly, including three real frames.
Low-level regression adds an extreme-aspect-ratio case: 53 resize/real-frame
comparisons and 10 normalization/stride cases pass. The real shared-feature test
now starts from decoded RGB, rather than supplying an already-normalized tensor;
all 12 cases again match exactly across both models and three precisions.

The probe now enables SAM3's source score-based memory selection. It also batches
text with the source's auxiliary `visual` token (plus `geometric` for SAM3.1).
These auxiliary slots preserve numerical behavior at the source text batch shape;
they do not replace or restrict the runtime text prompt.

`video_pipeline_parity.py` executes the original actual-weight raw frame engine,
including neural propagation and state updates. It can retain private reference
arrays and replay comparisons against those arrays without rebuilding the source
model. `--require-exact` gates masks/low tracking values and allows only 1e-8 JSON
score serialization rounding. Optional source batch/RoPE controls are recorded
in reports and distinguish configuration-matched diagnostics from builder defaults.
The native probe's optional `--trace` writes SAM3.1 history tensors for diagnosis.

On three real frames with text `person`, SAM3 BF16-reference is exact for every
raw mask and low tracking value; IDs and scores match. SAM3 FP16 versus the
precision-adapted source retains matching IDs/scores and differs by at most three
mask pixels per object (minimum IoU 0.9999112). Source SAM3 still forces tracker
features through BF16 during its gather path, even on one GPU; this differs from
the native FP16 path. Native FP16 versus the original BF16 neural mode is a
separate quality measurement, not an exact-parity claim. The three-frame fixture
is insufficient to establish dataset-level quality.

SAM3.1 comparison remains under investigation. With the source's one-frame batch
and complex RoPE configuration, the first two frames are exact. At frame 1's
memory update, masks/proxy scores, image features, positions and object pointers
match, but encoded memory differs; this affects frame 2. Default source batching/
real RoPE is recorded separately. Existing isolated memory comparisons passing
are not evidence that this combined path is fixed. Private state traces preserve
the failing intermediate tensors for continued investigation. No full-video
quality, predictor temporal filtering or complete user-action parity is claimed.

## Fix singleton stride in the global SAM3.1 memory update

The memory divergence above was reproduced by changing only the singleton batch
stride of a channels-last image. Source memory encoding reconstructs BCHW from
sequence features; the global native update passed raw BCHW directly. Initial
and mask-correction paths already used the source view. All three now share that
view conversion, without copying data or changing values/precision/features.

The strengthened storage/source-neural regression fails before the fix and passes
afterward: 1,356 exact comparisons across CUDA FP32/FP16/BF16-reference, plus 452
CPU FP32 comparisons with one CPU thread. Resident, offloaded and paged state and
bucket rebuild are covered. A default-four-thread CPU attempt terminated with
the previously observed null-address crash; this is not a CPU stability fix.
CTest passes 25 CUDA-enabled and 14 custom-CUDA-disabled checks.

The matching batch/RoPE three-frame SAM3.1 coherent comparison now passes the exact
gate. Against original builder-default BF16 settings, the separate minimum IoU
is 0.9994449; native FP16 vs that reference is 0.9990008 on this short fixture.
Those measurements do not establish general quality. Extending the matching
configuration to 18 frames is exact through frame 16 but fails at frame 17,
after periodic reconditioning. That remaining transition is documented in
[the investigation](VIDEO_MEMORY_DIVERGENCE.md); no complete-video parity claim
is made. Source state capture is now explicitly enabled with `--trace-state` so
ordinary comparisons do not write large internal tensors unnecessarily.

## Match SAM3.1 correction history policy in the coherent probe

The 18-frame failure above is resolved. The corrected frame's tensors were all
exact; the probe promoted edits into conditioning history while original SAM3.1
keeps already tracked frames in non-conditioning history. The probe now explicitly
sets `all_edits_conditioning=false` for SAM3.1, preserving SAM3's distinct True
setting and leaving the configurable low-level session API intact.

Both the 18-frame regression and a fresh 34-frame original-neural SAM3.1 run pass
the strict gate, including corrections at frames 16 and 32 and subsequent tracking.
Every raw mask and pre-plan tracker value matches; scores tolerate only 1e-8 JSON
rounding. This uses BF16, batch-one grounding and complex RoPE, the same diagnostic
configuration as before. No object/query/prompt limits or new weights were added.
See `video-recondition-history-exact34.json` and the detailed investigation linked
above. Predictor temporal buffering, final filtering and user-action lifecycle are
still outside this raw frame probe and remain to be implemented.

A fresh original SAM3 BF16 run also passes the same 34-frame strict raw comparison
with its existing correction policy (`video-recondition-history-sam3-exact34.json`).
Both standalone runs use `PATH=/nonexistent`, all 200 queries, four objects and
exactly 34 shared-trunk evaluations. CUDA-enabled and custom-CUDA-disabled probe
builds succeed; no Actions or physical Windows/Turing validation was performed.

## Native final output stage

The probe now connects the raw neural/update pipeline to `VideoOutputBuffer` and
`postprocess_video_output`. It reproduces delayed/batched emission, future-frame
confirmation filtering, removal snapshots, final boxes and overlap handling, and
writes `FRAME.final.*` files. Fresh 34-frame real-source comparisons pass for both
models, including exact final outputs and emission timing. This supersedes the
raw-only status above for this sequential probe, while action/prompt integration
remains incomplete. See [VIDEO_OUTPUT.md](VIDEO_OUTPUT.md) for behavior and tests.
