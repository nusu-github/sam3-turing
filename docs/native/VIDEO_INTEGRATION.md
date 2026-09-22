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
groups targets in first-encounter state order. These are **edit recipes**, not
neural session updates. The future executor must still run SAM3 preflight after
each affected candidate/state, and implement SAM3.1's affected-state behavior
(including all IDs in the changed state, not just target IDs) and subsequent
preflight. Caller masks and state are not modified during preparation.

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
