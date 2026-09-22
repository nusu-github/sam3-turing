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
