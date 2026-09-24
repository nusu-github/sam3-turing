# Video instance edits

`sam3/video_edit.h` connects user points and exact masks to the tracking
sessions, video metadata, action history and displayed-frame cache. The owning
predictor ([VIDEO_PREDICTOR.md](VIDEO_PREDICTOR.md)) calls these helpers; they
share the session factories and model cores, add no weights and impose no object
cap.

## SAM3 behavior

`edit_video_points` finds the object's local session, or creates an empty one for
a new ID. It replaces prior points, accepts relative or model-grid coordinates
and optional previous-memory use, produces the original-resolution preview, runs
memory preflight, then clears nearby mask-only detector conditioning frames for
every object in that session (inclusive window of 16 frames, as in the original
high-level predictor). The preview uses the video hole/sprinkle cleanup before
selecting the requested object's mask.

`edit_video_mask` handles new and existing IDs and runs preflight while keeping
mask-only conditioning observations; it skips point-only conditioning removal
and preview cleanup for an authoritative mask.

Both paths set the object's first/frame scores to 1, clear its removed and
suppressed flags, record add/refine actions and replace only its displayed mask.
Confirmation arrays stay aligned by ID: point edits use source status 1 and the
detection-count threshold, authoritative masks set confirmed status 2.

Stateless first-point refinement removes the object's tracking and cache state
and creates a fresh session, matching the source option; already refined
objects keep their state. Prompt shape, model, frame and option validation runs
before any stateless removal.

`remove_video_user_object` removes local neural state and cached masks, updates
rank IDs, object scores and confirmation alignment, and logs the removal.
Per-frame historical scores and the maximum allocated ID are kept, as in the
source: an explicit user ID can be registered again, but automatic IDs never go
backwards. New objects run on the caller-assigned `VideoEditOptions.rank`;
existing objects run on their owning rank.

## SAM3.1 behavior

- The `Sam31VideoEditOptions` point overload uses zero cleanup area by default
  and can replace or append points. Stateless first refinement removes and
  recreates the object; user removal has its own overload.
- On the first refinement of an object in a grouped session, the object is moved
  into a singleton session. `Sam31TrackingSession::extract_object` rebuilds the
  changed dense-memory buckets from retained image features, effective masks and
  object-score inputs, remaps rows and slot pointers, restarts tracked-direction
  flags, and removes the source object only after the rebuild succeeds. History
  can stay on CPU or in lossless archives. Global ID order is kept; rank workload
  is recomputed from actual bucket counts.
- Point edits discard detector-only mask inputs without deleting their encoded
  history. A point edit establishes conditioning history even when detector and
  mask corrections use `all_edits_conditioning=false`, and repeated clicks update
  that same conditioning frame and its memory (the original point path promotes
  its first refinement to conditioning).
- Removing an already removed or unknown ID still clears cached masks and records
  the action, matching the source after user or hotstart removal, without
  changing rank IDs or bucket workloads.

The native extraction deliberately differs from the current original
`Sam3MultiplexTracking._extract_object_to_singleton_state`, which tries to demux
dense `[buckets,256,72,72]` memory as per-slot data, catches the failure and sets
the memory to `None`. On the real four-object video this loses the spatial
memory of frames 0, 16 and 32 in the extracted singleton. The native runtime
re-encodes those observations instead of dropping history
([VALIDATION.md](VALIDATION.md#deliberate-differences-from-the-source)).

## Validation

`sam3_video_pipeline_probe` runs a 34-frame, 200-query pipeline on real frames
(four people, periodic corrections at frames 16 and 32) and then an edit scenario:

- `--edit-probe` (SAM3): two-point refinement, exact masks for a new and an
  existing ID, partial propagation over 18–22, user removal and fetch, stateless
  refinement. All 34 raw, 34 final and 10 edit/propagation/fetch outputs match
  the original high-level methods exactly (BF16, TF32 off).
- `--edit-probe` (SAM3.1): the point preview and frame 18 match the unmodified
  source; frames 19–20 differ because of the extraction difference above
  (140,939 and 129,160 mask pixels, identical IDs and probabilities). With the
  test-only memory-rebuild adapter all four output sets match exactly.
- `--edit-sequence-probe` (SAM3.1): repeated and appended points, reverse
  propagation, a new point object, repeated removal and stateless refinement.
  All 15 edit/propagation/fetch outputs match the original after the explicit
  source repairs listed in [VALIDATION.md](VALIDATION.md#reference-adapters);
  without the pointer repair, reverse frame 17 differs by 120 pixels.

`multiplex_point_memory_invariants.py` and the extraction invariant fixture
(105 exact tensor comparisons across resident, offloaded and paged history)
check the native policies directly. Evidence: `evidence/video-edit-sam3-*.json`,
`evidence/video-sam31-edit-*.json`, `evidence/video-sam31-sequence-*.json`,
[video-sam31-extraction-invariants.json](evidence/video-sam31-extraction-invariants.json),
[video-sam31-point-memory-invariants.json](evidence/video-sam31-point-memory-invariants.json).
