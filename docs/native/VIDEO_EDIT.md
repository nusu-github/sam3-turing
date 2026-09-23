# High-level video instance edits

`sam3/video_edit.h` connects arbitrary point/exact-mask inputs to SAM3 tracking
sessions, metadata, action history and the displayed-frame cache. It uses shared
session factories/cores and introduces no new model weights or object cap.
SAM3.1 has a separate point-edit overload and singleton extraction policy,
described below. Exact-mask high-level edits in this file currently target SAM3.

## Behavior

`edit_video_points` locates an existing local owner or creates an empty session
for a newly registered ID. It replaces prior points, supports relative/model-grid
coordinates and optional previous-memory use, obtains the original-resolution
preview, runs memory preflight, then clears nearby mask-only detector conditioning
frames for every object in that session. The default inclusive window is16frames,
as in the original SAM3 high-level predictor. Preview cleanup uses the existing
video-specific hole/sprinkle policy before selecting the requested object's mask.

`edit_video_mask` handles both new and existing IDs and runs preflight while
retaining mask-only conditioning observations. It does not apply point-only
conditioning removal or point preview cleanup to an authoritative mask.

Both paths set the user object's first/frame scores to1, clear its removed and
suppressed flags, record add/refine actions and replace only its displayed mask.
Other objects retain cached masks. Optional confirmation arrays stay aligned by
object ID across addition/removal; point edits use source status1 and the detection
count threshold, while authoritative masks set confirmed status2. Public exact
reports compare outputs, not all internal bookkeeping arrays.

Stateless first-point refinement removes the existing object's tracking/cache
state and creates a fresh session, matching the source option. Already refined
objects retain their state. Basic prompt shape, model, frame and option validation
runs before a stateless removal. The regression explicitly confirms an out-of-range
stateless request preserves IDs and action history. Runtime/OOM failures across
multiple state mutations are not an all-session transaction.

`remove_video_user_object` removes local neural state and cached masks, updates
rank IDs/object scores and optional confirmation alignment, and logs a user remove
action. Per-frame historical scores and the maximum allocated ID are retained, as
in the source. Re-registering an explicit user ID remains possible; automatic ID
allocation does not move backward. `VideoEditOptions.rank` is the rank assigned
by the caller for new objects; existing objects must execute on their owning rank.
Inter-rank mask transport and an owning distributed predictor remain future work.

## Real-neural integration comparison

The optional `sam3_video_pipeline_probe --edit-probe` regression runs a34-frame
full pass with200detector queries and four people, preserving source periodic
corrections and final output scheduling. It then executes:

1. Two-point positive/negative refinement at frame18 for an existing ID.
2. An exact rectangular mask at frame20 for a new explicit ID9000.
3. A shifted exact mask at frame21 for the existing edited ID.
4. Partial propagation over frames18through22 for the two edited IDs.
5. User removal of9000 and cache fetch at frame20.
6. Stateless first-point refinement at frame22 for another existing ID.

The scenario is fixed only as a test fixture. Public edit APIs accept caller
points, masks, IDs and valid frame indices without those limits. The fresh original
reference uses the actual high-level `add_prompt`, `add_tracker_new_mask`,
`propagate_in_video` and `remove_object` methods; neural phases are not mocked.
All34raw and34final forward outputs, and all10edit/propagation/fetch output sets,
match exactly in IDs, probabilities, boxes and binary masks with BF16/noTF32.
Raw scores retain the existing1e-8JSON-rounding tolerance. These results cover
this video/operation sequence, not dataset-wide quality or every option combination.

After adding input validation, the latest standalone binary was rerun with
`PATH=/nonexistent` and compared to the saved original tensor references. It
remains exact. `video_pipeline_parity.py --reference-cache` now supports saved
`.final.npz`, `.partial.npz` and edit-tag references, avoiding unnecessary original
model initialization when only native code changes. It records cached-reference
scope separately and never regenerates reference tensors in that mode.

Reports: `video-edit-sam3-source{,.final,.edit}.json` and
`video-edit-sam3-cached{,.final,.edit}.json`. Private full tensors reside under
`video-edit-integrated`; development binaries/headers/logs are archived at
`native-foundation/video-edit-linux-cuda13`. CTest passes28CUDA-enabled and16
custom-CUDA-disabled checks. The latter uses this environment's GPU-capable
LibTorch and is not a portable CPU distribution. The executable has no libpython
or libtorch_python linkage. sm_75 cubins are compile evidence only; no physical
Windows/Turing testing or GitHub Actions was used.

## SAM3.1 first refinement and dense history

The `Sam31VideoEditOptions` point overload uses zero cleanup area by default and
supports replacing or appending points. On the first refinement of an existing
object in a grouped session, it moves that object into a singleton session before
running the interactive head. Shared neural cores and feature providers retain
one copy of the weights. Metadata keeps its global ID order, while rank workload
is recalculated from actual bucket counts. Stateless first refinement instead
removes/recreates the object. User removal has a SAM3.1 overload as well.

`Sam31TrackingSession::extract_object` reconstructs changed dense-memory buckets
from retained image features, effective masks and object-score inputs. It remaps
object rows and slot pointers, restarts tracked-direction flags, and removes the
source object only after the singleton rebuild succeeds. History can remain on
CPU or in lossless disk archives. Point edits discard detector-only mask inputs
without deleting their encoded history, then preflight the new point output.
These are per-operation/session guarantees, not an all-session transaction.

There is a deliberate difference from the current original implementation:
`Sam3MultiplexTracking._extract_object_to_singleton_state` attempts to demux dense
`[buckets,256,72,72]` memory as slot data. Its exception handler replaces memory
with `None`. This occurs in the real four-object video; old frames0,16,32 lose
spatial memory in the extracted singleton. Native extraction retains those
observations by re-encoding them. It does not drop history to obtain parity.

The unmodified-source regression matches the point preview and frame18 output.
Frames19 and20 have140,939 and129,160 binary-mask differences respectively, while
IDs and probabilities remain identical. These are behavior differences, not
accuracy improvements: no annotated quality benchmark has been run for this
policy. The full34-frame pre-edit raw/final results remain exact.

`--sam31-rebuild-extracted-memory` is an explicit **test-only** reference adapter.
It retains original memory-encoder inputs and uses the original neural encoder to
rebuild the singleton memories that the original extraction lost. It does not
replace neural outputs with native tensors. Reports distinguish this comparison
from unmodified-source parity; cached edit references retain the adapter flag.
All four output sets (point preview and frames18,19,20) match this explicitly
corrected source exactly in IDs, probabilities, boxes and binary masks. It
rebuilds34historical memories for this fixture. The adapter operates on the actual tracker module beneath its attribute-proxy
wrapper and accepts both positional and keyword calls from the source engine.

The extraction invariant fixture selects an object from slot1 and checks pointer
movement into singleton slot0, unchanged source-bucket memory, preserved masks,
retained spatial history, restarted direction flags and mask-only annotation
removal. Resident, offloaded and paged executions pass105exact tensor comparisons
with actual memory/frame cores and full-grid synthetic features.

Reports: `video-sam31-edit-unmodified-cached{,.final,.edit}.json`,
`video-sam31-edit-corrected{,.final,.edit}.json`,
`video-sam31-edit-corrected-cached{,.final,.edit}.json`,
`video-sam31-extraction-invariants.json` and
`video-sam31-sam3-regression{,.final,.edit}.json`. The SAM3 regression still matches
all34raw/final and10edit outputs after this integration. CTest remains28/16.
Private tensors are under `video-sam31-edit-investigation`; development snapshots
are under `native-foundation/video-sam31-edit-linux-cuda13`.

## Remaining work

Repeated/new/stateless points, removal/repeated removal and reverse propagation
now have a real-video sequence; see [VIDEO_EDIT_SEQUENCE.md](VIDEO_EDIT_SEQUENCE.md)
for source repairs and the unresolved120pixel reverse-frame difference. Broaden
remaining cancellation/previous-memory/image-only paths, then integrate semantic
text/geometric/visual prompt replacement/reset. Image-only fallback, distributed ownership and
transport, complete C ABI, codecs, portable SDK, CPU runtime stability and broad
quality/performance work remain. These SAM3 edit results do not establish complete
SAM3/SAM3.1 predictor functionality.
