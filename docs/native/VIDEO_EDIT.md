# SAM3 high-level point and exact-mask edits

`sam3/video_edit.h` connects arbitrary point/exact-mask inputs to SAM3 tracking
sessions, metadata, action history and the displayed-frame cache. It uses shared
session factories/cores and introduces no new model weights or object cap.
SAM3.1 singleton extraction and its distinct edit-history rules are not yet
integrated by these entry points; its low-level edit API remains available.

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

## Remaining work

Next integrate SAM3.1 first-refinement singleton extraction, multiplex history
reconstruction and its input/consolidation rules, then semantic text/geometric/
visual prompt replacement/reset. Image-only fallback, distributed ownership and
transport, complete C ABI, codecs, portable SDK, CPU runtime stability and broad
quality/performance work remain. These SAM3 edit results do not establish complete
SAM3/SAM3.1 predictor functionality.
