# SAM3.1 repeated point edits and object lifecycle

Point edits now establish conditioning history even when detector/mask corrections
use `all_edits_conditioning=false`. Repeated clicks update that same conditioning
frame and its encoded memory. Previously the native session placed a point edit
on an already tracked frame into non-conditioning history; the original point
interaction path promotes its first refinement to conditioning. Detector periodic
correction policy remains unchanged.

SAM3.1 high-level user removal also accepts an already removed/unknown ID. It
still clears cached masks and records the requested action, matching original
behavior after user removal or automatic hotstart removal. It does not change
rank IDs or bucket workloads when the ID is already absent.

## Real-video sequence and current result

`sam3_video_pipeline_probe --edit-sequence-probe` runs the full34-frame,200-query
pipeline, then the following sequence with actual shared neural cores:

1. Existing-object point edit at18 and forward propagation18–20.
2. First point edit at tracked frame19, then append a third point on19.
3. Reverse propagation to18 and17.
4. Add explicit ID9000 by point input at20 and propagate it over20–22.
5. Remove9000, fetch20, repeat the removal, and fetch20 again.
6. Stateless first refinement of another existing object at22.

The fixture uses four people before the new ID. API prompts, IDs, frames and
object counts are unrestricted; these constants are regression inputs only.
The standalone executable ran with `PATH=/nonexistent` in BF16/noTF32 on Blackwell.
SAM3.1 original grounding uses batch1/complexRoPE for this comparison.

All34pre-edit raw/final outputs remain exact. Of15edit/propagation/fetch output
sets,14match in IDs, probabilities, boxes and masks. Frame17reverse has120binary
mask differences after the fixes, with all IDs/probabilities/boxes still equal.
Per-object IoU is0.9991406868,1,0.9995450869,1. The second affected displayed mask
comes from final overlap resolution with the edited object. **The reverse-frame
pixel discrepancy remains unresolved.** This result does not establish bit-exact
full interactivity or dataset-wide quality.

The earlier native/reference comparison had192different mask pixels on17.
Promoting point conditioning and retaining the latest original encoded edit
reduces the difference to120. No output tolerance is used to label that case
exact: the JSON reports explicitly retain `exact: false`.

## Explicit reference repairs and evidence

The original high-level sequence is not executable unmodified for all operations.
The test driver keeps each adaptation opt-in and records it in the report and
cached-reference metadata:

- `--sam31-rebuild-extracted-memory`: reconstruct dense memory lost by initial
  grouped-object extraction, as documented in [VIDEO_EDIT.md](VIDEO_EDIT.md).
- `--sam31-enable-repeat-refinement`: set original `iter_use_prev_mask_pred=true`.
  Its default false makes the second edit on frame19 fail at an assertion. The
  failure was reproduced with the real video and retained separately.
- `--sam31-default-remove-frame`: supply the omitted `frame_idx=None` in the
  internal stateless removal call. The original method requires that argument;
  the call/signature mismatch is recorded as a static source audit, not a separate
  unadapted neural runtime test.
- `--sam31-refresh-refined-memory`: keep the original newly consolidated neural
  output when an older conditioning key and a new non-conditioning edit collide.
  Original preflight otherwise deletes the newly encoded output while its mask
  has already updated through aliases. On frame19after the second edit, the
  original mask matches native but stale memory differs by up to6.9375. Retaining
  the latest original output makes mask and memory both exact.
- `--sam31-preserve-singleton-history`: preserve untouched spatial tensors and
  pointers across the original remove/re-add of an already-singleton object.
  Original bookkeeping unnecessarily muxes `[1,256,72,72]` spatial history into
  `[1,16,256,72,72]`. It also rounds F32 positional encodings to BF16;66observed
  history transitions show a maximum absolute rounding error of0.00195235014.
  Preserving these tensors does **not** remove the remaining120pixel discrepancy.

These adapters retain original neural computations; they do not substitute native
predictions. They are research/test code, never part of the Python-free runtime.
The number and scope of these repairs matter when interpreting parity: this is
comparison against explicitly repaired original state handling, not unmodified
original high-level behavior. Raw and final pre-edit comparisons are unaffected.

## Other checks and persistence

`multiplex_point_memory_invariants.py` uses full-grid synthetic features and actual
frame/memory cores to check point conditioning with detector corrections kept
non-conditioning, point accumulation, changed latest mask/memory, and unchanged
prior-frame memory. The native routing/cache test checks repeated unknown removal
including stale cached masks. Original adapted low-level points, refinement and
repeated detector correction remain exact over153tensor outputs in BF16.
CTest passes28CUDA-enabled and16custom-CUDA-disabled checks; after extending
the repeated-removal case, the affected test was rerun successfully in both builds.

Public reports are `video-sam31-sequence-preserved{,.final,.edit,.history-casts}.json`,
`video-sam31-sequence-refreshed{,.final,.edit}.json`,
`video-sam31-sequence-source-issues.json`,
`video-sam31-sequence-session-regression.json`, and
`video-sam31-point-memory-invariants.json`. Full tensors, intermediate baseline
results and logs are private under `video-sam31-sequence`; development snapshots
are under `native-foundation/video-sam31-sequence-linux-cuda13`.

No GitHub Actions or physical Windows/Turing execution was used. This remains a
development runtime, not a portable SDK. Semantic text/geometry/visual prompt
replacement/reset, image-only fallback, complete C ABI, codecs, distributed
transport, portable packaging, CPU stability and broad quality/performance work
remain, together with the reverse-frame discrepancy above.
