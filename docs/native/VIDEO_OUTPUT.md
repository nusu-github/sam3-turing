# Video output buffering and final postprocessing

`sam3/video_output.h` exposes the native predictor output stage independently of
neural inference, with no Python/Triton dependency and no object count limit.
It is integrated into `sam3_video_pipeline_probe` after real detection/tracking,
metadata updates and periodic reconditioning. Raw diagnostic outputs remain
available; additional `FRAME.final.*` files contain final IDs, probabilities,
normalized xywh boxes, packed binary masks and the frame at which output became
available. This development probe is not yet the complete interactive predictor.

## Preserved source behavior

- Masks are ordered by sorted object ID. Empty masks and suppressed/removed/
  unconfirmed IDs are excluded before boxes are computed. Boxes use inclusive
  pixel maxima, so a single-pixel mask has zero box extent.
- Non-overlap resolution runs after boxes, using tracker probabilities; missing
  scores default to zero. Equal positive scores choose the first sorted ID.
  Multiple masks with nonpositive scores lose their pixels; a singleton bypasses
  overlap resolution. Rows are not filtered again after overlap suppression.
- Cached masks exclude hidden IDs but retain empty masks and the original
  pre-overlap masks. Optional production centers use final masks. Frame statistics
  pass through the library unchanged.
- Hotstart delay buffers outputs. Removed IDs accumulate only when delay is
  enabled, matching the source. Confirmation looks forward/backward by threshold
  minus one, clamped to video bounds. A future status that has not been observed
  remains absent; the implementation does not invent a confirmation.
- Each frame entering a postprocessing batch snapshots the removal set at that
  moment. Confirmation is looked up when that batch is actually processed.
  Batch size affects emission timing, without capping frames or objects.
- End-frame arrival drains the remaining buffers. `cancel()` drops pending output
  and closes the buffer; `reset()` starts a fresh invocation. Caller-owned tensors
  are cloned when enqueued so later mutation cannot alter delayed results.

`VideoOutputBufferOptions` supplies dimensions even for entirely empty frames,
video bounds, direction, delay, confirmation threshold and postprocessing batch
size. The integrated probe uses source defaults: delay 15, confirmation threshold
3, batch size 1 for SAM3 and 16 for SAM3.1. It retains the distinct source
correction-history settings established by the previous investigation.

Mask nonemptiness is reduced together, using one CPU transfer for the row flags
instead of a GPU synchronization for every object. Bounding boxes reuse the
existing axis-reduction implementation. Final tensors stay on the mask device
except IDs/probabilities; the application chooses when to transfer them. The
current implementation processes tensor postprocessing per emitted frame; it
matches source batch timing without the source's fixed-size CPU staging buffer.
This is not a claim of equivalent batching performance.

## Verification

`native/tests/video_output_parity.py` invokes the actual source
`Sam3VideoInference`/`Sam3MultiplexTracking` output generators, cache exclusion
logic and final postprocessors. Only the raw neural-frame supplier is replaced by
synthetic inputs in this focused policy test. There are 96 workflows per run:
both models, forward/reverse, delays 0/1/3/15, confirmation thresholds 1/3/20 and
SAM3.1 postprocessing batches 1/4/16. Cases contain empty frames, empty masks,
negative/zero/tied tracker scores, later removals and suppressed/unconfirmed IDs.
Each run has 13,144 exact checks, including emission frame, hidden IDs, cached
masks, output masks/IDs/scores/boxes and optional centers:

- `video-output-parity.json`: CUDA, source perflib boxes.
- `video-output-cpu-parity.json`: CPU, source perflib boxes, one CPU thread.
- `video-output-torchvision-parity.json`: CUDA, source torchvision boxes.

The C++ test separately checks mutation isolation, cancellation/reset/closed
state, frame statistics and 257 overlapping objects without an object cap.
CTest passes 27 CUDA-enabled and 15 custom-CUDA-disabled tests. The latter build
still uses the environment's GPU-capable LibTorch; it is not a portable CPU SDK.

`video_pipeline_parity.py --final-output` executes every actual original neural
frame, then replays those raw results through the original base output generator
and postprocessor. This isolates output scheduling without rerunning the neural
model; it does not mock neural outputs. Fresh 34-frame BF16 runs of SAM3 and SAM3.1
match the standalone C++ pipeline exactly for final masks, IDs, scores, boxes and
emission times, as well as the prior raw masks/low tracker values. SAM3.1 uses
matching batch-one grounding and complex RoPE for neural comparison; its output
batch size is the builder default 16. It emits its first batch after frame 29.
Reports: `video-output-{sam3,sam31}-source.json` and corresponding `.final.json`.
All 200 queries, four tracked people, two periodic corrections and exactly one
shared visual-trunk evaluation per frame remain enabled. Both native runs used
`PATH=/nonexistent`. This fixture does not establish dataset-wide quality.

Full references and native outputs are private under `video-output-integrated`;
the runtime snapshot is `native-foundation/video-output-linux-cuda13`. Public
reports contain only comparison metadata, not model weights/tensor outputs.
The shared library retains sm_75 cubins as compile evidence, with no physical
Turing or Windows validation and no GitHub Actions. The native executable does
not link libpython or libtorch_python.

## Remaining integration

The full action-history router, changed text/geometric/visual prompts, partial
propagation/fetch and cache invalidation must be integrated above this stage.
Existing low-level tracking edits/removal/reset/reverse are available but do not
alone constitute the source high-level predictor. Complete C ABI exposure,
codecs, multi-GPU communication, portable SDK packaging, general CPU runtime
stability and broad quality/performance validation remain open.
