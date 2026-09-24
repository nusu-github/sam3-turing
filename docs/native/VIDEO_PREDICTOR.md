# Owning C++ predictor

`sam3::VideoPredictor` (`sam3/video_predictor.h`) owns a local video's or image's
semantic prompt, neural sessions, object metadata, action history, displayed-mask
cache and output scheduling. The caller supplies decoded RGB frames and receives
IDs, scores, boxes and masks. The same class serves explicit image mode. The C
ABI wrapper is `sam3_predictor` ([PREDICTOR_C_API.md](PREDICTOR_C_API.md)).

```cpp
#include <sam3/video_predictor.h>
auto options = sam3::video_predictor_defaults(sam3::AssociationPolicy::Sam31);
options.mode = "fp16"; // fp32 and bf16_reference also supported
sam3::VideoPredictor video(store, vocabulary, read_rgb_frame,
                           frame_count, height, width, at::Device("cuda"), options);
sam3::VideoSemanticPrompt prompt;
prompt.text = user_utf8_text;
auto preview = video.add_prompt(frame_index, prompt);
sam3::VideoPredictorPropagation request;
request.start = frame_index;
video.propagate(request, [](int64_t frame, const sam3::VideoOutput& output) {
  display(frame, output);
  return true; // false cancels this invocation
});
```

## Prompts and lifecycle

- The frame provider returns U8 RGB `[3,H,W]`. Semantic prompts contain UTF-8
  text, arbitrary normalized xywh boxes with labels, or caller-encoded exemplar
  tokens.
- `add_prompt` replaces the semantic prompt: observations, IDs, caches and action
  history reset while the shared vision/detector/tracker modules stay loaded.
  Text weights are loaded temporarily; only the encoded tokens are kept, and an
  identical text reuses the previous encoding ([PERFORMANCE.md](PERFORMANCE.md#semantic-text-reuse)).
- One cached frame's trunk features feed both the detector and tracker necks.
- Point edits, exact-mask edits, user removal, cached fetch, partial propagation,
  forward/reverse propagation and reset are available on the owner
  ([VIDEO_EDIT.md](VIDEO_EDIT.md), [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md)).
- Calls must be exclusive, except `cancel()`, which may be called from another
  thread. Cancellation is checked between frames and does not interrupt a
  running kernel; callback cancellation discards pending buffered emissions.
- Requested centers are returned on preview, fetch, partial and full paths.
  Semantic shape/range validation happens before reset. Neural execution is not
  an all-session transaction.
- Point-only initialization seeds an empty displayed-frame cache so that a new
  SAM3.1 object is returned (the original source's missing-cache merge would
  drop it).
- Revisiting a SAM3.1 prompt frame and then running its periodic detector
  correction keeps an already-conditioning frame in conditioning history
  (`all_edits_conditioning=false` would otherwise demote the only annotation).
  Corrections on newly tracked frames stay non-conditioning.

Other owner settings: tracking devices and parallel ranks
([VIDEO_MULTIDEVICE.md](VIDEO_MULTIDEVICE.md)), displayed-mask storage
([OUTPUT_CACHE.md](OUTPUT_CACHE.md)) and video input preprocessing
([VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md)).

Single-model cores are shared within an owner. The `VideoPredictorModules`
constructor shares immutable cores across owners; the C API does this
automatically for children of one context. Features and state stay per owner.

## Image mode

`VideoPredictorOptions::image_only=true` selects the original predictor's
image-source policy and requires exactly one frame. A one-frame video keeps
`image_only=false`: frame count alone cannot distinguish the two. Both use the
same cores and weights at full resolution with all queries. For SAM3.1, image
mode uses `image_detection_threshold` (default 0.5) instead of the video birth
threshold (default 0.65), as in `sam3_multiplex_tracking.py:init_state` and
`sam3_multiplex_base.py:run_tracker_update_planning_phase`.

```cpp
auto options = sam3::video_predictor_defaults(sam3::AssociationPolicy::Sam31);
options.image_only = true;
sam3::VideoPredictor image(store, vocabulary, read_rgb, 1, height, width,
                          at::Device("cuda"), options);
sam3::VideoSemanticPrompt prompt;
prompt.text = user_text;
auto detection = image.add_prompt(0, prompt);
auto refinement = image.add_points(0, object_id, user_points);
sam3::TrackingPoints cleared;
cleared.points = at::empty({0, 2}, at::kFloat);
cleared.labels = at::empty({0}, at::kLong);
auto restored = image.add_points(0, object_id, cleared);
auto mask_edit = image.add_mask(0, object_id, user_mask);
```

The owner keeps each detector-created object's initial video-resolution binary
mask on the CPU. Clearing all point prompts re-feeds that mask, clears the frame's
point state and rebuilds conditioning memory; repeated clears keep the annotation
usable. An explicit mask replaces the restoration baseline; removal, semantic
replacement and reset discard it. Objects created only from points have no
baseline and use ordinary empty-point inference.

The original source cannot run this path: its empty-point branch calls
`tracker.add_new_mask`, which `Sam3VideoTrackingMultiplexDemo` does not define,
so direct-empty and point-then-empty calls raise `AttributeError`. The native
behavior implements the intended restoration and is checked by invariants; the
restored input mask equals the original detector input mask exactly.

`add_mask` works for both models, retains the mask as an authoritative input and
marks the object confirmed; point cleanup never removes components from it. As in
video, `cached_masks` hold masks before final overlap arbitration, so a displayed
mask can still lose pixels to another object's higher tracker score.

## Probes and validation

`sam3_video_predictor_probe` (video) and `sam3_image_predictor_probe` (image,
one-frame video or the `image-edits` sequence) run the owner on real frames with
all 200 queries and runtime text; their fixed prompt sequences are regression
fixtures, not API limits:

```sh
env PATH=/nonexistent sam3_video_predictor_probe STORE sam3 cuda bf16_reference FRAMES.txt BPE.gz OUTPUT
env PATH=/nonexistent sam3_image_predictor_probe STORE sam3.1 cuda bf16_reference IMAGE.ppm BPE.gz PROMPT.txt OUTPUT image
```

Against the original predictor (`native/tests/video_predictor_parity.py`,
`image_predictor_parity.py`) all 11 semantic-lifecycle output sets match exactly
for both models, the extended SAM3.1 box track (15 outputs) matches, and the three
image/one-frame-video previews match an unmodified reference (including the 0.5
versus 0.65 birth threshold). The reference settings and the explicit source
repairs these comparisons need are listed in [VALIDATION.md](VALIDATION.md).
Evidence: `evidence/video-predictor-*.json`,
[image-predictor-validation.json](evidence/image-predictor-validation.json).
