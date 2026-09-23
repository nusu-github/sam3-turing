# Image sources in the owning C++ predictor

`VideoPredictorOptions::image_only=true` selects the original predictor's explicit
image-source policy. It requires exactly one frame. A one-frame video keeps
`image_only=false`; frame count alone cannot distinguish the two input types.
Both use the same cores and modular weights. There is no image-specific weight
package, fixed prompt, reduced image resolution or reduced query count.

For SAM3.1, image mode uses `image_detection_threshold` (default 0.5) instead of
the video birth threshold (default 0.65). Other filtering and neural settings
remain configurable through the existing options. The source distinguishes these
paths in `sam3_multiplex_tracking.py:init_state` and
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

The owner retains each detector-created object's initial video-resolution binary
input mask on the CPU. Clearing all point prompts re-feeds that retained mask,
clears the edited frame's point state, and rebuilds conditioning memory. Repeated
clear operations keep the annotation usable. Stateless first refinement can
recreate the object before restoration. An explicit mask replaces its restoration
baseline; removal, semantic replacement and reset discard retained baselines.
Objects created only from points have no original mask and use ordinary empty
point inference. Empty edits with a box also stay on the ordinary point/box path.

`add_mask` now supports both models through the owner and SAM3.1 edit helper.
It retains authoritative mask inputs and marks the object confirmed. Configurable
point cleanup does not remove components from an authoritative mask. As in the
existing output API, `cached_masks` keeps masks before final overlap arbitration;
displayed masks may lose pixels to another object's higher or tied tracker score.
An authoritative input is not a request to disable the source overlap policy.

## Validation and source defects

`sam3_image_predictor_probe` accepts runtime image pixels, text and source kind:

```sh
env PATH=/nonexistent build/native/sam3_image_predictor_probe \
  STORE sam3.1 cuda bf16_reference IMAGE.ppm BPE.gz PROMPT.txt OUTPUT image
```

`video` selects a one-frame video. `image-edits` additionally runs a SAM3.1
regression sequence: stateless empty edit, point edit, two restorations, mask-only
object creation, point refinement of that mask, restoration, mask replacement,
removal and semantic reset. These fixed actions are test inputs, not API limits.
They reuse a single trunk evaluation until reset. PPM is only the development
probe's input format; the library accepts decoded RGB and still needs integrated
codecs for the final distribution.

`native/tests/image_predictor_parity.py` compares the semantic preview with an
unmodified original predictor. BF16/noTF32 is configured after construction;
complex RoPE and grounding batch 1 remain the reference configuration. This is not
the original default batch 16/real-RoPE deployment configuration.

Three semantic previews match the unmodified reference exactly in IDs,
probabilities, boxes and binary masks: person/image (four objects), food/image
(four objects), and the same food image as a one-frame video (zero objects).
The food scores are 0.5151515, 0.61676645, 0.5241636 and 0.5816993, so this fixture
actually exercises the 0.5 versus 0.65 birth-threshold distinction. The empty video
output is not produced by a reduced query count; all 200 queries still execute.
These are finite Blackwell BF16 fixtures, not dataset-wide quality evidence.

The source's own image empty-point path calls `tracker.add_new_mask`, but
`Sam3VideoTrackingMultiplexDemo` has no such method. Actual direct-empty and
point-then-empty calls both raise `AttributeError` in the retained person fixture.
The test records those errors separately without a source repair. Native
restoration is an implementation of the intended behavior and is checked by
invariants. The native restored input mask also matches the actual original
detector-input mask exactly before overlap arbitration. It is **not** claimed to match outputs of those failed source calls.
SAM3.1 high-level mask editing similarly integrates its native low-level mask
session support; no new upstream high-level mask-output parity is claimed.

Both model variants retain all 11 cached semantic video comparisons exactly, and
mask-only initialization now propagates through three frames in each model.
CTest passes 28 CUDA-enabled and 16 custom-CUDA-disabled tests. The latter still
links this environment's CUDA-capable LibTorch. See
[validation report](image-predictor-validation.json) for scope and reference settings.

This remains a development C++ API. The owning C ABI is available in [PREDICTOR_C_API.md](PREDICTOR_C_API.md). Integrated codecs,
multi-GPU transport, broader quality/performance work remain.
The earlier 120-pixel reverse-edit discrepancy is separate and unresolved.
No GitHub Actions or Windows/Turing physical tests are used.

The standalone development SDK is documented in [SDK.md](SDK.md).
