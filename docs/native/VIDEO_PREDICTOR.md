# Owning C++ video predictor (development API)

`sam3::VideoPredictor` owns a local video's semantic prompt, neural sessions,
object metadata, action history, displayed-frame cache and output scheduling.
A caller provides decoded RGB frames and receives IDs, scores, boxes and masks;
it no longer has to assemble the detector/tracker update loop itself.

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

The frame provider returns U8 RGB `[3,H,W]`. Semantic inputs can contain UTF-8
text, arbitrary normalized xywh boxes and labels, or caller-encoded exemplar
tokens. Replacing a semantic prompt resets observations, IDs, caches and action
history while retaining the shared vision/detector/tracker modules. Text weights
are loaded temporarily and only the encoded tokens are retained. One cached
frame's shared trunk features feed both detector and tracker necks; the fixture's
initial preview plus frames0–3 requires four trunk evaluations, not five.

Existing point edits, SAM3 exact-mask edits, user removal, cached fetch, partial
propagation, full forward/reverse scheduling and reset are available through this
owner. Calls must be exclusive, except `cancel()` can be called from another
thread. Cancellation is checked between frames; it does not interrupt a running
CUDA kernel. Callback cancellation discards pending buffered emissions. Requested
centers are returned on preview/fetch/partial/full paths. Semantic shape/range
validation precedes reset; neural execution is not an all-session transaction.
Custom cleanup/confirmation settings are passed through to edit helpers.

Point-only initialization seeds an empty displayed-frame cache so that a newly
added SAM3.1 object is returned. This deliberately avoids the original missing-cache
merge behavior. The regression checks that object/cache initialization, removal,
invalid semantic input state preservation and callback cancellation work; these
extra checks are native invariants, not original-output parity claims.

## Conditioning correction

Revisiting an initial SAM3.1 prompt frame and then running its periodic detector
correction previously moved the sole conditioning frame into tracked history when
`all_edits_conditioning=false`. The next frame failed with no annotation. Mask edits
and detector reconditioning now preserve an already-conditioning frame. Corrections
on new tracked frames remain non-conditioning. Full-grid synthetic neural tests
cover both routes, repeated point-memory refresh, and subsequent propagation.

## Actual video comparison

`sam3_video_predictor_probe` exercises semantic replacement and reset using actual
full1008 neural execution, all200queries, arbitrary library inputs and shared
modular weights. Its fixed person/box/point sequence is only a regression fixture:

```sh
env PATH=/nonexistent build/native/sam3_video_predictor_probe \
  STORE sam3 cuda bf16_reference FRAMES.txt BPE.gz OUTPUT
```

The original comparison is `native/tests/video_predictor_parity.py`. Both sides
use BF16/noTF32 on Blackwell; SAM3.1 source grounding uses batch1 and complex RoPE.
This is not the original default batch16/real-RoPE configuration.

- **SAM3:** all11 output sets match exactly in IDs, probabilities, boxes and binary
  masks, including text propagation, box-only replacement/propagation, combined
  negative-box/text replacement, fetch and reset. Live comparison also matches
  emission timing. The latest native build matches retained original tensors;
  older SAM3 cache files lack timing metadata, so that cached rerun does not
  independently recheck timing. Center shape and point-only/cancel invariants pass.
- **SAM3.1:** all11 output sets now match exactly in IDs, probabilities, boxes,
  binary masks and emission timing against a reference with an explicitly repaired
  forward detector bound. The original generator includes `start+max_steps`, while
  its batched detector excludes that endpoint and raises `IndexError` in
  `_batch_find_inputs`. `--sam31-inclusive-grounding-bound` adds one only to the
  detector bound; generator range and output scheduling stay unchanged. This is
  an adapted-reference result, not unmodified default-configuration parity.

### Precision audit and output visibility correction

The earlier SAM3.1 reports incorrectly described both sides as TF32-disabled.
`Sam3MultiplexVideoPredictor.__init__` enables TF32 after the previous test driver's
initial configuration. Layer traces showed identical first-layer attention and
normalization up to the FP32 FFN, where the results diverged. A controlled replay
with native TF32 enabled matches all84captured decoder-layer tensors from two
frames exactly; disabling it reproduces that first divergence. Configuring both
TF32 flags **after construction**, recording them, and asserting them before each
frame makes ten previously differing output sets exact without changing native
neural code. Historical non-exact reports are retained with `precision_audit`;
they must not be interpreted as equal-precision comparisons. New cached-reference
metadata records the actual post-construction flags.

The remaining box-track frame was hidden only by native output orchestration.
The original computes a GPU `to_suppress_mask` candidate, but its planning code
never publishes that candidate into `rank0_metadata["suppressed_obj_ids"]`. Its
predictor filters by that published host set. The native owner and coherent probe
now follow this output policy; the low-level GPU candidate calculation remains
available separately. On the regression frame, a native keep-alive counter of0
no longer hides ID0; its23058mask pixels match the source exactly. This also keeps
its cached output available for subsequent fetch/refinement.

A separate original behavior remains relevant: batched detection reconstructs
geometry from `find_inputs`, while semantic add_prompt stores boxes in a per-frame
map. Native retains that geometry. The optional test-only
`--sam31-preserve-batched-geometry` passes stored geometry to the original prompt
builder for investigation. **It is not enabled in the primary11-output exact
comparison.** Internal keep-alive counters can consequently differ (native0 versus
original-2 in this fixture) even though displayed outputs match. The11-output
result does not establish state equivalence. A separate extended fixture supplies
`BOX_STEPS=5` to the native probe and `--box-steps 5` to the reference, tracking the
box from frame1through6. With both explicit source adapters enabled, all15output
sets match exactly, including masks and emission timing. The geometry adapter's
actual use is recorded and asserted. This remains a finite fixture, not proof of
arbitrary trajectories, all states or dataset-wide quality.
The prior120-pixel reverse-edit discrepancy is separate and remains unresolved.

The existing34-frame SAM3.1 pipeline also remains exact in raw masks, tracker
values, scores, final displayed outputs and emission timing against retained
original references after the conditioning correction. That text-only baseline
does not exercise this owner's new semantic replacements.

## Remaining scope and persistence

This is a development C++ API, not the finished full-function distribution.
SAM3.1 high-level exact-mask orchestration is explicitly rejected here; its low-level
mask session API exists. Image-only fallback policy, complete video C ABI, codecs,
multi-GPU transport, resumed-session checkpoints, portable SDK packaging and broader
quality/performance remain. Caller-encoded visual tokens and reverse scheduling
are available through the owner but are not newly compared by this semantic fixture.
Single-model cores are shared within one owner; independent owners currently load
independent cores. No Python/Triton process is needed by the standalone executable.
The current binaries still depend on this development environment's LibTorch.

CTest passes28 CUDA-enabled and16 custom-CUDA-disabled tests (the latter build
still links the installed CUDA-capable LibTorch). Actual Windows/Turing execution
is left to the user. sm75 cubins and DLL export/copy plumbing are build evidence,
not hardware validation. No GitHub Actions are used. Code is in the development
branch; weights, reference tensors and Linux build snapshots stay in the private
Hugging Face bucket named in the project onboarding notes.
