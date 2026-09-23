# Coherent SAM3.1 memory update investigation

Status: fixed for the captured divergence, 2026-09-23. The historical investigation
below explains why isolated component parity missed this integration bug. Full
predictor functionality and broader quality validation remain incomplete.

The reproducible fixture is `assets/videos/0001/{0,1,2}.jpg`, runtime text `person`,
CUDA Blackwell, BF16-reference. Native input now matches source high-level
bilinear/F16 preprocessing. The source diagnostic uses the public builder with
FA3 disabled, complex RoPE, detector batch size 1, no compilation, warmup marked
complete and one CPU initialization thread. These configuration overrides isolate
numerical differences; a separate default batch-16/real-RoPE report is retained.
No neural tracking/update phase is mocked. Source weights are the supplied
SAM3.1 checkpoint; its builder's first tracker-only partial load emits expected
missing/unexpected keys before the merged model load.

After matching the source three-slot text batch (`person`, `visual`, `geometric`):

| Comparison | Frame 0 | Frame 1 | Frame 2 |
| --- | --- | --- | --- |
| Raw output masks | Exact | Exact | Differ |
| Low tracking values before update | Empty | Exact | Max error 4.625 |
| Retained image / position | Exact | Exact | Exact |
| Retained object pointer | Exact | Exact | Differ |
| Memory encoding input masks / proxy scores | N/A | Exact | Masks differ |
| Retained encoded memory | Exact | Max error 0.0625 | Max error 0.8183594 |

At frame 1, 387,108 of 1,327,104 retained memory values differ. Memory position
matches exactly. Native trace reports zero conditioning objects on frame 1,
BF16 retained image and F32 memory input masks; source conditioning set is also
empty. Source encoder inputs are BF16 image (channels-last stride) and BF16
packed masks (contiguous NCHW).

Replaying the captured source image, packed masks, high masks, proxy scores and
empty conditioning set through the **standalone native memory module** matches
the source. The full `memory_frame` operation also matches retained source memory
exactly after the source BF16 storage cast. Both contiguous and channels-last
image layouts match in that replay. Direct replay therefore narrows the issue to
what differs in the live session update path or its retained state; it does not
resolve it. Comparing F32 pre-storage memory against BF16 retained memory alone
creates ordinary rounding differences and is not a valid failure criterion.

Private reproducibility data:

- `video-pipeline-v3/reference-sam31-bf16`: original builder-default reference.
- `video-pipeline-v6/{sam31-trace,reference-sam31-trace}`: masks, scores and state.
- `video-pipeline-v7/reference-sam31-trace`: actual encoder inputs and layouts.
- `video-pipeline-v8/sam31-trace`: native optional trace output.
- `native-foundation/video-pipeline-source-linux-cuda13`: binaries, headers,
  logs and `diagnose_video_memory.py` (standalone replay used above).

## Resolution

Recomputation inside the standalone process matched its retained state for both
the live tracking core and an independent memory module. The packed encoder
masks also matched the source exactly. The remaining difference was the image's
singleton batch stride: raw native BCHW used `[1327104,1,18432,256]`, while the
source sequence-to-BCHW view used `[256,1,18432,256]`. Both represent identical
BF16 values and are channels-last. Forcing the raw stride in Python reproduced
all 387,108 differing memory values; forcing the source stride produced zero
differences. Ordinary `contiguous(memory_format=channels_last)` was insufficient
because it preserves a tensor already considered contiguous in that format.

`Sam31TrackingFrame::update_memory` now uses the same sequence-to-BCHW view as
its existing initial/corrected memory paths. Those three callers share a helper.
The change adjusts a view; it does not copy image data, change values, lower
precision, limit prompts/objects or add weights.

The strengthened `video_memory_storage.py` uses channels-last shared features
and compares the immediate global rewrite with the actual source neural encoder,
before a later layout rebuild can hide the defect. It fails before the fix
(345,206 mismatches in the synthetic regression) and passes afterward: 452 exact
comparisons per CUDA precision mode, including resident/offloaded/paged histories
and changed bucket layouts. The three-frame coherent comparison now passes
`--require-exact`: all masks and low tracking values match, and scores agree within
the documented JSON serialization tolerance. Builder-default batched/real-RoPE
and native-FP16 versus source-BF16 reports remain separate numerical comparisons;
they are not claimed byte-identical or dataset-level quality evidence.

Diagnostic replays and fixed output data are saved privately in
`native-foundation/video-memory-stride-linux-cuda13` and
`video-memory-stride-fixed`. Historical failing tensors remain available under
the paths above. General CPU runtime instability is a separate unresolved issue.

## Longer sequence exposes a separate remaining transition

Extending the coherent fixture to 18 consecutive source frames, with the same
matched BF16/batch/RoPE configuration, gives exact masks and low tracking values
for frames 0 through 16. Frame 17 differs after the periodic frame-16
reconditioning step: mask differences `[53,29,826,70]`, maximum low-logit error
7.125. All four IDs and first-detection scores remain aligned. The strict
18-frame comparison fails and is retained as `video-memory-stride-sequence18.json`.
This does not invalidate the reproduced/fixed stride defect, but it prevents a
claim of complete coherent video parity. The next investigation is the retained
frame-16 state and reconditioning transition, using private
`video-sequence-18/{native-sam31,reference-sam31}`. Do not loosen the gate or
skip periodic corrections to make the longer case pass.

## Resolved: correction history category in the coherent probe

The captured states after frame 16 match in every compared value: low masks,
object pointers, shared images/positions, encoded memory/positions, global
1152-square memory masks and proxy scores. The difference was the history
category, not the reconditioning arithmetic. The low-level native session defaults
to promoting edits into conditioning history. The original SAM3.1 tracker defaults
`add_all_frames_to_correct_as_cond=False`, keeping an edit of an already tracked
frame in non-conditioning history. Consequently frame 17 selected a different
memory set in the probe. Original SAM3 explicitly sets this option to True in
`sam3_tracking_predictor.py`; it must keep its separate setting.

The coherent probe now explicitly selects False for SAM3.1 only. Low-level session
configuration remains available for interactive callers. The exact 18-frame gate
now passes, and a fresh 34-frame original-neural run passes every raw mask/low-value
comparison through two periodic corrections, at frames 16 and 32. All 200 detector
queries, four tracked people and one shared visual trunk evaluation per frame
remain enabled. Scores use the existing 1e-8 JSON-rounding tolerance. Evidence:
`video-recondition-state-diagnosis.json`, `video-recondition-history-exact18.json`
and `video-recondition-history-exact34.json`. BF16, batch-one grounding and complex
RoPE are the explicit matching configuration; this is one video fixture, not a
broad quality benchmark or complete predictor parity.

Source `--trace-state --trace-frames 15 16 17` now limits saved internal traces
without skipping any inference frame. Full before/after outputs, source outputs
and traces are retained privately in `video-recondition-trace`,
`video-recondition-fixed` and `video-sequence-34`.
