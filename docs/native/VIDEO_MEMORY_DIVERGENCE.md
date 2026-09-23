# Coherent SAM3.1 memory update investigation

Status: open, 2026-09-23. Do not treat isolated component parity as proof that this
combined pipeline is correct.

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

Next useful check: recompute memory immediately inside the native probe using
both the live tracking core and an independent memory module, preserving the
live layout/options/bucket matrix, then compare before and after storage/paging.
Avoid changing inference policy to make this fixture pass. Broader source video,
user-action and final predictor output validation remain required.
