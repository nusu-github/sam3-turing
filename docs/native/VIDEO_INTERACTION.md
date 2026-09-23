# Native action routing and partial video propagation

`sam3/video_interaction.h` connects the existing tracking sessions to action
history and displayed-frame caches. It adds source-compatible route selection,
propagation bounds, selected-object partial tracking, merge/fetch and cache reset.
The real neural coherent probe exercises these operations after its full pass.
This does not yet implement the complete semantic prompt and instance-edit API.

## State and execution

`VideoAction` represents add/remove/refine and full/partial/fetch/cancel events.
`route_video_actions` follows the source history heuristic: initial full pass,
selected IDs since the preceding propagation, boundary/two-pass fetch, and
SAM3.1 cancellation recovery. IDs deduplicated by a Python set are sorted in C++;
the selected set is identical and local sessions determine execution order.
`video_object_was_refined` recognizes both add and refine history. SAM3's existing
forced-tracker extension selects every active ID, or full propagation if empty.
SAM3's source history does not accept cancel events; its running sessions/output
buffer can still be cancelled. The new state class rejects such a history event.

`video_processing_range` resolves the earliest initialized frame when no start
is supplied and rejects propagation without any prompt/start. Forward includes
the start; reverse begins one frame before it. End bounds are inclusive, a reverse
zero-step request is empty, and arithmetic avoids overflowing large step counts.

`VideoInteractionState` owns action history and the filtered, pre-overlap masks
produced by the output stage. Recording clones masks for mutation isolation.
Fetch uses current object scores and per-frame tracker scores. Partial merge
upsamples selected low masks in FP32 with bilinear interpolation, replaces just
those cached IDs, applies suppression and the existing final output stage, and
updates the cache. Other IDs retain their saved masks. Per-frame tracker scores
remain raw tracker outputs on this source partial path; unlike the full path,
there is deliberately no additional sigmoid. SAM3 stores them as FP32; SAM3.1
preserves the incoming tensor precision. `forget_object` removes an ID from all
cached frames; `reset` clears cache and actions.

There are source-specific missing-cache behaviors: SAM3 fetch returns an empty
result, while SAM3.1 fetch requires a cached frame. SAM3 merges refined masks into
an absent frame; the original SAM3.1 `_build_sam2_output` returns empty before
merging in this case, which is retained here and covered by tests. The eventual
point/mask frontend must establish the expected cache entries; this behavior is
not evidence of completed new-object prompt integration.

`propagate_video_refinements` executes every local session containing a requested
ID with memory encoding enabled, including its other session members internally.
Only requested IDs are returned to merge. It uses source video mask cleanup and
returns raw tracker scores plus low masks, so multi-GPU callers can exchange the
small tensors before video-resolution interpolation. This helper performs no
inter-GPU transport, detection or association, and is not an all-session
transaction. Native sessions/cores continue sharing weights without new copies.

The existing probe records its final displayed masks in this state. Its optional
`--partial-probe` regression selects the first existing ID after a 34-frame full
pass, routes a refine event and propagates backward from start 18 for three steps
(frames 17,16,15), merges into cached output, and fetches it again to check equality.
This fixed scenario is a test configuration, not a restriction of the library API.
It does not inject a new point/mask edit. Ordinary probe prompts remain supplied
at runtime; no detector query or object cap was introduced.

## Evidence

`video_interaction_parity.py` invokes actual original source methods for routing,
processing ranges, low-mask conversion, cache merge/filtering and postprocessing.
Each CPU/CUDA run passes 998 histories/ranges, rejects two source-invalid
cancel-after-fetch histories, and passes 5,106 exact cache/output/score checks.
Merge cases include empty/absent caches, 1/7/257 objects and three input precisions.
The source's malformed cancellation sequence raises IndexError; native code
raises a descriptive error instead. These focused tests use synthetic mask inputs.

C++ tests cover forced routes, refined-object recognition, cache mutation
isolation, forgetting IDs, reset, reverse-start exclusion and missing-cache fetch.
CTest passes 28 CUDA-enabled and 16 custom-CUDA-disabled checks. The latter still
uses this environment's GPU-capable LibTorch, not a standalone CPU distribution.

The extended `video_pipeline_parity.py --final-output --partial-output` reference
executes actual original neural forward tracking, replays original final output
scheduling, then invokes the original high-level interactive predictor's partial
reverse propagation. It compares the standalone C++ results without mocking neural
phases. Both probes execute with `PATH=/nonexistent`, retain all 200 queries and
four people, and reuse cached features at the corrected frame. Reports and full
private reference tensors distinguish this from the focused synthetic tests.

## Work still required

The complete owning predictor must integrate semantic text/geometric/visual
prompt replacement and reset, actual instance point/mask edits, singleton
extraction/repacking, detector-conditioning removal near edits, and metadata
updates for births/removals. Action dispatch/cache execution alone does not prove
these features complete. Also remaining: full C ABI, codecs, multi-GPU transport,
portable Windows/Linux SDK, CPU runtime stability, broader quality and performance.
No GitHub Actions or physical Windows/Turing validation was used. sm_75 cubins
are compile evidence only. Model-derived tensors remain in the private HF bucket.

Both SAM3 and SAM3.1 pass the extended real-neural comparison: all 34 raw and
final forward outputs remain exact, and all final IDs/scores/boxes/masks match on
partial reverse frames 17,16,15. Native cache fetch matches the just-merged output.
BF16, no TF32; SAM3.1 uses batch-one grounding/complex RoPE for neural matching
and default output batch 16. Reports are `video-interaction-{sam3,sam31}-source`
with `.json`, `.final.json` and `.partial.json` suffixes. Full tensors are private
under `video-interaction-integrated`; development binaries, headers and logs are
saved in `native-foundation/video-interaction-linux-cuda13`.
