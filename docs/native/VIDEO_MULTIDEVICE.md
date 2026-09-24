# Tracking ranks on multiple devices

The owning `VideoPredictor` and the C ABI can place tracking sessions on a list of
devices ("ranks"). The constructor/context device stays the coordinator for
vision, detection, global association, output composition and the displayed-mask
cache; tracking sessions and memory live on their rank's device. Ranks run
serially by default, or on worker threads when parallel tracking is enabled.
Every semantic, point/box/mask, forward/reverse, fetch, reset and cancellation
route remains available.

Only single-GPU hardware was available: two ranks on one GPU and CUDA+CPU ranks
were tested. Physical CUDA-to-CUDA transport and multi-GPU speedups are
unverified.

## Configuration

Configure an unused predictor or call `reset()` first:

```cpp
predictor.set_tracking_devices({at::Device("cuda:0"), at::Device("cuda:1")});
predictor.set_parallel_tracking(true); // optional; serial by default
```

```c
const char *devices[] = {"cuda:0", "cuda:1"};
sam3_predictor_set_tracking_devices(predictor, devices, 2);
sam3_predictor_set_parallel_tracking(predictor, 1);
```

The list must be nonempty and contain CPU/CUDA devices; it is copied. Repeated
entries create logical ranks on one device that share its tracker core and
copied frame features (useful for protocol tests). Invalid configuration fails
before replacing the active one. Reset and semantic-prompt replacement keep the
settings; reconfiguring after an active prompt is rejected until reset. The C
setters are additive and do not change ABI 1 structures.

## Ownership and data flow

- One vision encoder/detector and one text path run on the coordinator, which
  computes complete frame features (including both SAM3.1 tracker necks) once per
  cached frame. Each distinct tracking device loads its tracker core from the
  same weight store and receives projected tracking features. Tracker parameters
  are replicated per distinct device, never per object, session or repeated entry.
- Cross-device copies stage through host memory: no NCCL or CUDA peer access is
  required. (PyTorch's documentation excludes NCCL on Windows, so the source
  predictor's NCCL/process-queue orchestration is not a portable deployment path.)
- Collection follows the source: local tracker masks are cleaned, then for
  multiple ranks masks and logits become contiguous FP32 and are concatenated in
  rank/metadata order (a single rank keeps its dtype). Collection restores
  metadata order and rejects missing, duplicate or cross-rank IDs; empty ranks
  stay explicit. Masks remain global through visibility and suppression before
  memory updates select each rank's rows.
- New user objects go to the rank with the fewest objects (the source's
  user-edit workload rule); detector births keep bucket-aware SAM3.1 placement. A
  first stateless refinement re-selects the least-loaded rank. Edited masks
  return to the coordinator before merging; partial propagation collects only the
  requested objects from all ranks.

### Parallel workers

The owner prepares each distinct device's frame features on the caller before
dispatch. Workers only read that snapshot, shared module parameters and their own
rank's sessions; they never call the shared encoder or the application's frame
provider. Bucket counts and affected IDs merge into the coordinator after all
workers succeed. Workers inherit ATen thread-local state and the caller's current
stream on their device, so ranks on one GPU share a stream (no extra CUDA streams
per GPU). Calls stay synchronous at the API boundary: every worker is joined
before returning, and the lowest rank's exception is reported. Session updates
are not transactional. One rank runs inline; several use scoped
`std::async(std::launch::async)` workers.

## Lower-level API

`sam3/video_collective.h` provides the same boundary for callers that assemble
the update loop themselves. A rank descriptor borrows a device, a session
collection and a factory; collections must be distinct and exclusively accessed.

```cpp
auto metadata = sam3::initialize_video_metadata(ranks.size(), coordinator);
auto tracking = sam3::propagate_video_tracking_ranks(
    frame, reverse, ranks, metadata, coordinator, options.cleanup_area);
auto plan = sam3::plan_video_update(frame, reverse, detections,
    tracking.masks, tracking.logits, metadata, options);
sam3::execute_video_update_ranks(frame, plan, detections, ranks, options);
auto output = sam3::build_video_outputs(plan, detections, height, width, options);
sam3::finalize_video_scores(plan.metadata, frame, plan.previous_ids,
                           tracking.logits);
metadata = std::move(plan.metadata);
```

Initialize metadata once per session. `Sam3VideoRank` and `Sam31VideoRank`
overloads share this contract; `VideoRankExecution::Parallel` selects worker
threads and then requires factories and feature providers that are safe for
concurrent calls.

The mixed-device tests exposed a CPU autocast failure: BF16 stored spatial memory
met FP16 pointers in CPU `cat`. CPU temporal concatenation now disables autocast
locally and uses ordinary dtype promotion; CUDA assembly is unchanged.

## Validation

- Collection tests: FP16/BF16/FP32/FP64, noncontiguous masks, reordering, large
  signed IDs, empty ranks and mixed CPU/CUDA inputs and destinations.
- With actual tracker weights and synthetic projected features, two logical
  ranks reproduce independent serial per-rank execution exactly (SAM3/SAM3.1,
  FP16/BF16, 17 objects across the 16-slot bucket boundary; births, visibility,
  cross-rank reconditioning, memory updates, forward/reverse, removal). Parallel
  workers reproduce serial execution exactly as well.
- The real-frame lifecycle probe spreads three prompted objects over two ranks;
  ten outputs per model and precision equal the single-rank owner.
- On the four-frame semantic fixture, single-rank and two-rank-parallel outputs
  equal the previous serial implementation byte-for-byte. Changing the number of
  ranks changes SAM3 masks slightly (159 pixels over both precisions, minimum IoU
  0.99976); a source replay with the same logical ranks reproduces the BF16
  result exactly ([VALIDATION.md](VALIDATION.md#logical-ranks)).

Probes (the final optional argument `1` enables parallel workers):

```sh
sam3_video_collective_test cpu|cuda
sam3_video_collective_probe STORE sam3 cuda:0,cuda:0 fp16 [1]
sam3_video_multidevice_probe STORE sam3.1 cuda fp16 FRAME.ppm BPE.gz cuda:0,cpu [1]
sam3_video_predictor_probe STORE sam3 cuda fp16 FRAMES.txt BPE.gz OUTPUT 1 cuda:0,cuda:0 [1]
sam3_predictor_c_probe ... OUTPUT video|image TRACKING_DEVICES
```

Evidence: [video-collective-validation.json](evidence/video-collective-validation.json),
[video-multidevice-validation.json](evidence/video-multidevice-validation.json),
[video-parallel-validation.json](evidence/video-parallel-validation.json).
