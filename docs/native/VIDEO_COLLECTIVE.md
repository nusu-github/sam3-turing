# Native tracking across rank collections

`sam3/video_collective.h` adds a synchronous C++ boundary for collecting tracker
predictions and executing a global video update on multiple rank collections.
The owning `VideoPredictor` uses this boundary with its existing single rank.
An owning multi-device predictor and C device-list API are **not implemented**
yet. Rank execution is currently serial, so this is not a multi-GPU speedup claim.

## Source contract and platform choice

The attached `pytorch/2.14/distributed.md`, “Backends that come with PyTorch”,
explicitly excludes NCCL on Windows. The source predictor's NCCL/process-queue
orchestration therefore cannot be the portable deployment path. This boundary
uses ordinary ATen C++ CPU/CUDA tensor copies in one process. Copies between
CUDA devices explicitly stage through CPU memory, requiring neither NCCL nor
CUDA peer access. The existing standalone LibTorch Windows/Linux build path
remains applicable; no new third-party runtime or model weights are added.
This is a design expectation from the attached documentation, not a Windows or
Turing execution result. Both physical checks remain user-owned.

`sam3/model/sam3_video_base.py` and `sam3/model/sam3_multiplex_base.py` clean
local tracker masks before collection. For multiple ranks, masks and object
logits become contiguous float32, then concatenate in rank/metadata order.
A single rank retains its local dtype. The native implementation preserves
that order, including cleanup before conversion. Native session grouping may
produce a different within-rank order; collection restores metadata order and
rejects missing, duplicate or cross-rank IDs. Signed 64-bit user IDs are retained.
Empty ranks remain explicit, including when every rank is empty.

Global masks must remain global through visibility/suppression decisions. Only
then may memory updates select rows belonging to each local session. The new
executor transfers those global masks, correction masks and detection tensors
to each rank's device, invokes the existing per-rank update, and merges bucket
counts and reconditioned IDs into the coordinator's plan. It does not trim the
object set, cap detections or specialize prompts.

## API ownership and usage

A rank descriptor borrows a device, session collection and session factory.
Collections/sessions must be distinct and exclusively accessed during the call.
Factories and feature providers must construct sessions on the declared device.
The API does not create full-predictor replicas or copy weights on disk.
Tensor transfer returns independent storage; gathering also owns its output.

A coordinator can call these steps around the existing detector:

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

Initialize metadata once per session, before the frame loop. Prediction IDs are
in the previous metadata order. `Sam3VideoRank` and `Sam31VideoRank` overloads
share this contract. As with the existing single-rank executor, a failure stops
later updates but does not roll back earlier session mutations. This boundary
does not implement the owning predictor's output buffering or user-edit routing.

A CPU FP16 failure found by the mixed-device probe was fixed in temporal-memory
assembly: stored spatial memory remains BF16, and CPU autocast rejects mixed
lower-precision inputs to `cat`. The CPU concatenation now temporarily disables
autocast and uses ordinary dtype promotion. Stored memory, subsequent neural
arithmetic and the CUDA assembly path remain unchanged. The original failure
log is retained alongside the passing rerun.

## Validation and its limits

[video-collective-validation.json](video-collective-validation.json) records
library/source hashes and evidence:

- Full local CTest suites: 19 CPU-build tests and 33 CUDA-build tests pass.
- Collection tests exercise FP16/BF16/FP32/FP64, noncontiguous masks, reordering,
  signed/large IDs, empty ranks, independent storage, invalid placement/shapes,
  and mixed CPU/CUDA inputs with CPU or CUDA destinations.
- Actual tracker weights with controlled synthetic projected features compare
  the new helper with independent serial calls to the existing rank executor.
  Both SAM3/SAM3.1 and FP16/BF16 pass with two logical ranks on CUDA device 0.
  Seventeen objects exercise SAM3.1's 16-slot bucket boundary. Births, global
  visibility, cross-rank reconditioning, memory updates, forward/reverse
  propagation and removal are checked; three neural predictions per case are
  exactly equal. Synthetic features make this a protocol test, not a visual
  accuracy evaluation.
- SAM3 FP16 additionally passes with one CUDA rank and one CPU rank. This uses
  the CPU backend of the CUDA LibTorch distribution and verifies actual
  heterogeneous neural execution and transport. It is not two physical GPUs.
- Existing owning-video probes compare the prior native library (592def8) with
  the new library on four real frames: semantic, box, combined prompts,
  point/mask initialization, cached fetch, reset, removal and cancellation.
  Both models and both precisions retain identical emitted bytes: 65 files per
  SAM3 case and 68 per SAM3.1 case, 266 files total. This is a regression against
  the prior native implementation, not unmodified distributed Python parity.
- Installed and reconstructed standalone SDK consumers run without Python on
  PATH. Recovery verifies every staged SDK file hash and symlink target.

The single available GPU is RTX PRO 4500 Blackwell, driver 580.159.04, official
LibTorch 2.10.0 CPU/cu130. Physical CUDA-to-CUDA transport across different GPUs,
parallel execution, Windows and Turing have not been tested. No Actions are used.

## Reproduction and persistence

The development build includes `sam3_video_collective_test` and
`sam3_video_collective_probe`. Both can also be built against the installed SDK
using `native/eval` and the matching LibTorch prefix, as described in
[IMAGE_PRECISION_AUDIT.md](IMAGE_PRECISION_AUDIT.md#reproduction).

```bash
sam3_video_collective_test cpu
sam3_video_collective_test cuda
sam3_video_collective_probe STORE sam3 cuda:0,cuda:0 fp16
sam3_video_collective_probe STORE sam3.1 cuda:0,cuda:0 bf16_reference
sam3_video_collective_probe STORE sam3 cuda:0,cpu fp16
```

`STORE` is the existing shared native weight directory. A list such as
`cuda:0,cuda:1` selects distinct devices when available. Repeating a device is a
logical-rank test; the probe shares the tracker core on that device. The fixed
17-object probe fixture is not a production object limit.

Private bucket `video-collective-sdk-overlay/overlays.json` pins the preceding
image-postprocessing SDK recipe and the small CPU/CUDA patches. These contain
the new header, native library and changed/new audit executables only; LibTorch,
media libraries and weights are reused. `native-foundation/video-collective-linux/`
retains source snapshots, logs, the original CPU failure, owner regression
outputs and comparison reports. These evidence fixtures are separate from the
deployable SDK.

Remaining work includes an owning device-list API, device-local feature/core
ownership, parallel rank workers and edit/reset/cancel coordination. Broader
quality, CPU long-run stability and prior open items also remain. The overall
development goal is active.
