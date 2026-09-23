# Owning video predictor with tracking devices

The C++ owning `VideoPredictor` and C ABI now accept a list of tracking devices.
The constructor/context device remains the coordinator for vision, detection,
global association, output composition and displayed-frame caching. Tracking
sessions and memory live on their assigned rank's device. All existing semantic,
point/box/mask, forward/reverse, cached fetch, reset and cancellation routes
remain available. Execution is currently synchronous and serial across ranks;
parallel workers and a physical multi-GPU performance validation remain open.

## Configuration

Configure an unused predictor, or call `reset()` first:

```cpp
predictor.set_tracking_devices({at::Device("cuda:0"), at::Device("cuda:1")});
auto devices = predictor.tracking_devices();
```

The equivalent C ABI is additive; ABI 1 structures retain their layouts:

```c
const char *devices[] = {"cuda:0", "cuda:1"};
sam3_status status = sam3_predictor_set_tracking_devices(predictor, devices, 2);
```

The list must be nonempty and contain CPU/CUDA devices. Repeated entries create
logical ranks on the same device, useful for protocol tests; they share their
immutable tracker core and copied frame features. Unsupported/unavailable
configuration fails before replacing the active rank configuration. Device lists
are copied and need not outlive the call. The default is one tracking rank on
the coordinator. `reset()` and semantic-prompt replacement preserve the selected
devices; reset releases observations, copied features, caches and action state.
Reconfiguration after an active prompt is rejected until reset.

As with other owning calls, configuration is exclusive. Only `cancel()` may run
concurrently with an active call; it retains the existing frame/callback-boundary
cancellation contract. A failed neural update is not an all-session transaction.

## Module and state ownership

Only one vision encoder/detector and one text-encoding path are needed. The
coordinator computes complete frame features, including both SAM3.1 tracker
necks, once per cached frame. A distinct tracking device loads its tracker core
from the same shared weight store and receives the projected tracking features.
No image/video/model-specific checkpoint variants or extra disk weight files
are generated. Runtime replication of tracker parameters is per distinct device,
not per object, session or repeated device-list entry.

Cross-device copies use the prior host-staged transport, without NCCL or peer
access requirements. The attached `pytorch/2.14/distributed.md` explicitly
excludes NCCL on Windows; this implementation instead uses the existing portable
ATen/C++ SDK interfaces. Windows/Turing physical tests remain user-owned.
Physical CUDA-to-CUDA transport across different GPUs is not verified here.

Full propagation uses global collection/planning and per-rank execution from
[VIDEO_COLLECTIVE.md](VIDEO_COLLECTIVE.md). Interactive edits locate the owning
rank. New user objects choose the rank with the fewest objects, matching the
source user-edit workload argument (detector births retain bucket-aware SAM3.1
placement). A first stateless refinement removes its old state and selects the
least-loaded rank again. Structural point validation precedes that removal.

Selected edit masks return to the coordinator before merging with other displayed
masks. Partial propagation collects only requested objects from all ranks and
transfers their masks/scores before global overlap handling. Removal updates the
correct local collection, including SAM3.1 bucket counts. The existing single-rank
edit overloads keep local-device behavior; new overloads accept an explicit output
device for coordinated callers.

A mixed CUDA/CPU SAM3.1 probe exposed the same CPU autocast concatenation failure
previously fixed in SAM3: BF16 stored spatial memory can meet FP16 pointers. CPU
final temporal concatenation now uses ordinary dtype promotion with autocast
locally disabled. CUDA temporal assembly and subsequent neural operations are
unchanged. The original failure and passing rerun are retained privately.

## Validation

See [video-multidevice-validation.json](video-multidevice-validation.json) for
exact cases, hashes and scope. Tests run on the available RTX PRO 4500 Blackwell
with official standalone LibTorch 2.10.0 CPU/cu130, driver 580.159.04.

The new real-frame lifecycle probe places three manually prompted objects across
two ranks, checks annotation preservation, point editing, removal, forward and
reverse propagation, fetch, callback cancellation, reset and reconfiguration.
Two logical CUDA ranks compare ten outputs exactly against a single-rank owner
for each model and FP16/BF16 mode. The probe additionally initializes a semantic
prompt and checks least-loaded reassignment for stateless refinement. CUDA+CPU
cases test both models in FP16; their numerical outputs are not claimed equal
to CUDA-only execution. The three-frame fixture deliberately reuses one real
image to isolate state routing; it is not a motion/quality benchmark.

Separate existing owning-video probes run on four real sequence frames with
one and two logical ranks, exercising full detector/tracker propagation and
semantic/box/combined prompt replacement. Default single-rank output is compared
to the preceding 843709a library. Different rank partitioning can change numeric
execution and is not presumed equal to one rank or unmodified distributed Python.
Default output matches all 266 files exactly. Two-rank SAM3.1 output also matches
this fixture exactly; SAM3 changes masks in five frames per precision, 159 pixels
in total across both precisions, with minimum per-object IoU 0.99976113699.
These differences are retained in the report, not accepted under a hidden
comparison tolerance. A subsequent [source logical-rank replay](VIDEO_COLLECTIVE_REFERENCE.md)
reproduces the BF16 two-rank outputs exactly; FP16 and physical distributed-source
validation remain open.

Pure C probes pass for SAM3/SAM3.1 video and SAM3.1 image in FP16 with two logical
ranks. CPU/CUDA CTest suites pass 19/33 tests. C11 consumers compile against only
the installed SDK prefix; the CUDA C consumer passes the image lifecycle probe.
Installed/reconstructed C++ probes pass with Python absent from PATH. Loader
traces resolve all workspace runtime libraries inside the selected SDK. Complete
recovery checks 150 CPU / 183 CUDA entries, including every file hash and symlink.

No detection count, query count, prompt type or output resolution is reduced.
No Windows, Turing or two-physical-GPU execution is claimed. No Actions are used.
Parallel scheduling, broader quality and CPU long-run stability remain open;
the complete development goal is active.

## Reproduction and persistence

Build against the installed SDK using `native/eval` and its matching LibTorch
prefix, as described in [IMAGE_PRECISION_AUDIT.md](IMAGE_PRECISION_AUDIT.md#reproduction).
The added native probe requires no Python:

```bash
sam3_video_multidevice_probe STORE sam3 cuda fp16 FRAME.ppm BPE.gz cuda:0,cuda:0
sam3_video_multidevice_probe STORE sam3.1 cuda bf16_reference FRAME.ppm BPE.gz cuda:0,cuda:0
sam3_video_multidevice_probe STORE sam3.1 cuda fp16 FRAME.ppm BPE.gz cuda:0,cpu
```

A separate C11 consumer builds without a LibTorch development prefix:

```bash
cmake -S native/eval/c-predictor -B build/c-predictor -DCMAKE_PREFIX_PATH=/absolute/path/to/sdk
cmake --build build/c-predictor --config Release
```

The existing development probes accept optional comma-separated tracking devices:
`sam3_video_predictor_probe ... OUTPUT BOX_STEPS TRACKING_DEVICES` and
`sam3_predictor_c_probe ... OUTPUT video|image TRACKING_DEVICES`.

Private bucket `video-multidevice-sdk-overlay/overlays.json` pins the preceding
collective SDK recipe and incremental CPU/CUDA patches. Dependencies and weights
are reused. `native-foundation/video-multidevice-linux/` retains sources, fixture
references, baseline/current outputs, logs and SDK recovery evidence separately
from the deployable runtime.
