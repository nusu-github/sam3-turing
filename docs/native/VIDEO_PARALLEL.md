# Parallel tracking ranks

Tracking ranks can now dispatch on separate C++ worker threads in the owning
video predictor and lower-level collective API. Both complete propagation and
partial object refinement retain global collection/output handling. Vision,
detection, planning, frame-provider calls and application output callbacks stay
on the caller. Parallel execution is opt-in; existing serial behavior is the
default.

```cpp
predictor.set_tracking_devices({at::Device("cuda:0"), at::Device("cuda:1")});
predictor.set_parallel_tracking(true);
```

```c
const char *devices[] = {"cuda:0", "cuda:1"};
sam3_predictor_set_tracking_devices(predictor, devices, 2);
sam3_predictor_set_parallel_tracking(predictor, 1);
```

Configure before use or after `reset()`. Reset and semantic replacement retain
both settings. Active reconfiguration is rejected. The C setter accepts only
0/1 and adds a symbol without changing ABI 1 structures. Existing C++ overloads
retain their symbols and serial semantics; new lower-level overloads accept
`VideoRankExecution::Parallel` after the existing explicit options.

## Ownership and synchronization

Before dispatch, the owner prepares each distinct tracking device's frame
features on the caller. Workers only read that frame snapshot, shared module
parameters and their own rank's sessions. A worker cannot invoke the shared
encoder or the user's frame provider. Each execution rank receives its own plan
and transferred input payloads. Bucket counts and affected IDs merge into the
coordinator only after all execution workers succeed. All masks remain global
through visibility decisions before selecting rank-local memory rows.

Workers inherit ATen thread-local state and the caller's current stream on their
assigned device. Preparing inputs, worker submissions and subsequent caller
operations therefore retain the same stream ordering and allocator ownership.
Ranks on the same GPU share that stream; this feature does not introduce parallel
CUDA streams on one GPU. Distinct devices can execute independently. Transport
continues to use host staging, with no NCCL or peer-access requirement.

Calls remain synchronous at the host API boundary. Every launched worker is
joined before returning or throwing. Multiple worker exceptions report the
lowest rank's failure. Session updates are not transactional: an exception can
leave any launched rank partially updated. No rollback is promised. Cancellation
and application callbacks keep the existing frame boundaries; cancelling does
not interrupt a neural kernel already in flight.

The lower-level parallel overloads require caller-supplied factories and feature
providers to be safe for concurrent calls across ranks. The owning predictor
provides that coordination itself. One rank executes inline, avoiding thread
creation. Multiple ranks currently use scoped `std::async(std::launch::async)`
workers; persistent pools and overlapping transfers remain potential follow-up
optimizations. No new model, dependency or weight variants are created.

## Portability basis

The attached `cppdocs/installing.md` describes both Windows and Linux C++
distributions. The attached `pytorch/2.14/amp.md` explicitly identifies autocast
state as thread-local, motivating the captured ATen state. The implementation
uses standard C++17 threads and generic c10 stream guards available in the pinned
LibTorch headers. Attached `cppdocs/api/c10/streams.md` explicitly permits passing
a stream between threads and describes serialization of kernels queued on that
stream. `Threads::Threads` is private to the implementation. It adds no
Python interpreter, Triton, process group or Linux-only inference dependency.
Windows and Turing physical validation remains user-owned; actual execution on
multiple physical CUDA devices is unverified in this environment.

## Validation and recovery

Results and artifact references are recorded in
[video-parallel-validation.json](video-parallel-validation.json). Numerical comparisons use identical rank
partitioning: the prior source replay already established that changing the
number of ranks can change SAM3 masks. Existing FP16 source-versus-native
residual differences are tracked separately in
[VIDEO_COLLECTIVE_REFERENCE.md](VIDEO_COLLECTIVE_REFERENCE.md).


CPU20/CUDA35 CTests pass. Actual-weight lifecycle probes pass both models in
FP32, FP16 and BF16 with two logical GPU ranks (ten exact comparisons each
against a single-rank manual-object control), plus both models in mixed CUDA+CPU
FP16. Provider thread assertions verify caller ownership. The fixture reuses
one real image across three frames to isolate lifecycle behavior; it is not a
motion accuracy benchmark.

Four real sequence frames cover single-rank, two-rank serial and two-rank
parallel owning execution for both models in FP16/BF16: all 798 output files
match their corresponding pre-change outputs byte-for-byte. Seventeen-object
lower-level probes with actual weights and synthetic projected features compare
parallel births, reconditioning, global memory updates, forward/reverse tracking
and removal to independent serial execution for both models/precisions. All
three neural prediction comparisons per case match exactly.

Pure C probes pass SAM3/SAM3.1 video and SAM3.1 image with parallel ranks; all
537 result files also match the previous serial implementation. A C11 client
built against only the installed SDK passes after incremental reconstruction,
with all 153 image output files unchanged. Loader tracing resolves all 35
workspace runtime libraries inside the recovered SDK, with no Python/Triton
library; PATH excludes Python. Reconstructed C++ worker and real-weight probes
also pass. The complete recovery checks 151 CPU / 184 CUDA entries, including
file hashes and symlinks.

Functional-run durations are retained for diagnosis. Some runs overlapped, and
one physical GPU cannot demonstrate multi-GPU scaling; no measured speedup is
claimed. The option provides parallel host dispatch and device scheduling while
preserving the tested outputs. It remains opt-in.

Use the existing probes with their final optional argument set to `1`:

```bash
sam3_video_multidevice_probe STORE sam3 cuda fp16 FRAME.ppm BPE.gz cuda:0,cuda:0 1
sam3_video_collective_probe STORE sam3.1 cuda:0,cuda:0 bf16_reference 1
sam3_video_predictor_probe STORE sam3 cuda fp16 FRAMES.txt BPE.gz OUTPUT 1 cuda:0,cuda:0 1
```

The private `video-parallel-sdk-overlay/overlays.json` recipe pins the preceding
multi-device SDK layer and CPU/CUDA patches. Evidence lives under
`native-foundation/video-parallel-linux/`. Identical native outputs are referenced
by SHA-256 to the preceding multi-device evidence archives instead of uploaded
again. No new weights or dependency bundles are needed. No Actions were used.
