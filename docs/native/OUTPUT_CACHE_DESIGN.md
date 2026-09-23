# Lossless displayed-mask storage

`VideoPredictor` can retain all displayed frame/object masks as one bit per pixel
in CPU memory or temporary disk archives. This removes those cached bool tensors
from the GPU without reducing the number of frames, objects or available edits.
The resident policy remains the default. It uses the same shared model store and
adds no model variants, weights, inference dependencies or runtime compilation.

Configure before use or after reset:

```cpp
predictor.set_output_cache(sam3::VideoOutputCacheStorage::PackedCPU);
// Alternatively, a caller-owned parent directory for temporary archives:
predictor.set_output_cache(sam3::VideoOutputCacheStorage::PackedDisk, directory);
auto stats = predictor.output_cache_stats();
auto frames = predictor.cached_frame_indices();
```

The additive C ABI functions are `sam3_predictor_set_output_cache` and
`sam3_predictor_output_cache_stats`. The storage constants are
`SAM3_OUTPUT_CACHE_RESIDENT`, `SAM3_OUTPUT_CACHE_PACKED_CPU` and
`SAM3_OUTPUT_CACHE_PACKED_DISK`; the last requires a UTF-8 directory path. Stats
are owning `sam3_result` int64 scalar fields named `storage`, `inspection_pinned`,
`frames`, `masks`, `resident_bytes`, `packed_bytes` and `disk_bytes`. Release them
with `sam3_result_release`. These calls use the existing exclusive owner guard,
including `SAM3_BUSY` during callbacks. No ABI-1 option structure or existing
public C++ class layout changes.

## Retention and failure behavior

Semantic reset clears cached data, temporary files and the inspection pin while
retaining the selected storage policy. Empty frames retain their index. Object
removal updates both resident and packed records. Reading restores the original
device and dense tensor strides; caller-owned returned/callback outputs remain
independent. No frame/object count limit is introduced.

The legacy C++ `interaction()` accessor exposes the entire resident cache by
reference. It materializes all packed frames and pins the cache resident until
reset so that later calls preserve that reference's usual semantics. Prefer
`cached_frame_indices`, `action_count` and stats for inspection. The C
`sam3_predictor_info` function uses these nonmaterializing accessors.

The standalone `VideoMaskCache` component accepts dense bool `[1,H,W]` masks.
It commits a replacement only after encoding and closing the archive succeeds.
The owner releases its latest resident copy only after storage succeeds; a
failed write leaves that copy resident and discards an obsolete packed version.
Resident frame replacement also commits only after every clone succeeds. Read,
CRC and truncation errors propagate instead of returning empty masks. These
guarantees concern cached masks; an entire neural edit/propagation call is not
transactional. A failed compaction can temporarily leave resident GPU storage.

Disk storage reuses the portable C++ `TensorArchive` implementation. It owns
unique temporary subdirectories under the caller's parent and cleans up its own
files on erase/reset/destruction. Do not move/delete that parent during use.
Archive metadata remains in process memory: these files are temporary caches,
not portable session checkpoints. Removed objects can retain unused bytes in a
frame's archive until replacement or cleanup; `disk_bytes` reports that payload.

`packed_bytes` counts active packed payload, including data held on disk.
`resident_bytes` counts logical resident tensor bytes. Neither describes process
RSS, filesystem allocation units, allocator reservation, model/neural history,
buffered raw outputs or tensors retained by the application. Compressed modes
do not make the whole predictor memory-bounded.

The attached `cppdocs/installing.md` documents LibTorch/CMake and Windows builds,
and `pytorch/2.14/cpp_extension.md` documents ATen/libtorch CUDA compilation.
This feature uses the existing ahead-of-time mask kernels, CPU ATen operations
and C++17 filesystem/streams. No Linux-only inference mechanism is added.
Windows/Turing physical execution remains user-owned; no Actions are used.

## Measurement and validation

The component benchmark holds 32 frames × 8 objects at 1920×1080 on Blackwell.
All three policies retain exactly the same masks/device/strides. The resident
CUDA allocator retains 536,918,016 bytes (about 512 MiB) above the live inputs;
packed CPU/disk modes retain zero CUDA bytes between operations. Logical mask
bytes are 530,841,600; packed payload is 66,355,200 bytes (63.28 MiB), one eighth
of the bool payload. These figures exclude inputs and the other state above.

One diagnostic process measured 6.64/55.88/46.20 ms to store all frames and
0.0026/0.380/2.91 ms per sampled read for resident/CPU/disk. The resident read
shares stored tensors, while compressed reads reconstruct independent tensors.
These are illustrative component costs, sensitive to allocator warmup, OS page
cache and storage; they are not model throughput comparisons. Compression and
transfer add work, which is why the policy is opt-in.

The benchmark is an installed-SDK consumer with CLI:

```text
sam3_video_mask_cache_benchmark DEVICE FRAMES OBJECTS HEIGHT WIDTH DIRECTORY REPORT.json
```

CPU/CUDA unit checks cover 257 objects, odd pixel counts, transposed dense masks,
independent reads, replacement rollback, shared-parent cleanup, CRC/truncation,
empty frames, deletion and move/reset. Full validation and artifact references
are recorded in `output-cache-validation.json`. Tests compare outputs to the
preceding native SDK, not a new claim of source-reference equivalence; the
earlier FP16 source residual remains documented in
`VIDEO_COLLECTIVE_REFERENCE.md`.

CPU22/CUDA40 CTests pass. Thirteen actual-weight C++ owner cases retain all 866
files across SAM3/SAM3.1, FP16/BF16 and resident/CPU/disk policies, including a
two-logical-rank parallel case. Six pure-C cases retain 1,074 files for both
video models and SAM3.1 image editing with CPU/disk policies and two logical
parallel CUDA ranks. They cover reset, point/mask edits, removal, cancellation,
callback errors, BUSY stats inspection and results retained past owner destruction.
Temporary archives are absent after the processes exit.

Complete SDK reconstruction verifies 157 CPU and 191 CUDA entries. Both recovered
cache tests pass, and the recovered pure-C SAM3.1 image lifecycle with disk cache
retains 153 files. It runs with `PATH=/nonexistent` and no `LD_LIBRARY_PATH`;
all 35 initialized workspace libraries come from the recovered SDK. Loader
tracing contains no Python/Triton runtime. Linux symbol comparison retains all
440 CPU / 447 CUDA previous public strong C/C++ symbols; 22 new symbols are added.

The private bucket `RamRom/sam3-turing-native-20260922` retains the SDK increment
at `output-cache-sdk-overlay/overlays.json` and validation evidence at
`native-foundation/output-cache-linux`. The increment reuses the preceding
rotary SDK's dependency and shared-weight layers. Output bytes that match prior
evidence are referenced by archive/member and SHA-256 instead of being copied
again.
