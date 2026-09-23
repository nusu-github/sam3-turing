# Read-only fetch from packed display caches

The owning predictor now postprocesses a packed frame directly from its decoded
masks. It retains the packed source instead of restoring a resident cache entry,
cloning it, deleting the source, and then repacking/transferring/writing the same
masks again. This also applies to propagation routed through cached outputs.
No masks, objects, frames, prompts or edit routes are removed.

The output postprocessor and current metadata/suppression rules are unchanged.
Each packed read owns independent tensors, so changing a returned mask cannot
change a later fetch. Empty frames retain their indices and CPU empty-output
behavior. Corrupt archive reads still fail without deleting the source record.

`VideoMaskCache::read` creates inference tensors. Previously, the intermediate
resident clone created normal tensors when the C++ caller was outside an
inference guard. That caller still receives a normal, writable cached tensor:
one compatibility clone is retained in this case. Guarded callers, including
all C ABI calls and propagation callbacks, avoid the clone. Tensor values,
strides and edit/fetch functionality do not depend on this choice.

Resident storage and legacy bulk inspection retain their existing behavior.
The owner also retains recovery compaction for resident frames left by an earlier
failed write; such recovery can still perform I/O. Ordinary reads from a fully
packed cache do not rewrite archives. No public ABI, dependency, model module or
weight file changes. The implementation uses the same portable C++/ATen path as
[the packed-cache feature](OUTPUT_CACHE_DESIGN.md).

The standalone `sam3_video_cache_fetch_probe` tests repeated reads, returned-mask
isolation, cached propagation without encoding, empty frames, corruption/recovery,
ordinary C++ tensor mutability, and legacy inspection/reset. It snapshots archive
paths, lengths and modification times and compares all saved output tensor bytes
and layouts against the preceding SDK. Its fixed four-frame/person input is a
measurement fixture only; the library keeps dynamic prompts and uncapped counts.

Detailed final validation, measurements and private recovery artifacts are
recorded in `cache-fetch-validation.json`. Source-reference FP16 residuals remain
separate from native-to-native exactness; see VIDEO_COLLECTIVE_REFERENCE.md.
Windows/Turing execution remains user-owned. No GitHub Actions are used.

## Measured scope

Three independent FP16 baseline/current process pairs per model/storage policy,
with alternating order and 16 synchronized reads after warmup, measured these
median times on Blackwell. No competing GPU work ran during measurement.

| Model | Storage | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| SAM3 | Packed CPU | 0.841 ms | 0.681 ms | 19.09% |
| SAM3 | Packed disk | 1.787 ms | 1.103 ms | 38.27% |
| SAM3.1 | Packed CPU | 0.845 ms | 0.679 ms | 19.65% |
| SAM3.1 | Packed disk | 1.754 ms | 1.096 ms | 37.52% |

These are guarded owning-predictor fetch times, excluding model loading,
encoding and C result packing. Disk reads use the host filesystem/page cache;
they do not measure cold storage. Ordinary C++ calls outside inference mode are
tested for writable tensor semantics but are not separately timed. Peak
additional CUDA allocation remains unchanged: 45,156,352 bytes for SAM3 and
46,680,064 bytes for SAM3.1 above live model/cache/expected-output tensors.

CPU22/CUDA40 CTests pass. The final library retains all 1,940 files across the
13 owner and six C API regression cases. Sixteen fetch-probe process pairs
(32 processes) cover both models, FP16/BF16 and CPU/disk policies; every pair
retains 40 tensor/metadata files. Archive identities remain unchanged only in
the new disk fetch path, confirming that writes were removed. Both versions
also pass ordinary C++ mutation isolation and the other lifecycle assertions.
All 462 CPU / 469 CUDA preceding public strong symbols remain available.

The complete recovered SDKs verify 158 CPU / 192 CUDA entries. Recovered cache
tests pass; the pure-C image/edit lifecycle retains 153 files and the installed
fetch probe retains 40. Native execution uses `PATH=/nonexistent`; all 35
initialized workspace libraries come from the recovered SDK, with no Python or
Triton runtime. The private bucket retains `cache-fetch-sdk-overlay/overlays.json`
(based on the output-cache SDK) and evidence at
`native-foundation/cache-fetch-linux`. Weights and dependency layers are reused.
