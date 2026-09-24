# Displayed-mask cache storage

`VideoPredictor` keeps every displayed frame/object mask so that fetches,
partial propagation and edits can reuse them. By default these bool tensors stay
resident on the inference device. Two opt-in policies store them losslessly at
one bit per pixel in CPU memory or in temporary disk archives, removing them from
the GPU without reducing frames, objects or available edits. They use no extra
model weights or dependencies.

## Configuration

Configure before first use or after `reset()`; reset keeps the selected policy.

```cpp
predictor.set_output_cache(sam3::VideoOutputCacheStorage::PackedCPU);
// or temporary archives under a caller-owned parent directory:
predictor.set_output_cache(sam3::VideoOutputCacheStorage::PackedDisk, directory);
auto stats = predictor.output_cache_stats();
auto frames = predictor.cached_frame_indices();
```

C ABI: `sam3_predictor_set_output_cache` with `SAM3_OUTPUT_CACHE_RESIDENT`,
`SAM3_OUTPUT_CACHE_PACKED_CPU` or `SAM3_OUTPUT_CACHE_PACKED_DISK` (the last
needs a UTF-8 directory). `sam3_predictor_output_cache_stats` returns an owning
`sam3_result` with int64 fields `storage`, `inspection_pinned`, `frames`,
`masks`, `resident_bytes`, `packed_bytes` and `disk_bytes`. Both use the owner's
exclusive-call guard (`SAM3_BUSY` during callbacks). No ABI 1 structure changes.

## Semantics

- Semantic reset clears cached data, temporary files and the inspection pin.
  Empty frames keep their index. Object removal updates resident and packed
  records. Reads restore the original device and dense strides.
- A packed read decodes the masks and postprocesses them directly; the archive
  is not restored, repacked or rewritten, including for propagation routed
  through cached outputs. Each read returns independent tensors. C ABI calls and
  callbacks read under an inference guard; a C++ caller outside inference mode
  receives one extra clone so its tensor stays ordinarily writable.
- The legacy C++ `interaction()` accessor returns the whole resident cache by
  reference, so it materializes all packed frames and pins the cache resident
  until reset. Prefer `cached_frame_indices`, `action_count` and the stats; the
  C `sam3_predictor_info` uses only these non-materializing accessors.
- Replacement commits only after encoding and closing the archive succeed; a
  failed write leaves the latest copy resident. Read, CRC and truncation errors
  propagate instead of returning empty masks. A whole neural edit or propagation
  call is still not transactional.
- Disk storage uses the portable `TensorArchive` implementation, owns unique
  subdirectories under the caller's parent and deletes its files on erase,
  reset and destruction. Archive metadata lives in process memory: these are
  caches, not session checkpoints. Do not move or delete the parent during use.
- `packed_bytes` counts active packed payload (including data on disk),
  `resident_bytes` logical resident tensor bytes and `disk_bytes` file payload,
  which can include bytes of removed objects until replacement. None of them is
  process RSS; compressed modes do not bound the predictor's total memory
  (model, neural history and buffered outputs are separate).

## Measurements

Component benchmark (`sam3_video_mask_cache_benchmark DEVICE FRAMES OBJECTS
HEIGHT WIDTH DIRECTORY REPORT.json`), 32 frames × 8 objects at 1920×1080 on
Blackwell: the resident policy keeps 536,918,016 CUDA bytes; packed CPU/disk keep
none between operations. Packed payload is 66,355,200 bytes, one eighth of the
530,841,600 bool bytes. Storing took 6.64 / 55.88 / 46.20 ms and a sampled read
0.0026 / 0.380 / 2.91 ms (resident / CPU / disk, one process; resident reads
share tensors). Compression and transfer cost time, which is why the policy is
opt-in.

Owning-predictor fetch on Blackwell after the read-only change, FP16, three
alternating process pairs, 16 reads each:

| Model | Storage | Before | After |
|---|---|---:|---:|
| SAM3 | Packed CPU | 0.841 ms | 0.681 ms |
| SAM3 | Packed disk | 1.787 ms | 1.103 ms |
| SAM3.1 | Packed CPU | 0.845 ms | 0.679 ms |
| SAM3.1 | Packed disk | 1.754 ms | 1.096 ms |

Disk reads use the warm page cache. `sam3_video_cache_fetch_probe` checks
repeated reads, returned-mask isolation, cached propagation, empty frames,
corruption recovery and that archive files are not rewritten during fetch.

## Validation

CPU/CUDA unit tests cover 257 objects, odd pixel counts, transposed masks,
replacement rollback, shared-parent cleanup, CRC/truncation errors, empty frames,
deletion and move/reset. The actual-weight C++ owner cases (resident, CPU and
disk; SAM3/SAM3.1; FP16/BF16; including two logical ranks) and the pure-C
lifecycle cases reproduce the preceding native outputs exactly and leave no
temporary archives. Evidence:
[output-cache-validation.json](evidence/output-cache-validation.json),
[cache-fetch-validation.json](evidence/cache-fetch-validation.json).
