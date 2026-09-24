# Native weight store v1

Development exporter: `native/tools/export_weights.py`. Runtime reader:
`sam3::WeightStore` in `native/include/sam3/weights.h`. Python is used only to
read the source checkpoints and export; the C++ reader needs no Python. It uses
C++17 filesystem/streams, ATen and zlib, with no mmap or POSIX-only APIs.

The private `native-weights-v1/` directory holds both model versions as module
shards, one shared shard, an index and a JSON integrity/provenance manifest:
17 files, 6,900,865,193 bytes. Image and video, SAM3 and SAM3.1 all read the same
store; there are no per-use or per-precision copies. The original checkpoints
are a separate backup and are not needed at runtime.

## Layout and semantics

- `weights.s3i`: little-endian index. The reader loads only metadata at startup.
- `MODEL--MODULE.s3w`: raw contiguous tensor data, each tensor at a 64-byte
  aligned offset. No quantization or precision conversion.
- `shared.s3w`: tensors referenced from several modules or model versions,
  deduplicated by dtype + shape + SHA-256 of the logical bytes. The same raw
  bytes with a different dtype or shape are not interchangeable.
- `manifest.json`: SHA-256 and size of each data/index file, tensor identities
  and named model mappings. SHA-256 provides archive integrity; the per-tensor
  CRC32 detects payload corruption, not authenticity.

Index header: ASCII `SAM3WGT1` (8 bytes), record count (u64). Each record:

| Field | Encoding |
|---|---|
| Tensor name (`model/checkpoint.key`) | u32 UTF-8 byte count, bytes |
| Shard filename (flat, no directory separators) | u32 UTF-8 byte count, bytes |
| Offset, payload bytes | u64 each |
| CRC32 (IEEE/zlib) | u32 |
| Dtype code | u32 |
| Rank | u32 |
| Shape dimensions | rank × u64 |

Dtype codes 0–11: bool, uint8, int8, int16, int32, int64, float16, bfloat16,
float32, float64, complex64, complex128. Complex values keep interleaved
real/imaginary parts. Scalars have rank zero; an empty dimension has zero payload
bytes. Payloads are little endian and version 1 requires a little-endian host.

The reader checks magic/version, duplicate names, shape/size consistency,
integer overflow, alignment, shard path syntax, truncation and payload CRC32.
Strings are capped at 64 KiB and rank at 64; no pickled input or code is
executed. Verify `manifest.json` file hashes before deploying a copy.

`read(name, device)` allocates only that tensor, verifies it on the CPU and
copies it to the device. `read_prefix(prefix, device)` loads a module; pass a
trailing `.` for an exact module boundary. There is no process-global cache or
hidden full-model allocation: dropping the returned tensors releases the memory.
Repeated independent reads allocate independent tensors.

## Integrity checks

Every tensor's complete payload is checked with zlib's CRC32 before it is
transferred. zlib is called with at most 1 GiB per call, carrying the CRC state
across calls. `sam3_weights_large_test` writes 1 GiB of zeros plus 17 nonzero
bytes, checks the known checksum, then corrupts data beyond the chunk boundary
and requires rejection (it needs about 1 GiB of temporary disk and RAM).

Against the previous byte-at-a-time CRC loop, on a warm filesystem
(Blackwell/EPYC development host):

| Work | Before | After |
|---|---:|---:|
| Verify all 3,088 tensor references on CPU | 18.766 s | 2.618 s |
| Load SAM3 vision/text/grounding on CUDA | 9.497 s | 1.577 s |
| Load SAM3.1 vision/text/grounding on CUDA | 9.565 s | 1.591 s |

Loading still allocates the same CUDA memory. Evidence:
[weight-crc-validation.json](evidence/weight-crc-validation.json).

## Export and verification

Inventory the checkpoints first, then export:

```sh
python native/tools/inventory_weights.py --checkpoint sam3=/private/sam3.pt --checkpoint sam3.1=/private/sam3.1_multiplex.pt --output /private/inventory.json
python native/tools/export_weights.py --inventory /private/inventory.json --checkpoint sam3=/private/sam3.pt --checkpoint sam3.1=/private/sam3.1_multiplex.pt --output /private/native-weights-v1
```

The inventory identifies tensors by dtype, shape and SHA-256 of their logical
bytes. The exporter refuses an existing output directory, validates every
representative and alias against the inventory, writes into a temporary sibling
directory and renames it after success. Source checkpoints are not modified.

Verify without Python:

```sh
sam3_weights /private/native-weights-v1 verify
sam3_weights /private/native-weights-v1 verify sam3.1/ cpu
sam3_weights /private/native-weights-v1 list sam3/tracker.
```

All 3,088 tensor references (6,951,840,400 logical bytes; 6,898,771,600 unique)
pass this check. `native/tests/weight_parity.py` compares the Python exporter
with the C++ reader for all 12 dtypes, scalar/empty/noncontiguous tensors,
cross-model aliases and CPU/CUDA placement. Evidence:
[weight-store-validation.json](evidence/weight-store-validation.json),
[weight-sharing.json](evidence/weight-sharing.json),
[weight-loader-validation.json](evidence/weight-loader-validation.json).
