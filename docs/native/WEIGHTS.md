# Native weight store v1

Development exporter: `native/tools/export_weights.py`. Runtime reader:
`sam3::WeightStore` in `native/include/sam3/weights.h`. Python is used only to
read source checkpoints and export artifacts; C++ reads the exported store
without Python. The implementation uses C++17 filesystem/streams and ATen,
with no mmap or POSIX-only APIs. User owns Windows/Turing execution validation.

The private `native-weights-v1/` directory contains the two model versions as
module shards, one shared shard, an index and a JSON integrity/provenance
manifest. Original checkpoints remain in the separate `weights/` backup tree;
they are not dependencies of the native store and must not be bundled with it.
No image/video-specific copies of either full model are generated.

## Layout and semantics

- `weights.s3i`: little-endian index. The reader loads only metadata at startup.
- `MODEL--MODULE.s3w`: raw contiguous tensor data, each tensor starts at a
  64-byte-aligned offset. There is no quantization or precision conversion.
- `shared.s3w`: tensors referenced from multiple modules/model versions, deduped
  by dtype + shape + SHA256 of logical bytes. Same raw bytes with different
  dtype or shape are not interchangeable.
- `manifest.json`: SHA256 and size for each data/index file, tensor identities
  and named model mappings. SHA256 provides archive integrity information;
  runtime per-tensor CRC32 detects payload corruption, not authenticity.

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
float32, float64, complex64, complex128. Complex values retain interleaved
real/imaginary representation. Scalar rank is zero; an empty dimension has
zero payload bytes. All payloads are little endian. Version 1 requires a little
endian host (the intended x86 Windows/Linux hosts qualify).

Reader checks include magic/version, duplicate names, shape/size consistency,
integer overflow, alignment, shard path syntax, truncation and payload CRC32.
The metadata parser caps individual strings at 64 KiB and rank at 64. It does
not execute pickled input or arbitrary code. Release integrity verification
should additionally verify `manifest.json` file hashes before deployment.

`read(name, device)` allocates only that tensor, verifies it on CPU, and copies
it to the requested device. `read_prefix(prefix, device)` loads a module's
matching tensors; pass a trailing `.` for an exact module boundary. Drop the
returned tensors/module to release memory. There is no process-global cache or
hidden permanent full-model allocation. Physical sharing is already enforced
on disk; repeated independent reads currently allocate independent tensors.

## Reproduce

Run the inventory first (see `native/README.md`), then:

```sh
python native/tools/export_weights.py --inventory /private/inventory.json --checkpoint sam3=/private/sam3.pt --checkpoint sam3.1=/private/sam3.1_multiplex.pt --output /private/native-weights-v1
```

Export refuses an existing output directory, validates every representative
and alias against the inventory, writes into a temporary sibling directory,
and renames it after success. It does not modify source checkpoints.

After the native build, verify all weights without Python:

```sh
build/native/sam3_weights /private/native-weights-v1 verify
build/native/sam3_weights /private/native-weights-v1 list sam3/tracker.
```

On Windows use the corresponding `build/native/Release/sam3_weights.exe`.

## Current evidence and limits

All 3,088 tensor references (6,951,840,400 logical bytes) were read and checksum
verified by the C++ CLI with Python absent from PATH. The unique data total is
6,898,771,600 bytes; index, alignment and JSON metadata add small overhead.
Synthetic exporter-to-C++ tests compare exact logical bytes for 12 dtypes,
scalar/empty/noncontiguous source tensors, cross-model aliases, CPU/CUDA
placement and stale-inventory rejection. This weight-reader check does not establish model-level
inference parity. Tokenizer, image/geometry, tracker/multiplex and owning session
implementations were added subsequently; see [QUICKSTART_JA.md](QUICKSTART_JA.md)
for the current runtime and remaining source-reference differences.
