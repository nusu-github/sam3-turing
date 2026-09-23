# Faster weight integrity checks

`WeightStore::read` now uses the already-required zlib implementation of IEEE
CRC32 instead of its scalar byte-at-a-time loop. Every tensor still has its
complete payload checked before transfer to the requested device. The index,
shards, CRC values, tensor values and public ABI are unchanged. There are no new
dependencies or precision-specific weight packages.

The zlib count parameter is bounded: the loader passes at most 1 GiB per call
and carries the CRC state across calls. A portable CPU test writes 1 GiB of
zeros plus 17 nonzero tail bytes, verifies the independent known checksum, then
corrupts the tail beyond the chunk boundary and requires rejection. It also
checks valid and invalid empty-tensor checksums. This test uses about 1 GiB of
temporary disk/RAM; its file is removed on exit. Both SDKs install it as
`sam3_weights_large_test`, along with the existing `sam3_weights` verifier:

```sh
sam3_weights /path/to/native-weights-v1 verify
sam3_weights /path/to/native-weights-v1 verify sam3.1/ cpu
```

This change uses the existing cross-platform zlib dependency and C++ weight
loader. The standalone LibTorch platform basis and Windows build instructions
remain in [SDK.md](SDK.md). No Linux-only export or checksum API is introduced.
Windows/Turing execution remains user-owned; no GitHub Actions are used.

## Measurement scope

`weight-crc-validation.json` records three alternating baseline/current process
pairs after warmup, with no competing builds or GPU jobs. Whole-store verification
loads and CRC-checks all 3,088 tensor references (6,951,840,400 logical bytes),
releasing each tensor before the next. Module loading constructs vision, text
and grounding modules on CUDA and synchronizes. The latter excludes CUDA context
initialization, tokenization, inference and tracking-module construction.

These are warm host-filesystem measurements on the development EPYC/Blackwell
host, not cold storage, complete application startup or Turing performance.
Checksum-only microbenchmark gains must not be treated as inference gains.

| Measured work | Before (median) | After (median) | Reduction |
| --- | ---: | ---: | ---: |
| All weight references, CPU verification | 18.766 s | 2.618 s | 86.05% |
| SAM3 vision/text/grounding loading, CUDA | 9.497 s | 1.577 s | 83.39% |
| SAM3.1 vision/text/grounding loading, CUDA | 9.565 s | 1.591 s | 83.37% |

Module-loading current/peak CUDA allocation is unchanged: 3,407,298,048 bytes
for SAM3 and 3,428,063,744 bytes for SAM3.1. These figures exclude tracking.

Validation includes the full CPU/CUDA CTest suites, repeated weight edge tests,
six actual-weight C API image/video lifecycle cases, and recovery of both SDKs.
The recovered C lifecycle runs without Python on PATH, with loader tracing to
check that runtime libraries come from the recovered SDK. Existing original-
reference FP16 residuals in VIDEO_COLLECTIVE_REFERENCE.md are unaffected by this
native-to-native equality check.

CPU23/CUDA41 full CTests pass; the final empty-tensor assertions pass the targeted
CPU2/CUDA3 weight tests. The six C API cases retain all 1,074 output files exactly.
All 462 CPU / 469 CUDA preceding public strong symbols remain available.

The private bucket stores `weight-crc-sdk-overlay/overlays.json`, applied after
the cache-fetch overlay, and `native-foundation/weight-crc-linux` evidence.
Existing weights, runtime dependencies, fixtures and identical output payloads
are reused rather than redistributed in each optimization layer.
