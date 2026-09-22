# Native runtime development log

## 2026-09-22 — mask/NMS foundation

User clarified that no Turing hardware will be available. Do not block on its
procurement. Continue independently and optimize after functional completion
until asked to stop. Hardware performance claims must remain Blackwell-only.

Implemented a shared C++/CUDA library, no Python link dependency:

- Mask packing/unpacking, independently padded rows, noncontiguous input support.
- Chunked bilinear resize and sigmoid with original dtype rounding.
- Generic IoU-matrix NMS, stable score ties, strict greater-than suppression,
  no detection count cap. Current CUDA baseline uses one cooperative block for
  greedy suppression and an ATen boolean matrix, leaving optimization scope.
- Development dispatcher registration and standalone C++ CPU/CUDA executables.
- Initially added Windows/Linux CPU CI; subsequently removed per user instruction.

Validation: 4 local CTest cases and 46 CPU + 46 CUDA parity scenarios passed.
Coverage includes empty input, tail bits, noncontiguous layouts, float16/32/64,
FP16 sigmoid threshold rounding, invalid shapes, tied scores, threshold equality,
8,203 detections, and a nondefault CUDA stream. See `ops-validation.json`.
`cuobjdump` found sm_75 and sm_120 cubins (Torch CMake also added intermediate
architectures). C++ CUDA test passed with Python removed from PATH. Shared
library is linked to the development installation's LibTorch, not yet packaged
for standalone relocation. Turing hardware execution has not been tested.

Next work: finish connected components / EDT, capture model reference fixtures,
measure shared checkpoint tensors, implement the common weight store and native
model components. No complete SAM3 native inference, C ABI, Windows CUDA proof,
or model-level numerical comparison exists yet.

## Validation ownership update

User requested no GitHub Actions for now and will handle Turing/Windows testing.
The native workflow run was canceled, the workflow disabled and removed. Linux
CPU CI had completed successfully before cancellation; Windows results are not
a gate or a claimed validation. Future checks run locally; no Actions polling
or new CI runs should be introduced. Provide user-facing reproduction commands
for Windows/Turing without blocking development on those platforms.

## 2026-09-22 — connected components, EDT and sharing inventory

Added CPU/CUDA 8-connected equal-value components with 64-bit labels/counts,
root-index labeling and zero background. CUDA uses atomic union-find and never
merges across image boundaries. Added separable Euclidean distance transform
with float32 output and double-precision envelope intersections; singleton
spatial dimensions and empty batches are supported. Upstream's finite infinity
for all-foreground input is retained.

Validation completed locally:

- 4 CTest CPU/CUDA cases, plus a separate build with custom CUDA kernels disabled
  (2 CTest cases), all passed.
- Independent reference suite: 67 CPU + 67 CUDA scenarios passed, including
  noncontiguous images, multivalued components, diagonal connectivity, large
  cross-block regions and brute-force distance comparisons.
- Direct original-Triton comparison: 6 mask, 4 NMS, 4 component and 2 EDT cases
  passed. `upstream-ops-validation.json` records the environment.
- Compute Sanitizer memcheck and synccheck: zero errors on the standalone CUDA
  primitive test. These checks exercise the probe, not a full model workload.

Exact checkpoint inventory: 3,088 tensor references, 2,998 unique typed/shaped
contents, 6,951,840,400 logical bytes and 6,898,771,600 unique bytes. Exact
sharing saves 53,068,800 bytes; only 33,733,632 unique bytes occur in both model
versions. Thus SAM3 and SAM3.1 encoders must not be assumed to share parameters.
The larger distribution saving will come from reusing each version's modules
across image/video use, not duplicating full checkpoints per execution path.
`native/tools/inventory_weights.py` reproduces the analysis; the detailed
inventory is private and the size summary is `weight-sharing.json`.

All of these primitives are available to C++ callers and development tests;
the original SAM3 Python inference paths have not yet been rewired. The native
library is still not a full model runtime. Next: model reference fixtures,
on-demand shared weight storage, C++ model components and host session control.

## 2026-09-22 — lossless module store and native language encoder

Implemented the v1 weight store, C++ reader and lossless exporter. Metadata is
opened without allocating the full model; individual tensors or module prefixes
can be loaded to CPU/CUDA and released by the caller. Identical typed/shaped
contents are stored once, including cross-model/module references. The complete
private store is 6,900,865,193 bytes across 17 files, including metadata/alignment.
Original source checkpoints are backups outside the deployable store.

All 3,088 named tensor references were read with payload checksum validation by
a standalone C++ executable, with Python removed from PATH. Synthetic exporter
and C++ reader comparisons passed exact bytes for 12 dtypes (including BF16 and
complex), scalar/empty/noncontiguous tensors, sharing and CPU/CUDA placement.
Truncation, inconsistent shape, checksum corruption, invalid shard paths and
stale alias inputs are rejected. Format and commands: `WEIGHTS.md`.

Implemented the complete 24-layer SAM3/SAM3.1 VE encoder in C++/ATen, reading
only the language module from the shared store. The API accepts arbitrary token
batches/lengths within the model's existing 32-token context; it returns all
three outputs used by upstream VETextEncoder. The unused pooled projection is
not executed because upstream VE discards it. Tokenizer is not yet ported.
GPU FP32/TF32-off comparisons against both checkpoint versions passed on
multi-prompt, variable-length and single-token cases; maximum absolute error
was zero in all six cases. See `text-cuda-validation.json`.

Captured unmodified SAM3 image eager FP32 reference tensors using the truck
asset: three independent text prompts, a positive box, positive+negative boxes,
and a changed confidence threshold. All 200 architectural queries and the
original 1008 input resolution were retained. The private reference directory
contains backbone/text features, raw detector outputs and final results; summary
is `image-reference-summary.json`. These are reference fixtures, not evidence
of a native image implementation. Interactive point/mask, batch and video
fixtures remain outstanding.

Next: native tokenizer, visual backbone/neck and geometry/detector components,
then interactive image and video/session control. Keep user-owned Turing/Windows
validation and no-Actions instruction in force.

CPU text comparisons also passed with zero maximum absolute error for the same
six cases (`text-cpu-validation.json`). The standalone C++ text CLI loaded the
private module store and executed on CUDA with Python removed from PATH. The
custom-CUDA-disabled build passed its 3 CTest cases; CUDA-enabled build passed
6. This proves the module/loader path for those inputs, not tokenizer or whole
model parity. Model modules and session lifetime control remain incomplete.
