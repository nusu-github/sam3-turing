# CUDA-X maintenance survey, 2026-09-24

Target: native Windows/MSVC, CUDA 13.0, SM75, LibTorch 2.10 cu130.
Decisions concern this repository and environment, not general library rankings.
Online documentation may describe a newer release than the installed toolkit;
implemented CUB and probed NPP interfaces were checked against installed headers.

The subsequent whole-repository CUB pass is documented in
[CUB wide review](CUB_WIDE_REVIEW.md), including DeviceTransform adoption and
rejected transformations.

## Library-to-code map

| Library | Repository candidate | Decision |
| --- | --- | --- |
| CUB / CCCL | INT8 and symmetric INT4 row extrema | Adopt block collectives inside existing fused kernels; no new launches or intermediate tensors. Affine INT4 trial rejected below. |
| CUB BlockLoad/BlockStore | INT4 packing, pixel transpose | Defer: INT4 uses paired/striped ownership; pixel conversion needs a two-dimensional coalesced transpose. A generic load/store substitution alone does not express the complete layout. |
| Thrust | Elementwise affine correction, filtering | Available, but current correction is one short kernel and filtering already uses ATen. No demonstrated maintenance saving here. Explicit stream selection would be essential. |
| CUTLASS | INT4/INT8 GEMM and epilogues | Already used. Keep the existing SM75 template GEMMs; newer API examples are not automatic drop-ins for the installed implementation. |
| cuBLASLt | GEMM + bias/GELU | Existing INT8 integration retained. Built-in GELU uses a tanh approximation; the current restore uses erf and an explicit Half rounding boundary. Replacing it would change numerical behavior. |
| cuBLASDx | Device-side GEMM in fused Attention | Defer on this host: official requirements list Windows/MSVC as unsupported. Moving compilation to NVRTC adds infrastructure rather than simplifying this build. |
| cuDNN Frontend | Convolution/normalization/Attention fusion | Candidate only after checking the exact operation graph, dtype and architecture support surface. Existing ATen convolution already delegates to vendor kernels; another graph wrapper needs a concrete benefit. |
| NPP | EDT, connected components, resize | EDT prototype completed below. Not a universal replacement; preserve existing public semantics. |
| cuTENSOR | Pixel layout/type conversion | Permutation/type conversion is relevant, but descriptor/plan management and a new library dependency have not been justified for the existing small transpose kernel. Not benchmarked. |

The useful reduction in maintenance comes from transferring scheduling and
collective synchronization to a library, not from compressing code onto fewer
lines. Do not replace a compact kernel with a larger adapter merely to use a
library name.

## INT4 CUB conversion

`native/tools/int4_experiment.cu` now expresses symmetric absmax with BlockReduce
using `BLOCK_REDUCE_WARP_REDUCTIONS`. CUB owns warp and block aggregation; thread
0 still publishes the scale through a barrier. This covers ordinary,
boundary-fused, MSE-assisted, and H16/H64 rotated symmetric variants. The custom
MSE sum tree and Hadamard shuffle arithmetic remain unchanged.

An affine min/max trial used a single custom `RowRange` collective. Outputs
matched, but its MSE variant remained about 6% slower after changing from the
default CUB algorithm to warp reductions (1.890 vs 2.002 ms in old/new/new/old
process order). Ordinary affine improved from the initial default-algorithm
regression with warp reductions, but the MSE regression remained. The affine
change was therefore reverted; no duplicate production implementation or runtime
switch was added. Trial code/logs are retained only under `.cache/cuda-library-pass2`.

`native/tools/int4_bench.cpp` adds quantization timings in this order:
symmetrical, symmetrical MSE, affine, affine MSE, H16, H64. It also records a
deterministic FNV-1a fingerprint of ordinary packed weights/activations and scales,
affine packed outputs/scales/offsets, and rotated outputs/scales. A matching hash
is a regression aid, not a mathematical proof or a substitute for CPU references.
Existing scalar quantization, integer GEMM, fused/separate, rotation and MSE-error
checks remain active on a nondefault stream.

Final old/new/new/old comparison, 5184x4736, median of two process medians
(each process: 10 warmups and 20 CUDA-event samples per variant):

| Quantization | Previous (ms) | Final (ms) |
| --- | ---: | ---: |
| Symmetric | 0.369104 | 0.315024 |
| Symmetric + MSE | 1.826315 | 1.890220 |
| Affine (unchanged) | 0.316608 | 0.317896 |
| Affine + MSE (unchanged) | 1.930680 | 1.897155 |
| H16 | 0.499680 | 0.470024 |
| H64 | 0.702208 | 0.667768 |

MSE symmetric was 3.5% slower in this series; the ordinary and rotated paths
were faster. These short process comparisons include allocation and exhibit
run-to-run variability (also visible in unchanged affine paths). No universal
speedup or end-to-end improvement is claimed. All four fingerprints were
`14503258391710240430`. See `experiments/results/native_rtx2060/cub-int4-final-*.json`
and `cub-int4-final-summary.json`. The old executable and its old native DLL were
kept in an isolated directory, so before/after runs use distinct implementations.
Final Compute Sanitizer memcheck reported zero errors; racecheck filtered to
`quant4` reported zero errors/warnings. Full CTest and model-level quality runs
were not repeated in this pass: production changes are confined to the opt-in
INT4 quantization collective and were checked with its operator harness.

## NPP EDT feasibility probe

Run from the repository root:

```powershell
.venv/Scripts/python.exe experiments/probe_npp_edt.py experiments/results/native_rtx2060/npp-edt-probe.json
```

The probe calls the installed CUDA 13 NPP C API using ctypes and application-managed
stream context. No model dispatch, build dependency or system configuration is
changed. It compares NPP with the current native CUDA EDT on a nondefault stream.

All 12 cases were bit-equal: 64x64, 65x97, 288x288 and 1008x1008, each with random
binary data, a single zero site, or all zero sites. Illustrative 1008x1008 timings:

| Input | NPP, preallocated (ms) | Native operator (ms) |
| --- | ---: | ---: |
| Random | 0.813 | 6.225 |
| Single site | 0.473 | 4.492 |
| All sites | 0.821 | 4.177 |

These are preliminary Python-driven CUDA-event observations. NPP buffers are
preallocated; the native call includes tensor allocation and conversions, and
host launch delays can affect event intervals. They are not controlled kernel-only
or model-level speedup claims. NPP scratch at 1008x1008 was 25,034,776 bytes.

Reasons not to promote it to the main implementation yet:

- NPP PBA requires both dimensions in 64..32767 and at least one site. The current
  native API supports smaller inputs and returns its finite-infinity result on
  all-foreground masks. Those cases require fallback or explicit adaptation.
- Only B=1 was tested. Batch handling, mixed empty/nonempty masks and CUDA graph
  behavior are unverified.
- Connected components additionally requires preserving equal-valued connectivity,
  background treatment, canonical minimum-index labels and per-pixel sizes. NPP
  labeling alone does not establish those repository-specific semantics.
- Resize uses ATen antialiased bilinear behavior and specific byte rounding;
  matching the interpolation name is not a parity test.

## Primary sources inspected

- [CUB BlockReduce](https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockReduce.html): block collective and custom reduction operators.
- [CCCL determinism](https://nvidia.github.io/cccl/unstable/cccl/determinism.html): floating-point reduction results can change with reduction structures and versions; avoid casually replacing MSE/Welford sums.
- [Thrust stream policies](https://github.com/NVIDIA/thrust/blob/main/examples/cuda/explicit_cuda_stream.cu): explicit CUDA stream execution.
- [CUTLASS overview](https://docs.nvidia.com/cutlass/latest/overview.html): architecture-specific mixed-precision building blocks.
- [cuBLAS epilogues](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html): GELU approximation definition.
- [cuBLASDx requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html): toolkit, host compiler and Windows limitations.
- [cuDNN graph support](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html): graph-specific support surfaces.
- [NPP EDT](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-distance-transform): PBA, ROI limits and required sites.
- [NPP label markers](https://docs.nvidia.com/cuda/archive/11.0/npp/group__image__filter__label__markers.html): connectivity and label APIs; implementation would need validation against installed headers.
- [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/index.html): tensor permutation and conversion capabilities.
