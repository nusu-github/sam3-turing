# CUDA library usage

Which CUDA-X/CCCL libraries the native kernels use, which substitutions were
tried and rejected, and the constraints behind those choices. Target when these
decisions were made (2026-09-24): Windows/MSVC 14.44, CUDA 13.0 with its bundled
CCCL 3.0.1, SM75, LibTorch 2.10 cu130, C++17.

Principle: adopt a library primitive when it removes scheduling or collective
bookkeeping without adding launches, intermediate tensors, dependencies or a
changed floating-point reduction order. Do not wrap a compact kernel in a larger
adapter only to use a library name. Online CCCL documentation describes newer
releases (for example `cuda/std/algorithm` needs CCCL 3.2 / CUDA 13.2); check the
installed headers before using an API.

## Current map

| Source | Operation | Library / decision |
|---|---|---|
| `src/ops_cuda.cu` | Mask packing | CUB `DeviceTransform` over counting iterators; bit order and tail padding preserved |
| `src/ops_cuda.cu` | Mask unpacking | `thrust::tabulate` with a device lambda |
| `src/ops_cuda.cu` | Greedy NMS | Hand-written; sequential suppression is not a scan/select |
| `src/components_cuda.cu` | Component size lookup | CUB `DeviceTransform`; 64-bit labels and background handling preserved |
| `src/components_cuda.cu` | Parent initialization, union-find, root counting | Original kernels (atomic union, canonical roots) |
| `src/vision_position_cuda.cu` | Per-axis position table | `thrust::tabulate` |
| `src/vision_position_cuda.cu` | Full position expansion | Original kernel (CUB version was slower, see below) |
| `src/vision_fusion_cuda.cu` | Welford / residual normalization | Original reduction order (a CUB rewrite would need new numerical validation) |
| `src/rotary_cuda.cu`, `src/rotary_pair_cuda.cu` | RoPE | Original fused pair writes and platform-specific FMA order |
| `src/roi_align_cuda.cu`, `src/edt_cuda.cu` | ROIAlign sampling, parabola-envelope EDT | Original kernels; no standard collective applies |
| `tools/approx_kernels.cu`, `tools/approx_boundary.cu` | INT8 row absmax | `cub::BlockReduce<float,256>` inside the existing fused kernels |
| `tools/int4_experiment.cu` | INT4 symmetric absmax (incl. MSE and H16/H64 rotation) | `cub::BlockReduce` with `BLOCK_REDUCE_WARP_REDUCTIONS`; affine min/max, MSE sums and Hadamard shuffles unchanged |
| `tools/int4_experiment.cu` | INT4 GEMM | External CUTLASS (optional `SAM3_EXPERIMENT_INT8_GEMM` build) |
| `tools/pixel_transform.cu` | FP16 NHWC → FP32 NCHW | Padded shared-memory transpose; an elementwise transform would lose coalescing |
| `tools/kitchen_attention.cu` | INT8 attention adapter | External ComfyKitchen kernels unchanged |
| INT8 GEMM | `at::_int_mm` | ATen (cuBLASLt underneath); custom CUTLASS tiles and cached cuBLASLt algorithms were rejected ([INT8_GEMM_SEARCH](../../experiments/results/native_rtx2060/INT8_GEMM_SEARCH.md)) |

Build notes:

- On Windows include CUB after the c10 CUDA headers. Including it before
  `CUDAGuard` exposed the Windows `small` macro to a LibTorch allocator parameter.
- `--extended-lambda` is set only for `src/ops_cuda.cu` and
  `src/vision_position_cuda.cu`, which contain Thrust device lambdas.
- Thrust calls use `thrust::cuda::par_nosync.on(current_stream)` with raw device
  pointers; tensors stay owned by ATen and are never captured in lambdas.
  `par_nosync` avoids unnecessary synchronization but does not make every
  algorithm asynchronous (host-returning `reduce`, selections and sorts allocate
  and synchronize), so those were not used for device-resident results.

## Adopted changes and measurements

All operator timings are CUDA Graph replays (30 batches × 20 replays after
warmup, allocation excluded), median of process medians in old/new/new/old
order. They are operator-level observations on the RTX 2060 Max-Q, not model
speedups. Every recorded operator output hash matched
`2cdf93d7663962e687777a78fc912a6bd2c6a9e5d34e22bc702714c52836fa63`; the 54
bit-exact position cases, nondefault streams, graph replay with changed inputs
and Compute Sanitizer memcheck passed.

### INT8 row reductions (CUB BlockReduce)

Replaced the custom reduction tree and warp aggregation with
`cub::BlockReduce` plus `cuda::maximum<>`; thread 0 publishes the scale through
shared memory and a barrier. GELU, Half rounding, clipping and launch sizes are
unchanged. 5184×4736: separate restore+quantize 1.524810 → 1.520390 ms, fused
0.701704 → 0.695944 ms (equivalent). The five full-model image cases of the INT8
research configuration stayed byte-identical. Evidence: `cub-*.json`.

### Mask pack/unpack, component sizes, position axis table

| Operator graph | Input | Before (ms) | Final (ms) |
|---|---|---:|---:|
| Pack | 1008×1008 | 0.015836 | 0.015382 |
| Unpack | 1008×1008 | 0.048899 | 0.032302 |
| Components | 256×256 random | 0.098698 | 0.074538 |
| Components | 256×256 zero | 0.020766 | 0.023121 |
| Components | 256×256 one | 0.391730 | 0.398242 |
| Position FP16 | 288×288 | 0.469702 | 0.468371 |
| Position FP32 | 288×288 | 0.487729 | 0.490452 |

Unpack and the position axis table were then moved from CUB `DeviceTransform`
to `thrust::tabulate` (unpack 0.032333 → 0.027035 ms; position within
−1.2…+1.6%), removing their functor types and explicit counting iterators.
Evidence: `cub-wide-*.json`, `cub-wide-final-summary.json`,
`thrust-*.json`, `thrust-combined-summary.json`.

### INT4 symmetric quantization

| Quantization, 5184×4736 | Before (ms) | Final (ms) |
|---|---:|---:|
| Symmetric | 0.369104 | 0.315024 |
| Symmetric + MSE | 1.826315 | 1.890220 |
| H16 | 0.499680 | 0.470024 |
| H64 | 0.702208 | 0.667768 |

Fingerprints of all packed outputs were unchanged. Evidence:
`cub-int4-final-*.json`, `cub-int4-final-summary.json`.

## Rejected substitutions

| Candidate | Result |
|---|---|
| CUB full position expansion | Bit-exact, but 288×288 FP16 0.631 → 0.901 ms and FP32 0.587 → 0.856 ms |
| CUB component parent initialization | Random 256×256 about 9% slower, repeated in reversed order |
| Thrust pack / conditional-transform size lookup | Pack slower with high variance; size lookup no shorter and not performance-equivalent |
| Affine INT4 `RowRange` collective | Outputs matched; affine+MSE stayed about 6% slower |
| NPP EDT (PBA) | 12 cases bit-equal and faster in a preliminary probe (1008×1008 random 0.813 vs 6.225 ms), but requires both sides 64–32767 and at least one site, only B=1 was tested, and it does not cover connected-component semantics |
| cuBLASLt GELU epilogue | Uses the tanh approximation; the runtime restores with erf and an explicit Half rounding point |
| cuBLASDx | Official requirements list Windows/MSVC as unsupported |
| cuDNN Frontend, cuTENSOR | No demonstrated benefit over existing ATen/vendor kernels or the small transpose kernel; not benchmarked |
| libcu++ `span`/`mdspan` | Usable in device code, but not yet applied: only worthwhile where it shortens shape/stride handling (ROIAlign, RoPE) |

A wider 2026-09-24 survey of the PyTorch C++, CUDA 13.0.3, cuBLAS, cuDNN,
cuTENSOR, cuSPARSE, cuSPARSELt, cuDSS and CCCL documentation is in
[CUDA_PYTORCH_LIBRARY_REVIEW_20260924_JA.md](CUDA_PYTORCH_LIBRARY_REVIEW_20260924_JA.md)
(Japanese). It found no library that replaces the large exact kernels, and lists
four candidates still to validate on the RTX 2060: dropping the position kernel,
a host NMS pass, one shared RoPE rounding helper and `cub::DeviceFor` launches.

The probes and parity scripts used here (`experiments/cccl_highlevel_probe.cu`,
`experiments/check_cub_ops.py`, `experiments/probe_npp_edt.py`) were removed
after the review and remain in commit `7d7fddb`. Raw results are in
[experiments/results/native_rtx2060](../../experiments/results/native_rtx2060/).
