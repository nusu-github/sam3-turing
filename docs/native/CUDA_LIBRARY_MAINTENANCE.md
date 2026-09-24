# CUDA library refactor: INT8 row reductions

2026-09-24, Windows / RTX 2060 Max-Q (SM75), CUDA 13.0, bundled CCCL 3.0.1,
LibTorch 2.10 cu130. This pass prioritizes maintenance, not new approximations.

## Changes

- `native/tools/approx_kernels.cu`: use `cub::BlockReduce<float,256>` for row absmax.
- `native/tools/approx_boundary.cu`: use `cub::BlockReduce<float,Threads>` for
  fused restore/GELU/quantization absmax, including 128/256/512-thread variants
  (only the adopted 256-thread kernel remains after the repository cleanup).
- Use `cuda::maximum<>`, the installed CCCL API, and explicitly broadcast the
  thread-0 scale with shared memory and a barrier. Per-thread absmax starts at zero
  and uses `fmaxf`, so reduction inputs are nonnegative and NaN-free.
- Preserve GELU, Half rounding, division, clipping, round-to-nearest conversion,
  layouts, launch sizes, and current-stream behavior.

CUB is header-only and comes with the installed toolkit. No download, new link
library, CMake include workaround, or new runtime switch was required. Total
source reduction is only three lines including the new includes/comments; the
useful simplification is removing the custom reduction tree and warp aggregation.

## Validation and timings

`sam3_boundary_bench` now independently checks scales and INT8 values against a
CPU scalar implementation, on a nondefault CUDA stream. Widths 1, 31, 32, 257,
1024, 4736, 8192 cover tails and production dimensions. Random, zero, and constant
inputs pass, as do `threads128`, `threads512`, and `readonly` variants. The new
`--check-only` option omits timing loops for sanitizer runs.

Old/new/new/old process order, four alternating timing batches per process,
5184 x 4736, median of eight batch medians per implementation:

| Path | Previous (ms) | CUB (ms) |
| --- | ---: | ---: |
| Separate restore + quantize | 1.524810 | 1.520390 |
| Fused restore + quantize | 0.701704 | 0.695944 |

These differences are small; treat performance as equivalent, not a demonstrated
speedup. The old executable was retained before relinking; each benchmark
executable compiles its own quantization kernels. The new benchmark additionally
uses a nondefault stream and performs untimed CPU reference checks.

Full-model truck, bag, child, wheel, and empty cases match the pre-refactor
`pixel-ba018a0/quality-fused_nchw-*` masks, scores, boxes, and query IDs byte for
byte. Configuration: INT8 MLP boundary + INT8 QKV, fused RoPE, Kitchen sequence
attention, fused FC2/norm, and fused NCHW pixel conversion. This verifies no output
change in that existing opt-in configuration; it does not establish new model
accuracy or an end-to-end speedup.

Compute Sanitizer memcheck: zero errors; racecheck filtered to `quant_rows`
(including `restore_quant_rows`): zero errors or warnings. The full CTest suite
was not rerun in this pass; validation targeted the changed kernels and five
full-model output comparisons. Timing and variant results are under
`experiments/results/native_rtx2060/cub-*.json`; build/sanitizer logs and model
outputs are under `.cache/cub-refactor` and `.cache/native-perf/cub-refactor`.

## Library boundaries retained

- INT4/INT8 GEMMs already use CUTLASS; production INT8 also uses cuBLASLt.
- Welford normalization and MSE scale selection involve floating-point sums.
  Replacing their reduction order requires separate exactness/quality evaluation.
- RoPE, Hadamard, and Attention shuffles encode data layouts, not just reductions.
  A generic block primitive is not a drop-in replacement.
- The pixel conversion uses a padded shared-memory transpose for coalescing.
  A generic elementwise transform would not automatically preserve its memory
  access pattern. Keep that small specialized kernel for now.

For subsequent refactors, favor library primitives that remove algorithmic
bookkeeping without extra launches, intermediate tensors, or new dependencies.

The next pass, including symmetric INT4 adoption, an affine INT4 trial and an
NPP EDT probe, is recorded in [CUDA-X survey](CUDA_X_SURVEY.md).
