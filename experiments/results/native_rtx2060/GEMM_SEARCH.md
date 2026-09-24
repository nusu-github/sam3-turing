# MLP GEMM algorithm search

Measured 2026-09-23, RTX 2060 Max-Q, LibTorch 2.10.0+cu130, native FP16 compute
storage. No candidate established a useful model-level improvement. Runtime
BLAS defaults remain unchanged.

## Complete image inference

Fresh processes ran in default/Lt/Lt/default order, each with two warmups and
ten measured truck inferences. `TORCH_BLAS_PREFER_CUBLASLT` was explicitly set
to 0 or 1. All other settings matched the prior stage benchmark, including
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and cuDNN benchmark disabled.

| BLAS preference | Pooled wall median, 20 samples |
|---|---:|
| Default | 558.17 ms |
| Prefer cuBLASLt | 562.79 ms |

Masks, boxes, scores and query indices matched the existing native result byte
for byte in all four processes. The 0.83% difference is small and is not a
statistically established regression, but provides no evidence for enabling
the override. A backend preference does not guarantee a different kernel for
every operation. See [blas-comparison.json](blas-comparison.json).

## Direct heuristic search

Added diagnostic target `sam3_gemm_algorithms`, which compares `at::linear` with
explicit cuBLASLt candidates for the two MLP matrix shapes:

- FC1: `[5184,1024] @ [4736,1024].T + bias`.
- FC2: `[5184,4736] @ [1024,4736].T + bias`.

Inputs are seeded synthetic contiguous FP16 tensors, accumulation is FP32 and
the epilogue adds FP16 bias. The 32 MiB workspace limit returned 3 FC1 and 7 FC2
heuristic candidates, all with algorithm ID 5 but different configurations.
This is the heuristic candidate set, not an exhaustive search of all cuBLASLt
configurations. Tile, stage, split-K and workspace values are retained in JSON.

The probe isolates GEMMs; input casts, GELU, residuals and model execution are
excluded. It does not validate model accuracy or production speedups.

Initial one-direction measurements showed a misleading apparent advantage that
did not survive reverse-order timing. The final check used three fresh
processes, 200 ATen warmup calls per shape and four adjacent ATen/candidate
timing pairs per candidate, alternating order. Each timing block includes three
warmups and ten CUDA-event measurements. There are no correctness-check kernels
between the paired measurements. GPU clocks were not locked.

| Candidate | ATen median | Candidate median | Median paired ratio |
|---|---:|---:|---:|
| Best FC1 paired ratio, heuristic rank 1 | 2.687 ms | 2.836 ms | 1.048 |
| Best FC2 paired ratio, heuristic rank 0 | 2.656 ms | 2.633 ms | 0.990 |

FC1 alternatives were slower in the final paired comparison. FC2's best
candidate differed by only about 1%, with no established model-level benefit.
Nine of ten candidates exactly matched the ATen tensor on the synthetic fixture.
FC2 rank 1 used split-K=2 and differed in 1,422,432 of 5,308,416 elements, with
maximum absolute error 0.00390625; it did not improve paired timing either.
None was integrated into the model.

## Reproduction and artifacts

Build target `sam3_gemm_algorithms` in the existing CUDA development-tools build.
Run `sam3_gemm_algorithms OUTPUT.json`, or use
`experiments/monitor_native_bench.py OUTPUT_DIR EXE OUTPUT.json` from the repo
root. The output parent directory must exist. The executable needs the same
LibTorch/CUDA DLLs as the other development tools. For the backend test, set
`TORCH_BLAS_PREFER_CUBLASLT=0` or `1` in each fresh process before invoking the
existing image latency command. Do not change the process-wide setting in a
running model to emulate this comparison.

[gemm-paired.json](gemm-paired.json) contains the final paired measurements and
checks. [gemm-heuristics.json](gemm-heuristics.json) retains the earlier warmed
forward/reverse experiment for context. Raw logs and telemetry are under
`.cache/native-perf/gemm-*` and `.cache/native-perf/blas-*`. The new diagnostic
target builds and completes successfully. No runtime operators changed in this
investigation; the prior 41/41 runtime checks remain applicable.

The useful conclusion is to stop treating a BLAS setting or heuristic tile swap
as the main optimization opportunity on this configuration. Fusion of surrounding
operations or a more specialized kernel would require separate measurements and
precision validation; this experiment does not claim those will be faster.
