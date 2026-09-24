# Windows/Turing follow-up for upstream review

Base: `ba018a0` on `codex/native-onboarding`. Tested platform: RTX 2060 Max-Q
(SM75), Windows, MSVC 14.44, CUDA 13.0 / CCCL 3.0.1, LibTorch 2.10 cu130.

## What this branch adds

- Opt-in INT8 MLP/projection, fused QKV restoration/RoPE, FC1-to-FC2 boundary,
  FC2/residual/next-norm and pixel-layout experiments, with profiling and probes.
- Optional external Turing attention, ComfyKitchen and CUTLASS INT4/INT8 adapters.
  External sources remain outside this repository and are selected explicitly
  through CMake options/paths; see the individual experiment reports for pins.
- CUB row reductions and component-size/mask transforms; Thrust tabulate for
  unpacking and position-axis generation. The accepted Thrust changes preserve
  exact outputs; rejected substitutions and timing variation are documented.
  These CUDA sources were validated against CUDA 13.0's bundled CCCL 3.0.1.
- Native CMake modules, split vision encoder responsibilities, shared private
  CUDA declarations, common quantized-linear helpers and experiment-setting tests.

Approximate modes remain opt-in. INT4 and some combined INT8 configurations fail
quality gates against the FP16 reference and are not promoted to defaults.
Exact old/new results in the refactoring checks mean parity with the same mode
before refactoring, not equivalence of quantized modes to FP16.

## Final validation

- CTest: **54/54**, including live/cached/invalid experiment-setting lifetimes.
- Full-model old/new comparison: all masks, scores, boxes and query IDs match
  byte-for-byte in nine conditions (ordinary FP16; optimized truck/bag/child/wheel/
  empty; last-layer INT4; unfused INT8 projections/MLP; global-only INT8 MLP).
- Native CUDA and development tools disabled: configuration and the three vision
  source compilations pass using installed CUDA-enabled LibTorch. This is not a
  full CPU-only SDK or Linux validation.
- Earlier maintenance passes include exact operation comparisons, nondefault
  streams, CUDA Graph replay and Compute Sanitizer checks; see their reports.

No performance gains from separate experiments should be added together. The
final refactor timing checks are short regression checks, not new speedup claims.
Model weights, donor checkouts, local caches and build products are not shipped.

## Review map

- [Refactoring and final nine-condition evidence](REFACTORING.md)
- [CCCL high-level API investigation and accepted changes](CCCL_HIGH_LEVEL_REVIEW.md)
- [CUB inventory and rejected alternatives](CUB_WIDE_REVIEW.md)
- [CUDA library maintenance validation](CUDA_LIBRARY_MAINTENANCE.md)
- [Windows build instructions](WINDOWS_BUILD.md)
- [Experiment evidence index](../../experiments/results/native_rtx2060/README.md)
- [Quantization quality limits and selective scope](../../experiments/results/native_rtx2060/PIXEL_AND_MLP_SCOPE.md)

Suggested review order: runtime/build layout, exact CUDA maintenance changes,
optional experiment plumbing, then individual approximation results. Historical
reports retain the commit/configuration they measured; this document and the
refactoring report describe the final handoff state.
