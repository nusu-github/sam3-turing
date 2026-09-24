# Native runtime maintenance pass — 2026-09-24

## Scope

Separate responsibilities accumulated during the Windows/Turing experiments without
changing kernels, numerical expressions, default modes, environment variable names,
or the public `VisionEncoder` class. No dependencies are added.

## Code map

| File | Responsibility |
|---|---|
| `native/src/vision_encoder.cpp` | Weight loading, compute storage, pyramid/position generation and forward orchestration |
| `native/src/vision_attention.cpp` | Linear projections, QKV/RoPE and attention backend dispatch |
| `native/src/vision_block.cpp` | Residual/norm, MLP and preparation of the next block |
| `native/src/vision_experiments.h` | Private shared experiment selection and integer GEMM dispatch |
| `native/src/approx_kernels.h` | Private declarations shared by CUDA implementations, runtime and probes |
| `native/cmake/NativeSources.cmake` | Runtime source inventory, including the three vision implementation files |
| `native/cmake/Experiments.cmake` | Optional external backends, donor adaptations and associated probes |
| `native/cmake/Development.cmake` | Development tools, test registration and Windows runtime copying |

The root CMake file now describes dependency discovery, runtime configuration and
packaging. Its size decreases from 455 to 82 lines; the vision encoder entry file
decreases from 497 to 200 lines. Most of that is relocation, not removal of logic.
The common development-tool helper removes repeated executable/link declarations
while keeping target-specific options explicit.

## Invariants

- All `VisionEncoder` member bodies are identical to the saved pre-refactor version
  after ignoring the new explicit `detail::` qualifiers on experiment helpers.
- Inline shared helpers preserve function-local static environment caching. The
  three vision translation units receive the same feature definitions to avoid
  inconsistent conditional definitions between them.
- Existing probe targets still compile their diagnostic kernels directly. Shared
  declarations retain their existing linkage; they are not new installed SDK APIs.
- Existing compute streams, allocations, rounding points and profiling labels stay
  in the same operations. No kernel launch or numerical implementation is changed.
- CMake test names and commands are compared against the saved generated file.
  All 51 match. Existing object compile properties also match; only the two new
  vision object files are added.

## Validation artifacts

The pre-refactor DLL and generated build/test files are saved in
`.cache/maintenance-pass/`. Build logs, the alternate non-CUDA configuration and
old/new full-model comparisons are kept there as well. Model comparisons use
independent processes and compare masks, scores, boxes and query IDs byte for byte.

## Results

- CTest passes 51/51 (139.37 seconds), with unchanged test names and commands.
  The boundary, QKV/RoPE, FC2/norm, INT4 and INT8 restore probes also pass their
  existing `--check-only` validation. Test log:
  `experiments/results/native_rtx2060/maintenance-ctest.log`.
- CUDA Release runtime and all six probes consuming the consolidated declarations
  build successfully.
- With `SAM3_WITH_CUDA=OFF` and `SAM3_BUILD_DEVELOPMENT_TOOLS=OFF`, CMake configuration
  and all three vision object compilations pass. This is a conditional-compilation
  check using the installed CUDA-enabled LibTorch, not a full CPU-only SDK build.
- Full-model comparisons pass byte-for-byte for seven conditions: ordinary FP16
  on truck; the optimized INT8 boundary/QKV/Kitchen/fused-norm/pixel configuration
  on truck, bag, child, wheel and empty; and that configuration with INT4 FC2 on
  the last layer for truck.
- Truck smoke timings (3 warmups, 7 repeats, one old/new process pair each):
  ordinary FP16 494.68 → 500.65 ms; optimized 366.86 → 359.53 ms;
  last-layer INT4 364.88 → 355.72 ms. These are regression smoke checks with normal
  process/clock variability, not evidence of a performance improvement.
- Full-model outputs and timings: `experiments/results/native_rtx2060/maintenance-model.json`.

## Follow-up boundaries

Experiment implementations still live under `native/tools`, and the MLP block
still contains several experimental paths. Moving implementations and redesigning
configuration are separate changes: first-use cached settings and per-call settings
must not silently become interchangeable. Benchmark driver consolidation is another
candidate; preserve historical result files and per-experiment configuration.

## Second pass: settings and quantized projections

`vision_experiments.h` now contains only setting reads, validation and layer
selection; it no longer includes ATen or kernel declarations. Tensor operations
live in the private `vision_quantization.h` instead.

- `read_experiment` and `checked_experiment` do not cache values. Call sites retain
  their original execution position, including conditional Kitchen layout checks.
- MLP scope, INT4 FC2 mode and INT8 LT mode retain their function-local static
  strings. Validation remains **after** initialization: an invalid first value
  remains cached even if the environment changes after the exception.
- Constructor MLP selection remains a read without newly added validation; the
  block still validates its live MLP setting. Unset values use the same defaults.
- `quantize_rows` owns the repeated allocation/quantization sequence. Callers keep
  the flattened FP16 input alive through restoration as before.
- `quantized_linear` shares the ordinary INT8 projection/GEMM/restoration path
  between attention projections and unfused MLP. Fused RoPE and MLP boundary
  paths reuse row quantization while retaining their distinct epilogues.
- CUDA kernels, rounding points, profiling labels and explicit FC2 dispatch stay
  unchanged. Public headers and exported model class layout are unchanged.

The new CPU regression tests exercise live updates, cached scope and INT4 mode,
and invalid first values in separate processes. Old/new model comparisons add
unfused INT8 with both attention projections and global-only INT8 MLP scope to
the first pass's seven conditions. Logs and baseline binaries are under
`.cache/maintenance-pass2/`.


Second-pass model validation: all nine old/new conditions match masks, scores,
boxes and query IDs byte-for-byte. Results are in
`experiments/results/native_rtx2060/maintenance2-model.json`.
Truck timing smoke checks (one pair each, 3 warmups / 7 repeats):

| Mode | Before ms | After ms |
|---|---:|---:|
| exact | 491.89 | 492.83 |
| optimized | 365.37 | 354.26 |
| int4 | 352.70 | 357.10 |
| unfused | 430.97 | 433.87 |
| global | 427.80 | 431.71 |

These short runs detect large regressions; they do not establish a speedup.

Second-pass CTest: **54/54 passed** (141.47 seconds), including the three setting
lifetime tests. The three vision sources also compile with native CUDA and
development tools disabled, using the same installed LibTorch as the first pass.
Logs: `experiments/results/native_rtx2060/maintenance2-ctest.log` and
`.cache/maintenance-pass2/cpu.log`.
