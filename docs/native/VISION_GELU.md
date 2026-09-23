# Reuse the MLP projection buffer for exact GELU

The ordinary vision MLP now applies `at::gelu_` to its freshly allocated first
linear projection. That tensor has no other consumers. This avoids a separate
activation output allocation and improves locality while preserving the exact
GELU expression and the original dtype/rounding point between GEMM and GELU.
The BF16-reference fused epilogue is unchanged. No custom activation, lookup
table, approximation, image cache, public interface or weight format is added.

Compared with [VISION_FUSION.md](VISION_FUSION.md)'s already fused encoder,
three alternating independent process pairs on Blackwell measure FP16 batch-one
SAM3 at 53.141→51.740 ms (2.64% shorter) and SAM3.1 at 55.069→53.625 ms
(2.62% shorter). The installed native whole-vision benchmark uses all available
heads/positions, ten synchronized samples after three warmups, four CPU threads
and TF32 disabled. No builds or other GPU jobs overlap these measurements.
Preprocessing, loading, text, detection and tracking are outside the timing.

The sampled whole-vision peak additional CUDA allocation remains 278,798,848
bytes for batch one: another part of the forward determines that peak. Parameter
allocation and the shared modular weight store are unchanged. These measurements
do not establish Windows/Turing performance or a whole-predictor speedup.

Batch two shows a smaller additional improvement: SAM3 108.549→107.965 ms
(0.54%) and SAM3.1 112.666→112.199 ms (0.41%). Its observed peak additional
allocation increases from 555,500,032 to 556,548,608 bytes (+1 MiB).

A direct combined comparison against checkpoint `3a58138`, before the vision
fusion work, measures SAM3 58.260→52.235 ms (10.34% shorter) and SAM3.1
60.134→53.980 ms (10.23% shorter), FP16 batch one. This uses another three
alternating process pairs with the same installed benchmark, rather than adding
percentages from separate measurements. The baseline already includes FP16
parameter residency and the earlier rotary kernel. Peak additional allocation
changes from 277,750,272 to 278,798,848 bytes (+1 MiB).

Both models retain all 312 feature files/layout records across twelve cases,
covering FP16 batch 1/2 with Half parameter storage and original Float-storage
FP16/BF16/FP32 operation. CPU26/CUDA48 tests pass. Exact native regression does
not resolve the original-source FP16 residual described in
[VIDEO_COLLECTIVE_REFERENCE.md](VIDEO_COLLECTIVE_REFERENCE.md).
Nineteen owning C++/C API cases retain 1,940 files exactly. The recovered SDK
retains 153 C lifecycle files and 243 semantic-probe files; loader tracing finds
all 35 workspace runtime libraries inside the recovered SDK, with no Python or
Triton. All preceding strong exported symbols are unchanged.

The private increment `vision-gelu-sdk-overlay/overlays.json` applies after
`vision-fusion-sdk-overlay/overlays.json`. It reuses existing binary C/C++ clients,
dependencies and weights, allowing the preceding consumers to test compatibility
with the new core. Evidence is in `native-foundation/vision-gelu-linux`; the
public record is [vision-gelu-validation.json](vision-gelu-validation.json).
The runtime and benchmark require no Python or Triton. Windows build instructions
and the user's earlier physical test scope remain in
[WINDOWS_TURING.md](WINDOWS_TURING.md). No GitHub Actions were used.

Two custom GELU experiments remain outside the runtime. An exhaustive FP16
lookup table did not provide a consistent benefit, and exact-expression CUDA
vectorization improved a synthetic batch-one kernel but not batch two. Their
sources and measurements are retained privately under `gelu-lut-deferred` and
`gelu-vector-deferred` within `native-foundation`.
