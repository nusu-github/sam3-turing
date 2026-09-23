# Generate exact vision positions from per-axis tables

Vision position encoding previously computed the same sine/cosine values across
every row or column, then kept several full-resolution Float intermediates alive
until concatenation and conversion finished. The CUDA path now computes Float
tables for the two axes and broadcasts/casts directly into the final output.
Every requested position tensor is still generated on every forward. There is
no persistent cache, image reuse, fixed prompt, resolution restriction or new
weight variant.

The frequency vector uses the same ATen expression as before. Normalization,
division and sine/cosine retain the original Float arithmetic, followed by the
original output conversion. The returned `[B,256,H,W]` tensor keeps its original
channel-last strides and independent storage. CPU, empty inputs and other
floating dtypes use the original ATen expression; the custom CUDA expansion
supports Float, Half, BFloat16 and Double. The additive C++ operator is
`sam3::vision_position_encoding`, declared in `sam3/vision_position.h`.

The existing norm/layout/residual fusion also now applies to FP32 CUDA inference.
When autocast is disabled, its projection input stays Float. FP16/BF16 behavior,
matrix multiplications, weights and public class layouts remain unchanged.

## Measurements

The installed native benchmark compares against `908e476`, which already includes
norm/QK fusion and in-place MLP GELU. FP16 isolates the position change; FP32 also
includes the normalization extension. Each result uses three alternating process
pairs per model, ten synchronized samples after three warmups, all heads and
positions, four CPU threads and TF32 disabled. No builds or GPU work overlap.
Loading, preprocessing, text, detection and tracking are outside the measurement.

| Model | Precision / batch | Before → after (ms) | Shorter |
|---|---|---:|---:|
| SAM3 | FP16 / 1 | 51.893 → 50.524 | 2.64% |
| SAM3.1 | FP16 / 1 | 53.592 → 52.519 | 2.00% |
| SAM3 | FP16 / 2 | 107.918 → 104.814 | 2.88% |
| SAM3.1 | FP16 / 2 | 111.990 → 108.990 | 2.68% |
| SAM3 | FP32 / 1 | 319.472 → 316.997 | 0.77% |
| SAM3.1 | FP32 / 1 | 363.563 → 361.104 | 0.68% |
| SAM3 | FP32 / 2 | 591.313 → 584.543 | 1.14% |
| SAM3.1 | FP32 / 2 | 648.041 → 641.915 | 0.95% |

For FP16 batch one, peak additional CUDA allocation changes from 278,798,848
bytes to 221,003,776 (SAM3) or 275,644,416 (SAM3.1). For batch two it changes from
556,548,608 to 443,154,432 or 552,534,016 bytes. Other intermediates/returned
features determine the remaining peak. FP32 peaks and all parameter allocations
are unchanged in these measurements. These are Blackwell component results,
not Windows/Turing or complete-predictor performance claims.

The standalone prototype explains the memory reduction: at FP16 `[1,256,288,288]`,
position generation alone changes from 1.267 ms / 215,097,856 peak extra bytes to
0.125 ms / 42,762,752 bytes. This is an operator microbenchmark, not a whole-model
speedup. Full per-process samples, including FP32 batch one, are in the validation
record and private evidence.

A separate direct comparison against pre-fusion `3a58138` measures the combined
norm/QK/GELU/position changes at 57.922→50.745 ms for SAM3 (12.39% shorter) and
59.858→52.606 ms for SAM3.1 (12.12% shorter), FP16 batch one with all heads/positions.
Its baseline already includes FP16 parameter residency and the original rotary
kernel. This is measured directly, not a sum of percentages from different runs.

## Validation

The complete CPU27/CUDA51 suites pass. After adding the final fallback guard for
other floating dtypes, position tests were repeated on CPU, CUDA and a nondefault
CUDA stream: 54 strict bit/layout cases per execution, including Float8 fallback,
empty dimensions, non-square grids, strided/zero-channel reference tensors and
batch 1/2. They also check independence from image values, independently owned
outputs, unchanged inputs and invalid-input rejection. The CPU core hash remains
identical after the CUDA-only dispatch guard.

The full actual-weight matrix retains 312 files exactly; additional FP32 batch-two
comparisons retain 28 files. The final dispatch guard is followed by another
128-file FP16/FP32 feature check. Nineteen C++/C owning API cases retain 1,940 files.
These are native regression comparisons, not proof that the unresolved
original-source FP16 differences have disappeared.
Disabling position integration retains another 100 feature/text files exactly.
Re-enabling it restores the pre-ablation core hash exactly. All preceding strong
exported symbols remain; the C++ position operator is additive. The Linux CUDA
library contains sm75 cubins for the axis kernel and all four expansion dtypes.
This is compilation evidence, not Turing execution.

The rebuilt SDK uses previous binary C/C++ consumers and adds consumers of the
new operator compiled against installed headers/metadata. After recovery, the
C image lifecycle retains 153 files and the semantic probe 243. All 35 workspace
runtime libraries load from that SDK with Python absent from PATH and no injected
library path. No Python or Triton runtime is loaded.

## Reproduce

`SAM3_FUSE_VISION_POSITION=ON` is the default. Setting it to OFF restores the
previous position expression in `VisionEncoder`; the standalone operator tests
still test the new function. `SAM3_FUSE_VISION_NORM` independently controls norm
fusion, including FP32. Both switches affect only `vision_encoder.cpp` compilation.

```sh
cmake -S native -B build/native -DSAM3_FUSE_VISION_POSITION=OFF
cmake --build build/native --target sam3_vision_fusion_benchmark --parallel 4
build/native/sam3_vision_fusion_benchmark STORE sam3 fp16 FRAME.ppm 1 before.json 10
cmake -S native -B build/native -DSAM3_FUSE_VISION_POSITION=ON
cmake --build build/native --target sam3_vision_fusion_benchmark --parallel 4
build/native/sam3_vision_fusion_benchmark STORE sam3 fp16 FRAME.ppm 1 after.json 10
```

Start from a configured build and the dependencies in [SDK.md](SDK.md). Use `.exe`
on Windows and follow [WINDOWS_TURING.md](WINDOWS_TURING.md), including both
architecture settings. Repeat for `sam3.1`, `fp32` or another positive batch size.
Do not relink a library while a process is using it. New Windows/Turing kernel
execution remains with the user; no GitHub Actions were used.

The private increment is `position-fusion-sdk-overlay/overlays.json`, after
`vision-gelu-sdk-overlay/overlays.json`. Dependencies and the single shared modular
weight store are reused. Reproducers, hashes, measurements and recovery evidence
are in `native-foundation/position-fusion-linux`; the public record is
[position-fusion-validation.json](position-fusion-validation.json).
