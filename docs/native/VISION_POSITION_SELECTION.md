# Omit unconsumed vision position maps

High-level image and video inference now asks the shared vision encoder for only
the position maps its consumers use. Detection/tracking needs level 2 (72×72).
Interactive image decoding uses its separate prompt positional encoding and
consumes no vision position map. The trunk, every requested feature-pyramid
level, all queries and all prompt routes are unchanged. No model weights or
additional model variants are introduced.

The existing three-argument C++ `VisionEncoder::forward` still returns every
position map. A new overload accepts explicit position levels:

```cpp
// Keep the full pyramid/trunk, generate only positions[2].
auto features = vision.forward(image, mode, {"convs"}, {2});
// An empty fourth argument explicitly requests no vision position maps.
auto interactive = vision.forward(image, mode, {"sam2_convs"}, {});
```

The positions vector retains its original length and level indices. Unrequested
entries are undefined tensors; requested entries preserve their values, dtype,
shape, device and strides. Invalid/out-of-range or duplicate indices fail before
neural work. The existing overload and class layout retain their ABI. High-level
callers select this automatically; the C ABI and option structures are unchanged.
In particular, an image created with grounding capability retains level 2 even
if its first request is interactive, so later grounding prompts remain possible.

## Measurements

An actual-weight audit compares the full-position and selected-position overloads
in the same build, with identical inputs and heads. It compares the trunk, all
pyramid tensors and every retained position tensor exactly before measuring.
Both policies warm up; ten alternating pairs then measure synchronized wall time
and CUDA allocator bytes. FP16 batch-1 cases use three independent processes.
There was no competing GPU workload during these timing samples. Hardware is the
available RTX PRO 4500 Blackwell, with standalone LibTorch 2.10.0 cu130.

The median of the three per-process FP16 reductions is:

| Model | Encoder use | Time reduction | Peak additional allocation, full → selected |
| --- | --- | ---: | ---: |
| SAM3 | Video | 2.15% | 264.88 → 160.57 MiB |
| SAM3 | Grounding image | 2.24% | 264.88 → 154.41 MiB |
| SAM3 | Interactive image | 2.38% | 264.88 → 154.41 MiB |
| SAM3.1 | Video | 1.96% | 264.88 → 213.09 MiB |
| SAM3.1 | Grounding image | 2.12% | 264.88 → 154.41 MiB |
| SAM3.1 | Interactive image | 2.21% | 264.88 → 154.41 MiB |

Additional allocation is measured above live model/input tensors at each call.
These are encoder-only measurements, not complete predictor throughput or total
process VRAM. High-level callers already released unused position maps after
encoding; the optimization removes their construction and transient lifetime,
not an equivalent amount of persistent session memory. Reserved allocator memory
and other phases may remain unchanged. FP32 image cases retain the same measured
peak because another operation determines that peak. No Turing speedup is claimed.

FP16/BF16/FP32 batch-1 and FP16 batch-2 cases retain exact required features for
both models and all three encoder uses. The initial Nsight capture is retained
as diagnostic evidence; it includes initialization and must not be treated as a
steady-state latency benchmark. Reproduction:

```bash
sam3_vision_positions_benchmark STORE sam3 cuda fp16 FRAME.ppm 1 10 report.json
sam3_vision_positions_benchmark STORE sam3.1 cuda bf16_reference FRAME.ppm 2 1 report.json
```

The second example is supported by the tool; the recorded batch-2 matrix uses
FP16. Inputs are test fixtures, not runtime image, prompt or batch limits.
The tool builds via `native/eval` against an installed SDK. CUDA timing/statistics
are enabled with `SAM3_BENCHMARK_CUDA=ON`; CPU execution records tensor equivalence
without claiming CUDA memory statistics.

Detailed regression results and private artifact references are recorded in
`vision-position-selection-validation.json`. CPU20/CUDA35 CTests pass. The 19
image/video/tracking regression cases retain all 774 output files byte-for-byte
against the preceding SDK. Independent installed-SDK CPU FP32 checks retain all
required features for both models. A CPU FP16 timing attempt was stopped after
approximately 11 minutes without a completed result; it establishes neither a
pass nor a model failure. The corrected C video regression runner and its initial
argument error are preserved with the evidence.

The before/after diagnostic captures retain all 68 output files and reduce CUDA
kernel instances from 47,040 to 46,424. They include initialization and are not
used for the timing percentages above. This change does not resolve the earlier
FP16 source-reference residual documented in VIDEO_COLLECTIVE_REFERENCE.md.

The private bucket `RamRom/sam3-turing-native-20260922` stores the incremental SDK
recipe at `vision-positions-sdk-overlay/overlays.json` and evidence at
`native-foundation/vision-position-selection-linux`. The recipe reuses the prior
parallel SDK, dependencies and shared weights. Complete CPU/CUDA reconstruction
verified 152/185 entries; a recovered CUDA client also passed feature comparisons
without Python on PATH and loaded its workspace libraries from the recovered SDK.

CPU/CUDA builds retain the existing
portable ATen interfaces. The attached `cppdocs/installing.md` remains the basis
for Windows/Linux SDK compatibility; this change adds no OS-specific inference
API or dependency. Windows/Turing physical validation remains user-owned. No
GitHub Actions were used.
