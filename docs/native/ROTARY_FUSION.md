# Precompiled rotary encoding

The vision trunk and tracking memory attention use a C++ `rotary_embedding`
operation that fuses FP32 conversion, complex multiplication and conversion back
to the input dtype into one CUDA kernel. It accepts floating `[B,H,N,D]` values
and complex64 `[N,D/2]` frequencies. Batch size, token count, head count and even
head width are runtime dimensions. It does not change prompts, object limits,
model weights, attention selection or tracker history.

The CUDA path supports FP32, FP16 and BF16 values with contiguous final dimension
and ordinary contiguous frequencies. Other cases, including CPU, FP64, conjugate
frequency views and empty values, use the original ATen expression. Inputs are
not modified. The operation is available through `sam3/rotary.h`; existing C/C++
predictor APIs select it automatically without changing their ABI.

## Arithmetic and layout

The kernel uses the same `c10::complex<float>` multiplication as the standalone
LibTorch baseline. Disabling FMA or substituting hand-written formulas changed
results in an initial experiment, even after conversion to FP16/BF16; those
formulas were not adopted. The final implementation preserves tensor values and
layout on the recorded comparison cases, rather than relaxing tolerances.

Splitting the last dimension into complex pairs and using ATen TensorIterator's
allocation rules preserves strides, including size-1 dimensions. Allocating with
`empty_like` alone was insufficient. The pair allocation is reinterpreted as the
input dtype; no FP32 or complex intermediate tensor is materialized on this path.
Empty tensors retain the original expression because their stride normalization
differs. Device guarding and the caller's current CUDA stream are retained.

The attached `cppdocs/installing.md` documents LibTorch CMake and Windows builds;
`pytorch/2.14/cpp_extension.md` distinguishes ATen/libtorch APIs from Python
bindings and describes CUDA compilation. This implementation uses CMake/nvcc
ahead of time, the existing native library and ATen interfaces. It adds no
Python binding, Triton, runtime compiler or OS-specific inference dependency.
Windows/Turing physical verification remains user-owned; no GitHub Actions run.

## Validation and artifacts

On RTX PRO 4500 Blackwell with standalone LibTorch 2.10.0 cu130, three independent
FP16 batch-1 baseline/current process pairs measured the following median time
reductions. Each process warms both position policies and records ten synchronized
samples per role; these comparisons use only the selected-position policy shared
by both versions. Process order alternates, and no other GPU work overlaps.

| Model | Video encoder | Grounding encoder | Interactive encoder |
| --- | ---: | ---: | ---: |
| SAM3 | 3.30% | 3.45% | 3.48% |
| SAM3.1 | 3.35% | 3.56% | 3.61% |

The measured encoder-wide peak additional allocation remains unchanged. These
are encoder timings, not whole-predictor throughput, memory savings or Turing
performance. Diagnostic Nsight captures include initialization and are used for
kernel counts and output comparison, not these timing percentages.
The four-frame edit lifecycle retains all 68 output files and reduces kernel
instances from 46,424 to 44,872, including 800 invocations of the fused kernel.

Detailed results, measured scope, baseline identity and artifact references are
recorded in `rotary-fusion-validation.json`. Tests compare bytes, dtype, shape,
strides and input preservation across strided QKV/partial-memory layouts,
expanded batches, all three CUDA dtypes, fallback frequency views, nonfinite
values, signed zero, subnormal values and empty tensors. A separate executable
checks a nondefault CUDA stream.

CPU21/CUDA38 CTests pass, including 44 rotary cases per device/stream and two
invalid-input checks. The eight actual-weight feature cases retain every binary
tensor and layout, and 19 API regressions retain all 774 output files. These
nonempty neural fixtures ran before adding the empty-input fallback guard; final
CTests and benchmark runs include that guard. The library identities are retained
in the validation JSON. The final Nsight and recovered SDK checks also exercise
the final library on real inference inputs.
An additional installed-SDK audit retains bytes and strides on 240 small layout
cases on each of CPU and CUDA, including singleton token/head dimensions and
broadcasting along batch, head or token dimensions. Complete SDK reconstruction
verifies 154 CPU and 188 CUDA entries. Recovered operator/stream tests pass, and
the recovered pure-C predictor retains 153 image-lifecycle output files while
loading all 35 workspace libraries from the recovered SDK without Python on PATH.

Actual-weight vision comparisons use both models in FP16/BF16/FP32 batch 1 and
FP16 batch 2. Image/video/interactive regressions compare the preceding SDK's
recorded outputs. These fixtures establish exactness on the recorded cases;
they do not resolve the earlier FP16 source-reference residual described in
VIDEO_COLLECTIVE_REFERENCE.md or establish all-input equivalence.

The private bucket `RamRom/sam3-turing-native-20260922` retains the SDK recipe at
`rotary-sdk-overlay/overlays.json` and evidence at
`native-foundation/rotary-fusion-linux`. The recipe reuses the position-selection
SDK's dependency and shared-weight layers. Feature comparisons retain binary
hashes, metadata and a reproducible standalone client; temporary full feature
arrays can be regenerated from the unchanged shared weights and input frame.
