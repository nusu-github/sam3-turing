# Vision normalization, layout and paired rotary fusion

The vision encoder now combines adjacent operations on every forward. On this
Blackwell host, the installed native benchmark measured SAM3 at
57.778→53.316 ms (7.72% shorter) and SAM3.1 at 59.748→55.074 ms (7.82% shorter).
These are FP16 batch-one whole-vision timings with every available feature head
and position tensor. They exclude preprocessing, loading, text, detection and
tracking. They are not Windows/Turing measurements.

## What runs together

- FP32 LayerNorm, optional 24×24 window partition and projection dtype conversion.
- Attention inverse-window layout, FP32 residual addition, LayerNorm and MLP
  input conversion.
- MLP output residual addition and the next block's LayerNorm, window partition
  and QKV input conversion. The prepared input exists only inside that forward.
- Q and K rotary embeddings in one CUDA launch, retaining separate outputs.

The matrix multiplications and exact GELU remain the existing LibTorch operations.
There is no image reuse, prompt specialization, detection limit, new weight store,
or additional persistent feature cache. C ABI 1 and existing C++ class layouts
are retained. Existing C++ symbols remain exported; the new operators are additive.

The LayerNorm kernel preserves the reference Welford reduction/operation order.
Its PyTorch attribution and license are in `native/third_party`. Unsupported
layouts, CPU inputs, alignment and degenerate window grids take the reference
ATen expressions, preserving their output layout. The rotary fast path handles
the standard interleaved QKV layout; other layouts/types retain the original
per-tensor rotary operation. Half/BF16 conversions happen at the original stage.

Paired rotary explicitly follows the platform's observed imaginary-product FMA
order: `fma(b,c,round(a*d))` on Linux and `fma(a,d,round(b*c))` on Windows.
The Windows rule follows the supplied compatibility patch. Cancellation tests
compare against the running LibTorch; a different LibTorch build still requires
running the tests. See [WINDOWS_TURING.md](WINDOWS_TURING.md).

## Validation and measurement

CPU 26/26 and CUDA 48/48 CTests pass. Each applicable device/stream runs 240 norm,
360 attention residual/norm and 360 next-projection cases, plus 82 paired rotary
cases. Tests include invalid input, fallback layouts, cancellation, nonfinite
values, independent output storage and nondefault CUDA streams.

Twelve actual-weight feature cases retain all 312 compared files exactly, including
tensor layout metadata: both models, FP16 batch 1/2 with Half parameter storage,
and the original Float-storage constructor in FP16/BF16/FP32. Nineteen owning
C++/C API cases retain 1,940 files exactly. Disabling both integrations also
retains the two FP16 feature fixtures (100 files). These are native regression
comparisons; the original-source FP16 residual in
[VIDEO_COLLECTIVE_REFERENCE.md](VIDEO_COLLECTIVE_REFERENCE.md) remains unresolved.

The headline measurements use three alternating independent OFF/ON process pairs
per model, ten synchronized forwards after three warmups, four CPU threads and
TF32 disabled, without concurrent builds/GPU work. Median process medians are
reported. Vision parameter allocation is unchanged: 936,842,240 bytes for SAM3
and 949,690,880 for SAM3.1. Peak additional CUDA allocation changes from
277,750,272 to 278,798,848 bytes (+1 MiB). Earlier five-sample measurements used
a different consumer that also loaded text parameters; they are retained as
development evidence and superseded for the headline result.

A supplementary complete `add_prompt` benchmark alternates two input frames and
box labels, with fresh vision encoding on every call. Its process medians were
noisier: SAM3 130.738→123.768 ms, SAM3.1 149.974→150.678 ms. The SAM3.1 baseline
process medians ranged from 126.739 to 151.165 ms; this check does not establish a
whole-predictor speedup. The component result above is the supported performance
claim. All raw per-process samples and this external C++ consumer are retained.

The recovered SDK runs with Python absent from PATH and no injected library path.
The C image lifecycle retains 153 files, the semantic-replacement probe 243 files,
and loader tracing finds no Python/Triton runtime libraries. SDK validation probe
timings are not controlled performance measurements.

The Linux CUDA binary contains sm75 cubins for all 16 new kernel instantiations;
it also contains sm80/86/90/100/120 and compute120 PTX. LibTorch supplies explicit
NVCC `-gencode` flags and sets the normal CMake architecture variable to OFF, so
the build-info file records both that variable and the actual NVCC flags.
The user's Windows/Turing 41/41 result covers the earlier `19f8bf9` plus supplied
compatibility changes. It does not yet cover these new fused kernels.

## Reproduce and isolate the integrations

Use the SDK/build instructions and Windows dependency/toolset pins first. On
Windows set `TORCH_CUDA_ARCH_LIST=7.5` as well as
`-DCMAKE_CUDA_ARCHITECTURES=75`. Neither switch below changes weights or creates
a model variant. They affect only compilation of `vision_encoder.cpp`.

```sh
cmake -S native -B build/native -DSAM3_FUSE_VISION_NORM=OFF -DSAM3_FUSE_VISION_QK=OFF
cmake --build build/native --target sam3_vision_fusion_benchmark --parallel 4
build/native/sam3_vision_fusion_benchmark STORE sam3 fp16 FRAME.ppm 1 off.json 10
cmake -S native -B build/native -DSAM3_FUSE_VISION_NORM=ON -DSAM3_FUSE_VISION_QK=ON
cmake --build build/native --target sam3_vision_fusion_benchmark --parallel 4
build/native/sam3_vision_fusion_benchmark STORE sam3 fp16 FRAME.ppm 1 on.json 10
```

Use `.exe` on Windows and repeat for `sam3.1`. Batch/sample counts are positive
arguments, not model restrictions. The benchmark currently selects CUDA device 0.
Do not replace a DLL/shared library while a benchmark using it is running.
Run CTest after changing compiler, GPU or LibTorch; disabling integration switches
does not disable the exported fused-operator tests.

The private SDK increment is `vision-fusion-sdk-overlay/overlays.json`, applied
after `semantic-text-sdk-overlay/overlays.json`. Dependencies and the one shared
modular weight store are reused. Evidence, reproducer drivers and output hashes
are under `native-foundation/vision-fusion-linux`. See
[vision-fusion-validation.json](vision-fusion-validation.json) for the machine
readable record. No GitHub Actions or new Windows/Turing execution was performed.
