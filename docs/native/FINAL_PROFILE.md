# Final vision profile and rejected candidates

Runtime checkpoint: `1ee3440`. Nsight Systems 2025.5.1 captured one warmed FP16,
batch-one forward on RTX PRO 4500 Blackwell for each model and runtime. Every
head and position is returned. Capture starts after loading, preprocessing and
three warmups. These are diagnostic traces; use the separate alternating,
unprofiled measurements in [POSITION_FUSION.md](POSITION_FUSION.md) for speed claims.

| Model | Pre-fusion `3a58138` kernels | Current kernels | Current GEMM share of summed kernel time | Attention share |
|---|---:|---:|---:|---:|
| SAM3 | 711 | 416 | 67.94% | 11.95% |
| SAM3.1 | 705 | 430 | 65.81% | 11.60% |

The current residual/normalization kernels account for 6.15% / 5.90%; GELU for
2.79% / 2.65%; paired RoPE for 1.78% / 1.61%. Kernel execution covers 98.09% /
98.42% of the interval between the first kernel start and last kernel end. This
is a timeline measurement, not SM occupancy or tensor-core utilization. The
Blackwell trace uses LibTorch Flash attention; it does not identify or measure
the Turing attention backend. GEMM kernel names containing `relu` do not imply
that the model's GELU was replaced with ReLU.

A static resource inspection of the shipped **sm75** code finds 33 registers for
the Half→Half residual/norm kernel, 38 for norm→Half, 18 for paired Half RoPE,
20 for Half position expansion and 22 for the axis kernel. Their reported local
memory is zero. The axis kernel has a 32-byte stack. These are compiled resource
counts from the Linux binary, not measurements on Turing or a Windows compiler.

## Candidates retained as research evidence

| Candidate | Observed outcome | Decision |
|---|---|---|
| Image feature reuse | Depends on repeated-image usage; the user requested compute/pipeline work | Reverted; no inference validation claimed |
| Half GELU lookup table / vectorization | Exact Half patterns in operator tests, but inconsistent whole-workload/batch benefit | Not adopted |
| RoPE output allocation | 18 shape/dtype cases exact; actual-shape benefit negligible, batch-two regression | Not adopted |
| CUDA Graph replay | Small batch-one benefit, batch-two regression; roughly 394–396 MiB extra reserved memory at batch one, 798 MiB at batch two | Not adopted |
| Neck in-place GELU | Batch-one/two full-vision latency effectively unchanged; peak allocation unchanged | Not adopted; output parity was not tested |
| BLAS backend preference | Default / cuBLAS / cuBLASLt preferences yield effectively unchanged full-vision latency and peak allocation | Not adopted; preference alone does not guarantee a different operator implementation; output parity was not tested |
| Column-major Half projection weights | 200 real-weight feature/metadata files exact; batch-one latency worse by 14.70% / 13.11%, batch-two by 9.02% / 8.20%; active allocation also increased | Reverted |
| Alternative mask-cache packing | 60 cases exact, but slower CPU path and increased scratch | Not adopted |

Raw `.nsys-rep`, SQLite and CSV traces, diagnostic driver, kernel summary and
rejected experiment sources/results are in private `native-foundation/vision-final-profile`
and the corresponding `*-deferred` prefixes. They are not required deployment files.
The graph comparison retained its graph pool during its eager measurements; it
is a bounded experiment, not a production memory/performance claim.

## Next optimization priorities

The remaining largest target in this trace is the linear GEMM work in MLP and
QKV/output projections. Test a proposed GEMM implementation against real weights
and both model versions, including batch two and the complete predictor. A fused
GELU epilogue must preserve the intended GELU formula and the pre-activation Half
rounding point, or receive a separately justified numerical/quality evaluation.
Simply substituting the upstream BF16 epilogue is not an exact FP16 optimization.

For Windows/Turing, run the exact rotary, norm and position regressions first,
then profile the actual selected attention/GEMM implementations. The existing
native benchmark is available without Python:

```bat
sam3_vision_fusion_benchmark.exe STORE sam3 fp16 FRAME.ppm 1 sam3.json 10
sam3_vision_fusion_benchmark.exe STORE sam3.1 fp16 FRAME.ppm 1 sam31.json 10
```

It uses device zero, four CPU threads, TF32 disabled, three warmups and synchronized
samples. Repeat alternating baseline/current processes with no other GPU workload;
record GPU clocks, driver, LibTorch and build flags. Loading is reported separately
and is excluded from forward samples. This benchmark does not measure complete
image prompting or video propagation.
