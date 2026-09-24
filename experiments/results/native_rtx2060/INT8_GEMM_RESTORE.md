# INT8 GEMM + FP16 restoration — 2026-09-24

## Decision

Keep the existing ATen GEMM followed by fused FC2/residual/next-norm. The new SM75 GEMM epilogue eliminates the INT32 intermediate and reproduces the checked outputs exactly, but the complete FC2-to-next-norm operator chain is **9.25% slower** in this microbenchmark. It is a development-only experiment, not connected to model dispatch. No whole-model speedup or accuracy claim is made.

This follows the [INT8 GEMM search](INT8_GEMM_SEARCH.md). Rather than repeating tile tuning, it tests whether avoiding the intermediate tensor compensates for the custom GEMM's overhead.

## Implementation and comparison

`native/tools/int8_gemm_restore.cu` uses the same separately licensed external CUTLASS checkout/compatibility header as the preceding search. It uses SM75 INT8 tensor operations, INT32 accumulators, a 128×128×64 CTA tile, 64×64×64 warp tile and two pipeline stages. CUTLASS visitor callbacks apply activation scale per row, weight scale per output channel and FP16 bias, then round to FP16 and store directly. The compiled SM75 kernel uses 216 registers/thread, with zero stack frame and zero spill loads/stores reported by ptxas. The visitor type name `Sm80EVT` does not change the explicitly selected SM75 GEMM architecture.

The wrapper validates dtype, dimensions, contiguity, device, bounded K and alignment-compatible sizes, and runs on the current stream under a device guard. The development build remains behind `SAM3_EXPERIMENT_INT8_GEMM=ON`; the production inference path does not call the new API.

For M=5184, N=1024, K=4736, two separate processes each run four forward/reverse passes across four methods, with ten warmups and twenty CUDA-event samples per method. An initial 100 calls warm the existing GEMM+fused-norm path. The table pools eight pass medians per method. Output allocations and host wrapper calls are inside the measured sequence. No concurrent GPU/build job or clock changes were used; clocks and caches are not controlled, so absolute times must not be compared with earlier reports.

| Work measured | Existing path | Experimental path | Result |
|---|---:|---:|---|
| GEMM + scale/bias + FP16 output | 1.4078 ms | 1.3873 ms | 1.46% lower pooled median; small and inconsistent across passes |
| GEMM through residual + next window-partitioned Norm | **1.5282 ms** | 1.6695 ms | **9.25% slower** |

The second row is the relevant comparison for the already optimized intermediate Vision blocks. Existing code calls ATen `_int_mm` followed by `approx_fc2_norm`. The experiment calls GEMM+restore followed by `vision_residual_norm_projection`. The latter retains a materialized FP16 projection and a separate residual/Norm kernel. Both return bit-identical FP32 residuals and FP16 normalized tensors on these checks.

Removing the INT32 tensor avoids a logical 20.25 MiB write plus 20.25 MiB read per FC2. Against the existing fused-norm path, however, the new path adds a 10.125 MiB FP16 write plus read. Net logical intermediate traffic falls by 20.25 MiB, but measured time still rises. These are tensor byte counts, not measured DRAM traffic; caching and kernel scheduling matter. The measurements do not identify a unique hardware-level cause of the regression.

A direct GEMM+residual+Norm epilogue is not a trivial extension: this GEMM tile owns 128 of the 1024 output channels, whereas LayerNorm reduces the entire row. It would require a different tile/reduction or synchronization scheme. That option remains untested, and this result does not rule out every possible fused GEMM design.

## Validation and reproducibility

`sam3_int8_restore_bench` checks a tail tile (80,80,128) and the production FC2 shape on a nondefault stream. Cases include full-range signed INT8 random inputs, constant 127 (large INT32 accumulation), zero inputs, and regenerated random inputs. Scales and bias vary by row/channel. FP16 restored output is checked by its INT16 bit representation; production-shape residual and partitioned next Norm are checked by INT32/INT16 bit representations. All cases pass.

Compute Sanitizer memcheck and racecheck, filtered to CUTLASS kernels, complete all cases with **zero errors and zero race hazards/warnings**. Racecheck cases are also recorded in [their JSON](int8-restore-racecheck.json). These are operator checks, not a new full-model quality evaluation. The previous 51-test suite result belongs to the preceding report; it is not counted as a fresh run here.

Build the `sam3_int8_restore_bench` target in the existing CUDA 13 / MSVC 14.44 development configuration. Run through the existing native environment monitor:

```powershell
.venv/Scripts/python.exe experiments/monitor_native_bench.py .cache/native-perf/int8-restore-micro-1 build/native-windows-cu130/sam3_int8_restore_bench.exe experiments/results/native_rtx2060/int8-restore-micro-1.json
```

Add `--check-only` after the JSON path to skip timings. Data: [run 1](int8-restore-micro-1.json), [run 2](int8-restore-micro-2.json), [initial correctness](int8-restore-check.json), [memcheck cases](int8-restore-memcheck.json). Corresponding raw process logs and telemetry are under `.cache/native-perf/int8-restore-*`; build log is `build-int8-restore-2.log` in that directory's parent.

## Next useful work

The already profiled global/local Attention and MLP projections remain substantial costs. After this negative FC2 result, an INT8 QK attention experiment has a stronger reason to investigate than repeating plain GEMM replacements. It must include quantization and scale overhead, preserve the FP16 PV baseline, and pass the existing output-quality gates. That experiment has not been implemented by this change.
