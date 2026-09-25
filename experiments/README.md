# Experiments

Measurement and research tooling for this fork. Raw results are under
[`results/`](results/).

## Python image patch (concluded)

`local_turing_bench.py` reproduces the RTX 2060 table in the root README. Every
configuration runs in a fresh process and records OOMs, timings, memory and
output comparisons. Setup, method and results:
[results/local_rtx2060](results/local_rtx2060/README.md).

The RTX 3090 development sweep (rounds 1–68, 376 candidates) concluded on
2026-09-21. Its measurements remain in [results/README.md](results/README.md) and
[results/summary.csv](results/summary.csv); the decisions are summarized in
[NOTES.md](NOTES.md) and the public [image patch guide](../docs/TURING_IMAGE_PATCH.md).
Do not mix those RTX 3090 container numbers with the RTX 2060 measurements.

The sweep runner, candidate implementations, round configurations, video
experiments and archived reproduction helpers were removed after the sweep. They
are preserved in commit `080bec0` (`main`). To repeat an old comparison, use a
separate checkout, for example `git worktree add ../sam3-sweep 080bec0`.

## Native runtime (Windows / RTX 2060)

| Script | Purpose |
|---|---|
| `monitor_native_bench.py` | Run one native process (usually `sam3_image_latency`) with a Windows-only `PATH`, sampling whole-device NVML memory, clocks and power |
| `native_quality.py` | FP16-agreement gates (`compare`) and the 17 fixed regression cases |
| `profile_native_hotspots.py`, `analyze_native_hotspots.py` | Nsight Systems capture and NVTX-correlated GPU kernel attribution |
| `simulate_fp16_accumulation.py` | CPU-only estimate of GEMM rounding error for FP16/two-level accumulation, INT8 and BF16 on synthetic inputs ([survey](../docs/native/DATA_FREE_STRUCTURE_RESEARCH_20260925_JA.md)) |

The experiment index and decisions are in
[results/native_rtx2060/README.md](results/native_rtx2060/README.md). Drivers of
concluded experiments (`run_*_native.py`, `summarize_*_native.py` and similar)
were removed; they remain in commit `7d7fddb` together with the runtime they
measured.

The ongoing INT8/INT4 calibration research is driven by `run_quant_research.py`
and the related `prepare_quant_research_data.py`, `make_*`, `run_quant_*`,
`run_kitchen_center_bench.py`, `check_quant_observer_outputs.py` and
`summarize_quant_*` scripts. Its goal, fixed data splits, quality gates and
round-by-round results are in
[docs/native/QUANT_RESEARCH_LOG.md](../docs/native/QUANT_RESEARCH_LOG.md).
