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

Native benchmarks run `sam3_image_latency` through `monitor_native_bench.py`,
which restricts `PATH` to Windows system directories and samples whole-device
NVML memory, clocks and power. The experiment index and decisions are in
[results/native_rtx2060/README.md](results/native_rtx2060/README.md).

The INT8/INT4 calibration research is driven by `run_quant_research.py` and the
related `prepare_*`, `make_*`, `run_quant_*` and `summarize_quant_*` scripts. Its
goal, fixed data splits, quality gates and round-by-round results are in
[docs/native/QUANT_RESEARCH_LOG.md](../docs/native/QUANT_RESEARCH_LOG.md).
