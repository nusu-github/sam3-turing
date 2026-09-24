# Documentation map

## Python Turing patch (`sam3/turing*.py`)

- [TURING_IMAGE_PATCH.md](TURING_IMAGE_PATCH.md): usage, optional features, RTX 3090 measurements
- [experiments/results/local_rtx2060](../experiments/results/local_rtx2060/README.md): RTX 2060 Max-Q validation and uv setup
- [experiments/NOTES.md](../experiments/NOTES.md) (Japanese): which candidates were adopted or rejected and why
- [experiments/results/README.md](../experiments/results/README.md): every measured candidate
- [TURING_VIDEO_EXPERIMENTS.md](TURING_VIDEO_EXPERIMENTS.md): SAM 3.1 video monkeypatch results (RTX 3090 only)
- [patches/turing-image.patch](../patches/turing-image.patch): standalone patch for an upstream checkout

## Native C/C++ runtime (`native/`)

- [native/QUICKSTART_JA.md](native/QUICKSTART_JA.md) (Japanese): overview, recovery, usage and open issues
- [native/README.md](native/README.md): index of the native documents
- [native/README.md in the source tree](../native/README.md): build options, tests, probes, experiment switches

## Windows / RTX 2060 experiments and quantization research

- [experiments/README.md](../experiments/README.md): measurement and research tooling
- [experiments/results/native_rtx2060](../experiments/results/native_rtx2060/README.md): experiment index and decisions
- [native/QUANT_RESEARCH_LOG.md](native/QUANT_RESEARCH_LOG.md) (Japanese): ongoing INT8/INT4 calibration research

## History

- [HISTORY.md](HISTORY.md): phases, milestones and design constraints

## Upstream SAM 3

The [root README](../README.md) keeps the upstream introduction below the fork's
sections; see also [README_TRAIN.md](../README_TRAIN.md),
[RELEASE_SAM3p1.md](../RELEASE_SAM3p1.md) and the SA-Co evaluation guides in
[scripts/eval](../scripts/eval/).
