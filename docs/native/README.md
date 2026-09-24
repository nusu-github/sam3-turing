# Native runtime documentation

Start with [QUICKSTART_JA.md](QUICKSTART_JA.md) (Japanese): what the runtime is,
how to recover the private release, how to call it and what is still open. Build
options, tests, probes and experiment switches are in
[native/README.md](../../native/README.md).

| Topic | Documents |
|---|---|
| Build and distribution | [WINDOWS_BUILD.md](WINDOWS_BUILD.md), [SDK.md](SDK.md), [WEIGHTS.md](WEIGHTS.md) |
| C and C++ APIs | [C_API.md](C_API.md) (common rules, image, low-level video), [PREDICTOR_C_API.md](PREDICTOR_C_API.md), [VIDEO_PREDICTOR.md](VIDEO_PREDICTOR.md) (owning predictor, image mode) |
| Predictor features | [VIDEO_EDIT.md](VIDEO_EDIT.md), [VIDEO_MULTIDEVICE.md](VIDEO_MULTIDEVICE.md), [OUTPUT_CACHE.md](OUTPUT_CACHE.md) |
| Input | [MEDIA_IO.md](MEDIA_IO.md), [VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md) |
| Internals | [COMPONENTS.md](COMPONENTS.md), [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md), [CUDA_LIBRARIES.md](CUDA_LIBRARIES.md) |
| Performance | [PERFORMANCE.md](PERFORMANCE.md) |
| Validation | [VALIDATION.md](VALIDATION.md), [IMAGE_PRECISION_AUDIT.md](IMAGE_PRECISION_AUDIT.md) |
| Quantization research (ongoing) | [QUANT_RESEARCH_LOG.md](QUANT_RESEARCH_LOG.md), [INT8_INT4_RESEARCH_20260924_JA.md](INT8_INT4_RESEARCH_20260924_JA.md) (literature survey), [Windows/RTX 2060 experiments](../../experiments/results/native_rtx2060/README.md) |
| History | [docs/HISTORY.md](../HISTORY.md) |

[evidence/](evidence/) holds the JSON reports written by the source-parity
scripts (`native/tests/*.py`), probes and benchmarks that the documents cite.
Full tensors, logs and build snapshots are kept in the private Hugging Face
bucket, not in Git.

Measurements in these documents come from two machines: the Linux RTX PRO 4500
Blackwell development host (most component timings and all source-parity runs)
and the Windows RTX 2060 Max-Q 6 GB target. Each document states which one it
used; do not transfer Blackwell timings to Turing.
