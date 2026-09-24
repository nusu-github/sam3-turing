# Windows RTX 2060 initial native image benchmark

Latest follow-up: [pixel conversion fusion and selective MLP INT8](PIXEL_AND_MLP_SCOPE.md).

Earlier: [extended INT8 attention checks and zero-copy output layout](KITCHEN_EXTENDED.md),
including the quality limits of the combined INT8 path and attention-only results.

See [ConvRot and ComfyKitchen experiments](CONVROT_KITCHEN.md) for the latest
rotation-assisted INT4 quality investigation and external SM75 INT8 attention
evaluation. These are opt-in experiments, not default model changes.

The latest [controlled fusion OFF/ON comparison](FUSION_AB.md) measured
573.98→540.91 ms overall (5.76% shorter) and 7.58% shorter vision intervals,
with byte-identical outputs in all eight processes. Both variants use `908e476`;
GELU reuse is common to both and its separate effect is excluded.

## Upstream integration, 2026-09-24

Updated to `908e476`, including imported Windows compatibility fixes, semantic
text reuse, vision normalization/residual/layout and paired-RoPE fusion, and
in-place exact GELU. The old local norm prototype is archived; current runtime
sources match upstream, with both upstream fusion options ON. Full build and
48/48 CTests pass on Windows/Turing. A single five-sample truck check measured
500.35 ms median and preserved masks, scores, boxes and query indices byte for
byte against the earlier native result. This is a smoke check, not a controlled
speedup comparison. See [update-908e476.json](update-908e476.json).

The subsequent [stage profile](STAGE_PROFILE.md) attributes about 77% of CUDA
elapsed time to vision and 22% to the detector on the truck workload.
The [cuDNN/kernel investigation](CUDNN_KERNELS.md) confirms active cuDNN and
identifies MLP GEMMs and attention as the leading vision costs.
The subsequent [GEMM search](GEMM_SEARCH.md) found no useful improvement from
the cuBLASLt preference or the tested heuristic candidates.
An opt-in [LayerNorm-to-FP16 fusion](NORM_CAST.md) subsequently reduced local
inference median by 1.89%, with exact outputs on the tested fixtures.

## Update after 19f8bf9

The latest CRC acceleration and fixed-FP16 compute storage were integrated with
the existing local Windows fixes. The latency tool now explicitly selects Half
storage for vision/text. The full build and 41/41 CTests pass. One preliminary
process (two warmups, five timed samples) measured 562.41 ms median, 4.892 s
model/text setup and 2.214 GiB inference peak allocated. The previous three
processes had 11.361–11.447 s setup and 3.615 GiB peak. This is an initial check,
not a full repeated comparison; startup timings also depend on file caching.
Truck masks, scores, boxes and query indices are byte-identical to the previous
native result. See [update-19f8bf9.json](update-19f8bf9.json). The original
measurements and their conditions below are retained as historical evidence.

## Initial results before 19f8bf9

Measured 2026-09-23. Native C++ and the Python Turing FP16 patch have roughly
equal latency on this workload; these measurements do not establish a native
speedup. Native uses substantially more allocated GPU memory.

| Implementation | Pooled median, 15 samples | Three process medians | Inference peak allocated |
|---|---:|---|---:|
| Native modular C++ FP16 | 570.87 ms | 565.31 / 576.47 / 573.61 ms | 3.615 GiB |
| Python patched eager FP16 | 565.82 ms | 567.28 / 563.99 / 565.09 ms | 1.997 GiB |

The native pooled median is 0.89% slower. This is a small exploratory sample,
not a statistically established regression. Historical CUDA 12.8 measurements
are not a valid direct baseline for this CUDA 13.0 comparison.

## Conditions and scope

- Windows 11, RTX 2060 Max-Q 6 GB, driver 610.88, sm_75.
- Standalone Release LibTorch 2.10.0+cu130 versus Python torch 2.10.0+cu130;
  MSVC 19.44.35228 and CUDA compiler 13.0.88 for native.
- SAM3 checkpoint revision `3c879f39826c281e95690f02c7821c4de09afae7`.
  Native export preserves the checkpoint tensors losslessly.
- `assets/images/truck.jpg`, prompt `truck`, resolution 1008, threshold 0.5,
  all 200 queries. Both retain text weights and cache text features.
- Four CPU threads, TF32 disabled, cuDNN benchmark disabled, no torch.compile.
- Each fresh process performs one cold inference, two warmups and five timed
  inferences. Order: native, Python, Python, native, native, Python. No concurrent
  inference workloads. Timing synchronizes CUDA and includes preprocessing,
  upload, vision, detection and full-resolution mask postprocessing; disk reads
  and model construction are excluded. Image features are recomputed each time.
- Native starts from a decoded CPU RGB tensor and preprocesses on GPU; Python
  uses the existing Pillow input path. The native modular path retains vision
  outputs until detection completes; this is not an owning-predictor benchmark.
- Native retains FP32 checkpoint weights, while Python's patch stores selected
  weights in FP16. Baseline allocated bytes: 3,442,050,048 versus 1,728,215,040.
  FP16 execution mode does not imply identical resident weight storage.
- Peak allocated includes the cold inference (native takes the maximum of cold
  and warm peaks). It is allocator accounting, not total physical VRAM usage.
  External whole-device NVML samples include setup and are recorded separately;
  Windows WDDM residency is not established by allocator statistics.

Native cold inference took 1.013–1.222 s after text encoding/model setup.
Cold/setup scopes differ from Python and should not be used for a startup
speedup claim. Python was used only to orchestrate/monitor the native executable;
the native process ran with Windows system directories only on PATH.

## Output agreement

Compared with Python patched eager outputs, matching detections by maximum
total mask IoU. This is implementation agreement, not ground-truth accuracy.

| Case | Count, both | Differing mask pixels, summed | Minimum matched IoU |
|---|---:|---:|---:|
| truck | 1 | 2 | 0.99999686 |
| paper bag | 4 | 0 | 1.0 |
| child | 6 | 5 | 0.99988392 |
| wheel | 4 | 0 | 1.0 |
| elephant (empty) | 0 | 0 | N/A |

All outputs checked were finite. Scores match exactly after conversion to
float32; maximum box-coordinate difference is 0.020752 pixels. Outputs are
not bit-identical. This does not validate video, SAM3.1, other hardware, other
prompts or the full dataset.

## Reproduce

Build using [the Windows instructions](../../../docs/native/WINDOWS_BUILD.md)
with CUDA enabled, including target `sam3_image_latency`. Export the checkpoint
with `native/tools/export_weights.py`. Convert the source JPEG to RGB PPM with
Pillow and write a UTF-8 prompt file containing exactly `truck` (no newline).
The native tool's interface is:

```text
sam3_image_latency STORE IMAGE.ppm BPE.gz PROMPT.txt WARMUPS REPEATS OUTPUT
```

From the repository root, with the paths prepared as above:

```powershell
.venv/Scripts/python.exe experiments/monitor_native_bench.py .cache/native-perf/native-1 build/native-windows-cu130/sam3_image_latency.exe .cache/native-weights-sam3 .cache/native-perf/truck.ppm sam3/assets/bpe_simple_vocab_16e6.txt.gz .cache/native-perf/prompt.txt 2 5 .cache/native-perf/native-1
.venv/Scripts/python.exe experiments/monitor_native_bench.py .cache/native-perf/python-1 .venv/Scripts/python.exe experiments/local_turing_bench.py --name patched_eager --checkpoint PATH_TO_SAM3_PT --output .cache/native-perf/python-1 --reps 5
```

Use distinct output directories for each fresh process. The checked-in
[summary.json](summary.json) contains raw timing samples, environment, memory
statistics and per-case comparison results. Full local telemetry and binary
outputs remain under `.cache/native-perf`. No checkpoint data is checked in.
