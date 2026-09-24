# Native SAM3/SAM3.1 runtime

A Python-independent ATen C++/CUDA library for SAM3 and SAM3.1 image and video
inference: owning image/video predictors, C ABI 1 and C++ APIs, tokenizer,
pre/postprocessing, SAM3.1 object multiplex, optional media input and tracking
ranks on several devices. It links LibTorch, ICU and zlib (plus optional FFmpeg,
libjpeg and OpenCV), never Python or Triton.

- Getting started and recovering the private release: [QUICKSTART_JA.md](../docs/native/QUICKSTART_JA.md)
- Documentation index: [docs/native/README.md](../docs/native/README.md)
- Windows build: [WINDOWS_BUILD.md](../docs/native/WINDOWS_BUILD.md); SDK and Linux: [SDK.md](../docs/native/SDK.md)

## Layout

| Path | Contents |
|---|---|
| `include/sam3/` | Public C++ headers and `c_api.h` |
| `src/` | Runtime sources, CUDA kernels and private headers |
| `tools/` | Development probes and benchmarks, weight export, opt-in experiment kernels compiled into the runtime (`approx_*.cu`, `pixel_transform.cu`, `kitchen_attention.cu`, `int4_experiment.cu`) |
| `tests/` | CTest sources (`*.cpp`, `*.c`) and Python source-parity scripts (`*.py`) |
| `eval/` | Out-of-tree consumers of an installed SDK and the COCO-slice evaluator |
| `examples/c-client/` | Minimal C client installed with the SDK |
| `cmake/` | Source lists, media, experiments, development tools, SDK install and runtime bundling |
| `third_party/` | License notices for adapted PyTorch, torchvision, Pillow, ftfy, Python and ICU material |

The vision encoder is split by responsibility: `vision_encoder.cpp` (weight
loading, compute storage, position generation, forward orchestration),
`vision_attention.cpp` (projections, QKV/RoPE, attention dispatch) and
`vision_block.cpp` (residual/norm, MLP, next-block preparation). Component design
notes are in [COMPONENTS.md](../docs/native/COMPONENTS.md).

## Building

Requirements: LibTorch (standalone distribution recommended; any compatible
PyTorch exposing `TorchConfig.cmake` works for development), ICU 72–74 (74.2
validated) and zlib. On Ubuntu the latter are `libicu-dev` and `zlib1g-dev`;
Windows uses the pinned vcpkg manifest ([WINDOWS_BUILD.md](../docs/native/WINDOWS_BUILD.md)).

```sh
cmake -S native -B build/native -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release -DSAM3_WITH_CUDA=ON -DSAM3_TEST_CUDA=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=75;120"
cmake --build build/native --config Release --parallel 4
ctest --test-dir build/native -C Release --output-on-failure
```

Choose CUDA architectures your toolkit supports (75 is Turing). For CPU-only
builds use `-DSAM3_WITH_CUDA=OFF -DSAM3_TEST_CUDA=OFF`. Tests fail on unavailable
requested devices instead of falling back.

| CMake option | Default | Effect |
|---|---|---|
| `SAM3_WITH_CUDA` | OFF | Build the precompiled CUDA kernels |
| `SAM3_TEST_CUDA` | OFF | Register CUDA tests (needs a GPU) |
| `SAM3_BUILD_DEVELOPMENT_TOOLS` | ON | Build tests and probes; OFF for SDK builds |
| `SAM3_FUSE_VISION_NORM`, `SAM3_FUSE_VISION_QK`, `SAM3_FUSE_VISION_POSITION` | ON | Exact vision fusions ([PERFORMANCE.md](../docs/native/PERFORMANCE.md)) |
| `SAM3_WITH_MEDIA` | OFF | FFmpeg/libjpeg file input ([MEDIA_IO.md](../docs/native/MEDIA_IO.md)) |
| `SAM3_WITH_OPENCV` | OFF | OpenCV source preprocessing policy ([VIDEO_PREPROCESS.md](../docs/native/VIDEO_PREPROCESS.md)) |
| `SAM3_EXPERIMENT_KITCHEN_ATTENTION` + `SAM3_KITCHEN_ROOT` | OFF | External ComfyKitchen INT8 attention (pinned checkout, not vendored) |
| `SAM3_EXPERIMENT_INT8_GEMM` + `SAM3_CUTLASS_ROOT` | OFF | External-CUTLASS INT4 FC2 experiment (name kept for existing build caches) |

Install and bundle a relocatable SDK with `cmake --install` and
`cmake/BundleRuntime.cmake` ([SDK.md](../docs/native/SDK.md)).

## Tests

CTest registers the C/C++ tests in `tests/` (62 with CUDA enabled). They need no
weights or Python.

The Python scripts in `tests/` compare native components with the original
repository code and actual weights ([VALIDATION.md](../docs/native/VALIDATION.md)).
They are development tools only and need a Python environment with this
repository installed, the checkpoints and, for CPU image references,
`numpy==1.26.4 scikit-image==0.25.2 tifffile==2025.6.11`. For example:

```sh
python native/tests/parity.py build/native/libsam3_native.so --cuda
python native/tests/geometry_parity.py build/native/libsam3_native.so /private/native-weights-v1 --checkpoint sam3=/private/sam3.pt --report /tmp/geometry.json
```

## Development probes

Probes take the weight store, model (`sam3`/`sam3.1`), device and precision
(`fp32`, `fp16`, `bf16_reference`) plus the listed inputs. Image inputs are
binary P6 PPM; prompt files are UTF-8. Fixed prompts or counts inside a probe are
regression fixtures, not library limits.

| Probe | Exercises |
|---|---|
| `sam3_weights STORE verify\|list PREFIX` | Weight store integrity ([WEIGHTS.md](../docs/native/WEIGHTS.md)) |
| `sam3_tokenize BPE.gz < prompts.txt`, `sam3_text STORE MODEL DEVICE TOKEN_ID...` | Tokenizer, text encoder |
| `sam3_vision`, `sam3_geometry`, `sam3_detector_transformer` | Vision trunk/necks, geometry encoder, detector on synthetic features |
| `sam3_image_probe STORE MODEL DEVICE MODE IMAGE.ppm OUT THRESHOLD (TOKEN_ID... \| --text-file BPE.gz PROMPT.txt)` | Complete image grounding; writes `OUT.json` and packed masks |
| `sam3_image_latency STORE IMAGE.ppm BPE.gz PROMPT.txt WARMUPS REPEATS OUT [--profile]` | Image latency and outputs used by the Windows experiments |
| `sam3_image_benchmark STORE MODEL DEVICE MODE MANIFEST.tsv BPE.gz OUT` | COCO-slice audit ([IMAGE_PRECISION_AUDIT.md](../docs/native/IMAGE_PRECISION_AUDIT.md)) |
| `sam3_interactive`, `sam3_interactive_image` | Interactive prompt encoder/decoder and image sessions |
| `sam3_video_heads`, `sam3_memory`, `sam3_memory_attention`, `sam3_temporal`, `sam3_tracking_frame`, `sam3_tracking_session` | SAM3 tracking components on synthetic features |
| `sam3_tracking_video` / `sam3_multiplex_video STORE DEVICE MODE FRAMES.txt COMMANDS.txt OUT [HISTORY_DIR]` | SAM3 / SAM3.1 sessions on real frames driven by a command file (below); SAM3.1 can page history |
| `sam3_multiplex_propagation`, `sam3_multiplex_temporal`, `sam3_multiplex_frame`, `sam3_multiplex_update`, `sam3_multiplex_history`, `sam3_multiplex_session` | SAM3.1 multiplex components; an optional last argument pages history to a directory |
| `sam3_video_pipeline_probe STORE MODEL DEVICE MODE FRAMES.txt BPE.gz PROMPT.txt OUT [--trace\|--partial-probe\|--edit-probe\|--edit-sequence-probe\|--edit-sequence-trace]` | Coherent detector/tracker pipeline with raw and final outputs, compared with the source ([VIDEO_PIPELINE.md](../docs/native/VIDEO_PIPELINE.md)) |
| `sam3_video_predictor_probe`, `sam3_image_predictor_probe` | Owning predictor ([VIDEO_PREDICTOR.md](../docs/native/VIDEO_PREDICTOR.md)) |
| `sam3_c_api_probe`, `sam3_predictor_c_probe`, `sam3_media_c_probe`, `sam3_media_predictor_c_probe`, `sam3_video_preprocess_c_probe` | Pure C11 clients |
| `sam3_video_collective_probe`, `sam3_video_multidevice_probe` | Tracking ranks ([VIDEO_MULTIDEVICE.md](../docs/native/VIDEO_MULTIDEVICE.md)) |
| `sam3_video_mask_cache_benchmark`, `sam3_video_cache_fetch_probe` | Displayed-mask cache ([OUTPUT_CACHE.md](../docs/native/OUTPUT_CACHE.md)) |
| `sam3_vision_fusion_benchmark`, `sam3_vision_positions_benchmark`, `sam3_compute_storage_probe`, `sam3_semantic_text_probe` | Optimizations ([PERFORMANCE.md](../docs/native/PERFORMANCE.md)) |
| `sam3_boundary_bench`, `sam3_qkv_rope_bench`, `sam3_fc2_norm_bench`, `sam3_pixel_bench`, `sam3_int4_bench`, `sam3_kitchen_bench` `OUT.json [--check-only]` | Operator checks and timings of the experiment kernels |

`sam3_tracking_video` / `sam3_multiplex_video` read a UTF-8 manifest of PPM frames
(one path per line, relative to the manifest) and a command file whose non-empty,
non-`#` lines are:

```text
points FRAME ID CLEAR_OLD NORMALIZED USE_PREVIOUS_MEMORY [X Y LABEL]...
box FRAME ID NORMALIZED X0 Y0 X1 Y1
mask FRAME ID PPM_PATH
masks FRAME [ID PPM_PATH]...      (SAM3.1 simultaneous brushes)
preflight ENCODE_MEMORY
propagate START MAX_STEPS REVERSE ENCODE_MEMORY PREFLIGHT STOP_AFTER USE_CANCEL
clear FRAME ID
remove ID
reset
```

Booleans are `0`/`1`; `START=-1` and `MAX_STEPS=-1` use session defaults. Mask
prompts use the PPM red channel divided by 255. Outputs are
`OPERATION-OUTPUT_INDEX` JSON metadata, packed masks and native-endian F32 logits.

Weight tooling (Python, development only): `tools/inventory_weights.py`,
`tools/export_weights.py` ([WEIGHTS.md](../docs/native/WEIGHTS.md)),
`tools/capture_image_reference.py` (reference tensors) and
`tools/generate_tokenizer_tables.py` (frozen Unicode tables).

## Experiment switches

The INT8/INT4 research paths are selected by environment variables read by the
vision encoder. Unset variables keep the exact FP16 arithmetic. Set them before
the process starts: scopes, the MLP part, K centering and the INT4 mode are cached
at first use, and invalid values raise errors. They require CUDA with FP16
compute storage. Research results: [QUANT_RESEARCH_LOG.md](../docs/native/QUANT_RESEARCH_LOG.md)
and [experiments/results/native_rtx2060](../experiments/results/native_rtx2060/README.md).

| Variable | Values | Effect |
|---|---|---|
| `SAM3_EXPERIMENT_MLP` | `exact`, `int8_boundary` | W8A8 vision MLP with fused FC1 restore/GELU/requantization |
| `SAM3_EXPERIMENT_MLP_SCOPE` | `all`, `global`, `local`, `firstN`, `lastN`, `mask:0xHEX` | Blocks using the INT8 MLP (bit 0 = block 0) |
| `SAM3_EXPERIMENT_MLP_PART` | `both`, `fc1`, `fc2` | Quantize only one MLP linear |
| `SAM3_EXPERIMENT_FC2_NORM` | `exact`, `fused` | Fuse FC2 restore, residual and next-block norm |
| `SAM3_EXPERIMENT_PROJECTION` | `exact`, `qkv` | W8A8 QKV projection |
| `SAM3_EXPERIMENT_PROJECTION_SCOPE` | scope syntax | Blocks using INT8 QKV |
| `SAM3_EXPERIMENT_QKV_ROPE` | `exact`, `fused` | Fuse QKV restore with RoPE (needs `qkv`) |
| `SAM3_EXPERIMENT_ATTENTION` | `exact`, `kitchen`, `kitchen_rot`, `kitchen_all`, `kitchen_rot_all` | ComfyKitchen INT8 attention on global (or all) layers; Kitchen build only |
| `SAM3_EXPERIMENT_ATTENTION_SCOPE` | scope syntax | Blocks using INT8 attention |
| `SAM3_EXPERIMENT_KITCHEN_LAYOUT` | `head`, `sequence` | Attention output layout (sequence avoids a copy) |
| `SAM3_EXPERIMENT_KITCHEN_CENTER`, `..._CENTER_SCOPE` | `anchor`, `mean`, `mean_half`, `none`; `all`, `global`, `local` | K centering before quantization (`mean` uses atomic sums and is not run-to-run deterministic) |
| `SAM3_EXPERIMENT_INT4_FC2` | `exact`, `all`, `last8`/`last4`/`last2`/`last`, `w4`, `a4`, `affine`, `a4_affine`, `mse`, `a4_mse`, `w4_mse`, `rht16`, `rht64` | W4A4 FC2 variants; CUTLASS build and `int8_boundary` only |
| `SAM3_EXPERIMENT_PIXEL` | `exact`, `fused_nchw` | Fused pixel-decoder layout conversion (exact) |
| `SAM3_EXPERIMENT_FC2_CALIBRATION`, `SAM3_EXPERIMENT_FC2_MEAN_BIAS` | directory; `exact`/`enabled` | FC2 channel scale/shift (`fc2-affine.f32.bin`) and weight mean-bias correction (`fc2-mean.f32.bin`) |
| `SAM3_EXPERIMENT_QKV_CALIBRATION`, `SAM3_EXPERIMENT_QKV_MEAN_BIAS`, `SAM3_EXPERIMENT_QKV_OUTPUT_BIAS` | directory; `exact`/`enabled` | QKV channel calibration folded into norm1, mean-bias and output-bias corrections |
| `SAM3_EXPERIMENT_OBSERVE_FC2`, `SAM3_EXPERIMENT_OBSERVE_QKV`, `SAM3_EXPERIMENT_OBSERVE_QKV_ERROR` + `SAM3_EXPERIMENT_QKV_SHADOW_CALIBRATION` | output directory | Write per-layer calibration statistics from the unquantized FP16 path |

Profiling: `SAM3_PROFILE_NVTX=1` enables NVTX ranges; `sam3_image_latency` also
reads `SAM3_PROFILE_CAPTURE` (cudaProfilerStart/Stop around timed samples) and
`SAM3_BENCH_CUDNN` (cuDNN benchmark mode).
