# Development history

This fork was developed in four phases between 2026-09-20 and 2026-09-24. Commit
messages describe each change; this page maps the milestones to the documents
that describe the current state. The day-by-day logs that preceded this summary
(`docs/native/PROGRESS.md`, `docs/native/ONBOARDING.md` and the per-change notes
merged into the topic documents) remain available in commit `7d7fddb`.

| Phase | Commits | Current documentation |
|---|---|---|
| 1. Python Turing patch | `4516efa`..`080bec0` (`main`) | [TURING_IMAGE_PATCH.md](TURING_IMAGE_PATCH.md), [experiments/NOTES.md](../experiments/NOTES.md) |
| 2. Python-free native runtime | `2d68f1f`..`ba018a0` | [native/README.md](native/README.md) |
| 3. Windows / RTX 2060 experiments | `e6d8e56`..`cfc27c6` | [experiments/results/native_rtx2060](../experiments/results/native_rtx2060/README.md) |
| 4. INT8/INT4 calibration research (ongoing) | `7f54f0b`..`7d7fddb` | [native/QUANT_RESEARCH_LOG.md](native/QUANT_RESEARCH_LOG.md) |

## Phase 1: Python Turing patch (2026-09-20 – 09-22)

Runtime monkeypatches for SAM 3 image inference on small CUDA GPUs: FP16
conversion, memory reductions, text caching, compilation, optional INT8/INT4,
CPU text encoding and fused mask packing. 376 candidates were measured on an
RTX 3090 in rounds 1–68; the selected configurations were then validated on an
RTX 2060 Max-Q 6 GB (`080bec0`). The sweep harness and candidate code were
removed from later branches and remain in `main`.

## Phase 2: Python-free native runtime (2026-09-22 – 09-23)

### Design constraints

These acceptance conditions were fixed at the start and still govern the runtime:

- Deployment needs no Python interpreter or Python library. Python may be used
  during development for reference inference, weight export and comparisons.
- Dependencies must support both Windows and Linux; no Linux-only inference path.
  Each OS is built and validated separately.
- Image and video features of SAM3 and SAM3.1 are preserved. Fixing prompts or
  reducing detection queries, object counts or resolution does not count as a
  solution.
- No per-use copies of the full weights: shared and model-specific modules are
  stored once and loaded on demand.
- Code lives in Git; weights and weight-derived artifacts live in the private
  Hugging Face bucket.

LibTorch/ATen C++ with ahead-of-time compiled CUDA kernels was chosen as the
main route. AOTInductor packages and ExecuTorch CUDA were not adopted: the
available documentation did not establish Windows CUDA, sm_75 and Triton-free
builds together. Turing hardware was not available during this phase, so the
runtime was developed and validated on a Linux Blackwell GPU with sm_75 cubins
compiled in; Windows/Turing execution was left to the user. No GitHub Actions
are used.

### Milestones

| Date | Commits | Milestone | Documentation |
|---|---|---|---|
| 09-22 | `2d68f1f`, `60ac73b`, `1e9a5b5`, `9774725` | Native probes; mask packing, resize, NMS, connected components and EDT without Triton | [COMPONENTS](native/COMPONENTS.md) |
| 09-22 | `895e105` | Lossless modular weight store; native text encoder | [WEIGHTS](native/WEIGHTS.md) |
| 09-22 | `475dc6a`..`4f1551f` | Vision trunk/necks, geometry encoder, ROIAlign, detector, image grounding, tokenizer, interactive image sessions | [COMPONENTS](native/COMPONENTS.md) |
| 09-22 | `a949767`..`043edd3` | SAM3 tracking: video heads, memory encoder/attention, temporal selection, sessions, real-frame input | [COMPONENTS](native/COMPONENTS.md) |
| 09-22 | `d8beba9`..`9f09d53` | SAM3.1 multiplex: state, propagation decoder, temporal memory, dynamic sessions, lossless history paging | [COMPONENTS](native/COMPONENTS.md) |
| 09-22 | `bf08141` | C ABI 1 for image and low-level video | [C_API](native/C_API.md) |
| 09-22 – 09-23 | `73933b0`..`162ccc7` | Video association, hotstart, occlusion, reconditioning, global memory updates | [VIDEO_PIPELINE](native/VIDEO_PIPELINE.md) |
| 09-23 | `edf352e`..`21a3696` | Object births/removals, update planning, shared detector/tracker frames | [VIDEO_PIPELINE](native/VIDEO_PIPELINE.md) |
| 09-23 | `c1a86ca`, `6b72b1e`, `bb86f98` | Coherent comparison with the source; SAM3.1 memory-stride fix; correction-history policy | [VALIDATION](native/VALIDATION.md) |
| 09-23 | `c789650`, `6f23da2` | Output buffering and final postprocessing; action routing and partial propagation | [VIDEO_PIPELINE](native/VIDEO_PIPELINE.md) |
| 09-23 | `82df870`, `915fe66`, `d0244c5` | SAM3 and SAM3.1 point/mask instance edits | [VIDEO_EDIT](native/VIDEO_EDIT.md) |
| 09-23 | `9f1315f`, `552658b`, `487e323` | Owning C++ predictor, TF32 audit, explicit image mode | [VIDEO_PREDICTOR](native/VIDEO_PREDICTOR.md) |
| 09-23 | `acc59c3`, `b9cd72f` | Owning predictor C ABI with shared cores; standalone LibTorch SDK | [PREDICTOR_C_API](native/PREDICTOR_C_API.md), [SDK](native/SDK.md) |
| 09-23 | `641d234` | Reverse-edit difference traced to a stale pointer in the source | [VALIDATION](native/VALIDATION.md) |
| 09-23 | `5fbea4c`..`e7a12a6` | FFmpeg/libjpeg media input, verified seeking, source preprocessing policies | [MEDIA_IO](native/MEDIA_IO.md), [VIDEO_PREPROCESS](native/VIDEO_PREPROCESS.md) |
| 09-23 | `8e2564b` | COCO-slice FP16 image quality audit | [IMAGE_PRECISION_AUDIT](native/IMAGE_PRECISION_AUDIT.md) |
| 09-23 | `843709a`..`8e75249` | Tracking ranks on multiple devices, logical-rank source replay, parallel workers | [VIDEO_MULTIDEVICE](native/VIDEO_MULTIDEVICE.md) |
| 09-23 | `592def8`, `f56703d`, `fccd8d1` | Postprocess copy removal, position-map selection, fused rotary kernel | [PERFORMANCE](native/PERFORMANCE.md) |
| 09-23 | `fe87ffb`, `e61e81b` | Packed CPU/disk displayed-mask cache and read-only fetch | [OUTPUT_CACHE](native/OUTPUT_CACHE.md) |
| 09-23 | `a939d44`, `19f8bf9`, `3a58138` | zlib CRC, FP16 parameter residency, semantic text reuse | [WEIGHTS](native/WEIGHTS.md), [PERFORMANCE](native/PERFORMANCE.md) |
| 09-23 | `c5e9dc6` | User-validated Windows/Turing compatibility fixes (41/41 tests) | [WINDOWS_BUILD](native/WINDOWS_BUILD.md) |
| 09-23 | `571edc8`, `908e476`, `1ee3440`, `ba018a0` | Vision norm/layout/RoPE fusion, in-place GELU, per-axis positions; recovery documentation | [PERFORMANCE](native/PERFORMANCE.md), [QUICKSTART_JA](native/QUICKSTART_JA.md) |

## Phase 3: Windows / RTX 2060 experiments (2026-09-24)

The runtime was rebuilt and profiled on an RTX 2060 Max-Q (SM75, Windows,
CUDA 13.0). Opt-in INT8 MLP/QKV paths, ComfyKitchen INT8 attention, INT4 FC2,
pixel-decoder conversion fusion and CUB/Thrust maintenance changes were added;
external attention donors, CUTLASS INT8 tiles and several fusion variants were
rejected. The vision encoder and CMake files were split into smaller modules.
Results and decisions: [experiments/results/native_rtx2060](../experiments/results/native_rtx2060/README.md),
[native/CUDA_LIBRARIES.md](native/CUDA_LIBRARIES.md).

## Phase 4: INT8/INT4 calibration research (2026-09-24, ongoing)

A research loop with fixed COCO calibration/development/holdout splits and
FP16-agreement gates, looking for a configuration faster than selective INT8.
Rounds 1–8 covered FC2 channel calibration, per-block scopes, attention scopes,
K mean centering and QKV channel calibration. See
[native/QUANT_RESEARCH_LOG.md](native/QUANT_RESEARCH_LOG.md).

## Repository cleanup (2026-09-25)

Concluded experiment code (the Python sweep, native experiment drivers and the
rejected native paths) was removed, and the per-change and daily documents were
merged into the topic documents listed above. Everything removed remains in
commit `7d7fddb`.
