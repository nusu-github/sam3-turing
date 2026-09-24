# Windows / RTX 2060 native experiments

Measurements of the native runtime on the target laptop GPU: RTX 2060 Max-Q 6 GB
(SM75), Windows 11, driver 610.88, CUDA 13.0, MSVC 14.44, standalone LibTorch
2.10.0+cu130. Build: [WINDOWS_BUILD.md](../../../docs/native/WINDOWS_BUILD.md).

Unless a report says otherwise, the workload is one image (`truck.jpg`, prompt
`truck`) at 1008 px with all 200 queries, threshold 0.5, cached text and fresh
image features; disk I/O and setup are excluded. Timings come from
`sam3_image_latency` run through
[`experiments/monitor_native_bench.py`](../../monitor_native_bench.py) in fresh,
alternating processes. GPU clocks were not fixed: compare numbers only within
one report's series and never add gains from different reports.

Approximate modes are judged by agreement with FP16 (not ground truth): finite
outputs, equal detection counts, Hungarian-matched minimum mask IoU ≥ .98, score
difference ≤ .02 and box difference ≤ 1 px, over the 5 original and 12 extended
cases defined in [`experiments/native_quality.py`](../../native_quality.py).
The later INT8/INT4 research uses separate COCO splits.

## Current state

- **Default runtime**: exact FP16, unchanged by every experiment below. The
  upstream exact fusions are on by default and measured 573.98 → 540.91 ms here
  ([FUSION_AB](FUSION_AB.md)).
- **Research baseline ("selective")**, all opt-in through
  [experiment switches](../../../native/README.md#experiment-switches):
  `SAM3_EXPERIMENT_PIXEL=fused_nchw` (exact), ComfyKitchen INT8 attention
  (`kitchen_all`, `sequence` layout), W8A8 QKV with fused restore/RoPE,
  `int8_boundary` MLP on the 4 global blocks and fused FC2/norm. It passes the
  17 regression cases but only 31/33 COCO development prompts; the ongoing
  search for a faster configuration that passes everywhere is in
  [QUANT_RESEARCH_LOG.md](../../../docs/native/QUANT_RESEARCH_LOG.md).
- Rejected paths were removed from the runtime in the 2026-09-25 cleanup
  ([cleanup-parity.json](cleanup-parity.json): 34 retained-mode runs byte-identical,
  CTest 62/62). Their code, and the per-experiment drivers, remain in commit
  `7d7fddb`; the reports below are kept as records.

## Experiments

| Report | Question | Result | Decision |
|---|---|---|---|
| [Initial benchmark](#initial-native-benchmark-2026-09-23) (below) | Native vs Python Turing patch | 570.87 vs 565.82 ms; native allocated 3.615 vs 1.997 GiB | Memory gap closed by FP16 parameter residency |
| [STAGE_PROFILE](STAGE_PROFILE.md) | Where does time go? | Vision 77%, detector 22% of CUDA time | Profile |
| [CUDNN_KERNELS](CUDNN_KERNELS.md) | Is cuDNN used; does benchmark mode help? | cuDNN active; benchmark mode 0.12%, +0.63 s cold | Not adopted |
| [GEMM_SEARCH](GEMM_SEARCH.md) | cuBLASLt preference / explicit heuristics for FP16 MLP | No model-level gain | Not adopted |
| [NORM_CAST](NORM_CAST.md) | LayerNorm-to-FP16 fusion prototype | −1.89%, exact | Superseded by the upstream fusion |
| [FUSION_AB](FUSION_AB.md) | Upstream exact fusions OFF vs ON | −5.76% overall, byte-identical | Default ON |
| [CURRENT_HOTSPOTS](CURRENT_HOTSPOTS.md) | NVTX-attributed kernel time, FP16 vs INT8 MLP | Vision MLP 187 → 143 ms | Profile |
| [APPROXIMATION_RESEARCH](APPROXIMATION_RESEARCH.md) (JA) | Literature plan for non-bit-exact options | Prioritize selective W8A8 | Plan |
| [APPROXIMATION_EXPERIMENTS](APPROXIMATION_EXPERIMENTS.md) (JA) | tanh GELU, fused addmm, W8A8 MLP | INT8 MLP −7.77%; others no gain | INT8 kept (as `int8_boundary`); rest removed |
| [BOUNDARY_FUSION](BOUNDARY_FUSION.md) | Fuse FC1 restore/GELU/requantization | Faster, byte-identical to unfused INT8 | Opt-in `int8_boundary` |
| [NEXT_OPTIMIZATIONS](NEXT_OPTIMIZATIONS.md) | QKV / output-projection INT8, row restore | QKV −1.8 to −1.9%; others no gain | QKV opt-in; others removed |
| [PORTABILITY_REASSESSMENT](PORTABILITY_REASSESSMENT.md) (JA) | Re-check an external portability review | Next: global attention d=64, QKV restore/RoPE | Plan |
| [TURING_ATTENTION](TURING_ATTENTION.md) | External flash-attention-turing d=64 kernel | Faster global attention, fails the box gate | Removed |
| [QKV_RESTORE_ROPE](QKV_RESTORE_ROPE.md) | Fuse QKV restore with RoPE | 409.43 → 401.04 ms, byte-identical | Opt-in |
| [MLP_SWEEP](MLP_SWEEP.md) | Boundary launch variants; FC2 restore + residual + next norm | Variants no gain; FC2/norm −0.52% | FC2/norm opt-in; variants removed |
| [INT8_GEMM_SEARCH](INT8_GEMM_SEARCH.md) | CUTLASS INT8 tiles, cached cuBLASLt algorithms | All tiles slower; no reliable gain | Removed |
| [INT8_GEMM_RESTORE](INT8_GEMM_RESTORE.md) | CUTLASS GEMM with restore epilogue | Chain 9.25% slower | Removed |
| [INT4_FC2](INT4_FC2.md) | W4A4 FC2 with SM75 INT4 tensor cores | −4.30% (all 32 FC2) but large output errors | Kept opt-in for research |
| [CONVROT_KITCHEN](CONVROT_KITCHEN.md) | ConvRot-style INT4 rotation; ComfyKitchen INT8 attention | Kitchen all-vision −7.58%; rotation adds cost | Kitchen opt-in; INT4 rotation kept for research |
| [KITCHEN_EXTENDED](KITCHEN_EXTENDED.md) | 12 extra cases, attention-only INT8, sequence layout | Attention-only 17/17 and −6.37%; MLP/QKV INT8 fails 3 cases | Sequence layout opt-in |
| [PIXEL_AND_MLP_SCOPE](PIXEL_AND_MLP_SCOPE.md) | Pixel-decoder conversion copy; selective MLP INT8 | `fused_nchw` −3.24%, exact; global-4 MLP 17/17 and −5.29% vs attention-only | Opt-in; became the research baseline |
| CUDA library passes | CUB/Thrust/NPP maintenance of the kernels | See [CUDA_LIBRARIES.md](../../../docs/native/CUDA_LIBRARIES.md) | Adopted where exact |
| Maintenance refactor | Split vision encoder and CMake modules, shared helpers | CTest 51/51 → 54/54, nine modes byte-identical before/after (`maintenance*-model.json`) | Done |
| INT8/INT4 research | Calibrated mixed precision, Rounds 1–8 | Best 32/33 on COCO development | Ongoing ([log](../../../docs/native/QUANT_RESEARCH_LOG.md)) |

Raw data sits next to the reports: `*.json` results and summaries, `*.log`
CTest/sanitizer/validation logs, `*-counters.txt` Nsight Compute excerpts.
Research rounds use the prefixes `r3-`…`r8-` / `round3-`…`round8-`,
`quant-research-*` and the calibration/screen names cited in the log. Full
outputs and telemetry stay under the untracked `.cache/`.

## Initial native benchmark (2026-09-23)

Native C++ versus the Python Turing FP16 patch at `19f8bf9` plus the Windows
fixes, before FP16 parameter residency was integrated:

| Implementation | Pooled median, 15 samples | Three process medians | Inference peak allocated |
|---|---:|---|---:|
| Native modular C++ FP16 | 570.87 ms | 565.31 / 576.47 / 573.61 ms | 3.615 GiB |
| Python patched eager FP16 | 565.82 ms | 567.28 / 563.99 / 565.09 ms | 1.997 GiB |

Latency was roughly equal (native 0.89% slower, not statistically established).
Native then kept FP32 weights (3,442,050,048 bytes) while the Python patch stores
selected weights in FP16 (1,728,215,040 bytes). With FP16 parameter residency a
preliminary native process measured 562.41 ms, 4.892 s model/text setup (was
11.4 s) and 2.214 GiB peak allocation, byte-identical outputs
([update-19f8bf9.json](update-19f8bf9.json)). After importing the upstream
fusions at `908e476`, 48/48 CTests passed and a five-sample check measured
500.35 ms with byte-identical outputs ([update-908e476.json](update-908e476.json)).

Conditions: Python torch 2.10.0+cu130 for the reference; four CPU threads, TF32
and cuDNN benchmark disabled, no `torch.compile`. Each fresh process ran one cold
inference, two warmups and five timed inferences in the order native, Python,
Python, native, native, Python. Timing synchronizes CUDA and covers upload,
preprocessing, vision, detection and full-resolution mask postprocessing. Native
preprocesses a decoded RGB tensor on the GPU; Python uses its Pillow path. Peak
allocated includes the cold inference and is allocator accounting, not physical
VRAM (Windows WDDM can page to shared memory). Native cold inference took
1.01–1.22 s after setup.

Output agreement with the Python patch (detections matched by maximum mask IoU;
implementation agreement, not accuracy):

| Case | Count, both | Differing mask pixels | Minimum matched IoU |
|---|---:|---:|---:|
| truck | 1 | 2 | 0.99999686 |
| paper bag | 4 | 0 | 1.0 |
| child | 6 | 5 | 0.99988392 |
| wheel | 4 | 0 | 1.0 |
| elephant (empty) | 0 | 0 | N/A |

Scores match exactly after conversion to float32; the maximum box difference is
0.020752 px. Raw samples, environment and memory statistics:
[summary.json](summary.json).

Reproduce with a CUDA build that includes `sam3_image_latency`, weights exported
with `native/tools/export_weights.py`, the image converted to RGB PPM and a UTF-8
prompt file without a trailing newline:

```powershell
.venv/Scripts/python.exe experiments/monitor_native_bench.py OUT build/native-windows-cu130/sam3_image_latency.exe STORE truck.ppm sam3/assets/bpe_simple_vocab_16e6.txt.gz prompt.txt 2 5 OUT
.venv/Scripts/python.exe experiments/monitor_native_bench.py OUT_PY .venv/Scripts/python.exe experiments/local_turing_bench.py --name patched_eager --checkpoint PATH_TO_SAM3_PT --output OUT_PY --reps 5
```

Use a new output directory for every fresh process.
