# ConvRot and ComfyKitchen on Windows / SM75

Follow-up: [extended quality and output-layout results](KITCHEN_EXTENDED.md).
The broader cases expose limitations of the earlier MLP/QKV INT8 path;
attention-only now has a separate 17-case FP16 comparison and timing series.

2026-09-24; base `ba018a0`, local experimental changes; RTX 2060 Max-Q 6 GB,
CUDA 13.0, LibTorch 2.10 cu130. Defaults remain unchanged.

## Sources and scope

[ConvRot](https://arxiv.org/abs/2512.03673) proposes rotating both activations
and weights before quantization. Its normalized regular Hadamard transform
preserves the unquantized product and avoids the DC concentration of a usual
Walsh-Hadamard transform. The paper evaluates diffusion transformers, not SAM3;
its speed and quality numbers are not predictions for this hardware/model.

We independently implemented the paper's regular H4 construction and Kronecker
H16/H64 groups in FC2 quantization. Weight rotation happens at load; activation
rotation is fused into restore/GELU/quantization, with no separate activation
buffer. This is an adaptation, not a full reproduction: paper-default H256
does not divide this model's 4736 hidden channels. Rotated values are rounded
to FP16 before INT4 quantization.

[ComfyKitchen PR #103](https://github.com/Comfy-Org/comfy-kitchen/pull/103)
provides a separate fully integer attention candidate: signed INT8 Q/K/V,
unsigned quantized P, FP32 online softmax and INT32 matrix accumulators.
Its SM75 path explicitly composes smaller MMA instructions and replaces
asynchronous copies with synchronous copies. Its `convrot` option uses ordinary
Hadamard variants, not the regular H4 transform above.

External source is pinned to PR head
`215cc5a8e52f163bf8d4631030f452767c737ab5` (Apache-2.0, with upstream notices).
It is held under `.cache/portability-review/convrot-kitchen`, not vendored.
The native adapter restricts the experiment to noncausal FP16 BHND inputs,
equal Q/K/V shape, head dimension 64 and SM75. Quantization, allocations and
launch overhead are included in its measured call.

## FC2 quality

Five fixed image/prompt cases, including one empty case, compared with original
FP16. Gates were fixed before testing: same count, finite outputs, minimum
matched mask IoU >= 0.98, maximum score error <= 0.02, maximum box-coordinate
error <= 1 pixel. These are regression checks, not a dataset accuracy estimate.

| FC2 variant | Passing cases / 5 | Worst IoU | Max score error | Max box error (px) |
|---|---:|---:|---:|---:|
| W4 / A8, diagnostic | 2 | 0.95735 | 0.03760 | 1.717 |
| W8 / symmetric A4, diagnostic | 1 | 0.41944 | 0.34033 | 34.833 |
| W4 / affine A4 | 1 | 0.91609 | 0.08447 | 2.220 |
| W4 / affine A4, row MSE clipping | 2 | 0.88994 | 0.10938 | 3.734 |
| W4 / A4, regular H16 | 3 | 0.96980 | 0.01807 | 1.536 |
| W4 / A4, regular H64 | 2 | 0.95287 | 0.01172 | 1.171 |

Activation quantization is the larger error source in the isolation tests.
Affine quantization improves it substantially. Optimizing row reconstruction
MSE does not reliably improve segmentation. H16 is promising, but still fails
child and wheel cases; H64 fails truck, child and wheel. None qualifies for
unconditional adoption. Diagnostic W4/A8 and W8/A4 unpack to INT8 for GEMM;
their runtime is not a native mixed-bit performance result.

All exact-mode outputs match the prior optimized INT8 reference byte for byte.
Detailed comparisons against both INT8 and FP16 are in
[int4-method-ba018a0.json](int4-method-ba018a0.json).

## Operator validation

ComfyKitchen builds with CUDA 13.0/MSVC and passes independent SDPA checks on
the non-default stream at N=1,64,128,129,576,5184 (B=9 for local N=576).
Both rotation settings satisfy the preset max-absolute-error 0.05 and RMSE
0.005 thresholds. Small cases also use explicit FP32 softmax/matmul references.
The long-attention kernel compiles to 246 registers/thread with no spills;
this high register use still warrants performance measurement, not inference
from INT8 theoretical throughput.

All four model variants pass all five fixed cases against both original FP16
and the preceding optimized INT8 path. No detection counts change (1/4/6/4/0).
The following errors are against original FP16:

| Attention replacement | Passing / 5 | Worst IoU | Max score error | Max box error (px) |
|---|---:|---:|---:|---:|
| Global only | 5 | 0.99073 | 0.01270 | 0.359 |
| Global only, rotated | 5 | 0.99161 | 0.00586 | 0.358 |
| Global and local | 5 | 0.99205 | 0.01074 | 0.402 |
| Global and local, rotated | 5 | 0.99015 | 0.01172 | 0.355 |

These checks establish a useful candidate, not dataset-wide accuracy or
bit-perfect equivalence. Detailed data are in
[kitchen-ba018a0.json](kitchen-ba018a0.json).

## Attention performance

Same truck workload, cached text, fresh image features, batch 1 and 1008 input
resolution. Each mode has two fresh processes, five warmups and 15 timed
samples per process; mode order is reversed for the second pass. CUDA stage
profiling is enabled for all modes. Baseline already includes INT8 MLP/QKV,
fused QKV restore/RoPE and fused FC2 restore/next normalization. INT4 is off.

| Attention | Wall median (ms) | Wall p95 (ms) | Vision mean (ms) | Detector mean (ms) | Wall reduction |
|---|---:|---:|---:|---:|---:|
| Existing SDPA | 415.75 | 434.28 | 292.22 | 120.47 | — |
| Global INT8 | 387.37 | 407.96 | 268.96 | 116.95 | 6.83% |
| Global INT8, rotated | 391.71 | 416.19 | 270.36 | 119.86 | 5.78% |
| Global + local INT8 | **384.23** | **396.48** | **263.20** | 117.20 | **7.58%** |
| Global + local INT8, rotated | 391.42 | 403.87 | 268.18 | 119.20 | 5.85% |

The selected unrotated all-vision variant has process medians 384.82/383.71 ms,
versus baseline 418.18/415.48 ms. Vision drops 29.02 ms; detector variation
also contributes to wall differences and should not be attributed to the
attention implementation. This is a local two-process benchmark, not a
claim of identical speedup for every image or GPU power state.

Independent operator runs use 100 warmup calls and 20 CUDA-event samples per
measurement, four alternating-order passes in each of two processes. Medians
across their eight pass medians (quantization included):

| Shape B,H,N,D | SDPA (ms) | INT8 (ms) | Rotated INT8 (ms) |
|---|---:|---:|---:|
| 9,16,576,64 | 1.3290 | 0.9616 | 1.1120 |
| 1,16,5184,64 | 10.3709 | 4.4819 | 4.5874 |

The long operator is 2.31x as fast (56.8% less elapsed time). Microbenchmarks
have different clock/launch behavior from full inference; do not multiply
these isolated savings to predict whole-model time. The first exploratory
micro run used only five warmups and is excluded from this table. Raw runs:
[run 2](kitchen-micro-2.json), [run 3](kitchen-micro-3.json).

Prefer the unrotated `kitchen_all` experiment for further evaluation; rotation
adds cost here and does not improve the five-case worst IoU. Keep INT4 rotation
as a separate mixed-precision research candidate. Broader image/prompt testing
is still needed before changing defaults.

## Additional kernel checks

INT4 checks cover full signed nibble inputs, production dimensions, tails,
non-default CUDA streams, affine offset correction and fused versus separate
quantization. MSE-selected reconstruction error is checked against unclipped
quantization. H16/H64 orthogonality is independently checked; integer-valued
FP16 inputs match explicit FP32 rotation plus quantization bit for bit. Fused
rotated boundary outputs also match the separate path bit for bit.

Compute Sanitizer memcheck reports zero errors for both the full attention
operator check and the expanded INT4 method check, including production shapes.
See [attention log](kitchen-memcheck.log) and
[INT4 log](int4-method-memcheck.log).
The complete existing CTest suite passes 51/51 with the new DLL (143.32 s):
[validation log](windows-validation-kitchen.log). This suite complements,
but does not replace, the opt-in operator and five-case model checks above.

## Reproduction

The existing optional INT8 development build also enables the INT4 experiments.
The model comparisons used `experiments/run_int4_method_native.py quality|timing`
and `summarize_int4_method_native.py`, removed after the experiment and kept in
commit `7d7fddb`. Cold quality-run latency is not a speed benchmark.

For attention, configure `SAM3_EXPERIMENT_KITCHEN_ATTENTION=ON` and set
`SAM3_KITCHEN_ROOT` to the pinned checkout's `comfy_kitchen/backends/cuda`.
Build `sam3_kitchen_bench` and `sam3_image_latency`.
`SAM3_EXPERIMENT_ATTENTION=kitchen` or `kitchen_rot` selects global attention;
`kitchen_all` or `kitchen_rot_all` also selects local vision attention.
The experiment is opt-in and requires the external source at build time.
