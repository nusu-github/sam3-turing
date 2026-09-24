# INT8 attention: broader checks and output layout

Next results: [pixel conversion fusion and selective MLP INT8](PIXEL_AND_MLP_SCOPE.md).

2026-09-24, Windows RTX 2060 Max-Q, CUDA 13.0 / LibTorch 2.10,
base `ba018a0` with the local experiments from [the previous report](CONVROT_KITCHEN.md).
No default precision/backend setting is changed.

## Extended quality checks

The 12 additional cases were fixed before running: people in video frames
0/50/99, shoes in frames 0/99, shirts in frame 50, car/bread in groceries,
person/blue vest in the court image, and window/tire in the truck image.
The video frames are correlated, not independent dataset samples. Each is
processed as a standalone image, so this does not test tracking.

The original gates are unchanged: finite outputs, equal count, minimum
Hungarian-matched mask IoU >= .98, max score difference <= .02 and maximum
box-coordinate error <= 1 pixel. These measure agreement with FP16, not
ground-truth accuracy. Three cases produce no detections even in FP16
(dance99-shoe, groceries-bread, court-person); nine cases are nonempty.

| Candidate / reference | All passing | Nonempty passing | Worst IoU | Max score error | Max box error (px) |
|---|---:|---:|---:|---:|---:|
| Existing INT8 MLP/QKV / FP16 | 9/12 | 6/9 | .98928 | .03271 | 1.334 |
| Add INT8 attention / FP16 | 9/12 | 6/9 | .98995 | .02441 | 2.024 |
| Add INT8 attention / existing INT8 | 12/12 | 9/9 | .99180 | .01709 | .718 |
| INT8 attention only, MLP/QKV FP16 / FP16 | **12/12** | **9/9** | **.99721** | **.01123** | **.412** |

Existing INT8 and combined INT8 fail the same cases: dance0-person,
dance50-shirt and truck-window. Window count is 6 in FP16 versus 5 in both
MLP/QKV INT8 configurations. Attention-only restores 6 and passes the gates.
Thus the prior five-case result was insufficient to endorse all the earlier
quantization choices. The added attention remains promising; the MLP/QKV
quantization needs broader evaluation or selective precision restoration.

Combined with the prior five cases, attention added to existing INT8 passes
17/17 comparisons with that INT8 reference. Against FP16, the combined path
passes 14/17, including four empty cases. A subsequent run of attention-only
on the original five cases also passes all gates: this configuration now
passes 17/17 against FP16, of which 13 are nonempty. The original five use
the sequence-major output optimization below; the new twelve used head-major
output. Independent exact operator/layout checks validate that transformation.
Original-case results are in [the updated comparison](kitchen-ba018a0.json).

Raw results: [extended comparisons](kitchen-extended-ba018a0.json).
The runs used `experiments/run_kitchen_extended.py` and
`summarize_kitchen_extended.py` (mode `attention_only` cleared the earlier
quantization variables). Those drivers were removed after the experiment and
remain in commit `7d7fddb`; the twelve cases and the gate comparison now live in
[`experiments/native_quality.py`](../../native_quality.py).

## Remaining kernel time

Two Nsight traces with the combined INT8 path attribute CUDA kernel duration
through launch correlation to application NVTX ranges. Both preserve output
bytes versus the uninstrumented run. They are diagnostic traces, not normal
latency benchmarks. Leading categories per inference are:

| Category | Approximate kernel time (ms) |
|---|---:|
| Vision MLP FC1 GEMM | 39.6 |
| Vision MLP FC2 GEMM | 33.9 |
| Vision QKV GEMM | 26.3 |
| Vision local attention, including quantization | 25.7 |
| Detector encoder attention | 24.3 |
| Vision fused MLP boundary | 24.1 |
| Vision attention output projection | 19.8 |
| Vision global attention, including quantization | 18.2 |
| Vision attention output layout copy | 4.83 |

Detector heads also contain about 31 ms of other kernels; this coarse bucket
needs finer attribution before choosing an optimization. MLP GEMMs remain
substantial even after the attention improvement.
Raw traces summarized in [kernel attribution](kitchen-hotspots-ba018a0.json).
Within vision attention, Q/K quantization consumes about 6.59 ms and V
quantization 5.22 ms across local/global calls. These are possible fusion
targets, not a promise that all 11.8 ms can be removed. Anchor detection is
only about .23 ms, so disabling stabilization is a low-priority tradeoff.
The detector-head coarse bucket contains 14.45 ms of direct-copy kernels;
identify their individual conversion/layout call sites before changing math.

## Output layout experiment

The external attention kernel accepts output strides. The adapter can write
directly into BNHD storage while exposing the same BHND tensor shape. Head
concatenation then uses a view rather than copying the entire output. Math,
quantization and kernel tile size stay unchanged.

Opt in with `SAM3_EXPERIMENT_KITCHEN_LAYOUT=sequence`; `head` retains the
previous behavior. Both normal and rotated attention pass exact output
comparison between layouts for N=1/64/128/129/576/5184, including batched,
non-default-stream and packed input cases. The benchmark additionally checks
that head concatenation shares the underlying storage. The five original
full-model image cases also preserve masks, scores, boxes and query indices
byte for byte. The `head` control matches the previous build's output bytes.

Drivers: `experiments/run_kitchen_layout.py quality|timing`,
`summarize_kitchen_layout.py` and `profile_kitchen_native.py --layout sequence`
(removed after the experiment; in commit `7d7fddb`).

### Measured layout result

Two fresh processes per layout, five warmups and 15 samples/process, reversed
order for the second pair, same combined INT8 configuration throughout:

| Output storage | Wall median (ms) | p95 (ms) | Vision mean (ms) | Detector mean (ms) |
|---|---:|---:|---:|---:|
| Head-major | 382.70 | 392.67 | 260.29 | 118.16 |
| Sequence-major | **377.67** | 392.87 | **256.91** | 117.59 |

Median wall time is 1.31% lower; p95 is effectively unchanged. Process
medians are 380.45/386.62 ms versus 377.75/377.46 ms. Vision improves 3.37 ms;
the full wall difference is not solely GPU kernel savings. Two new Nsight
traces contain no kernels under `vision.attention_layout`, confirming removal
of the 32 layout copies. They preserve the reference outputs byte for byte.
Do not use instrumented wall times to estimate normal inference gains.

Compute Sanitizer memcheck reports zero errors for the expanded operator
checks exercising both layouts, including production dimensions and tails:
[validation log](kitchen-layout-memcheck.log).

Raw [layout comparison](kitchen-layout-ba018a0.json) and
[sequence-layout profile](kitchen-sequence-hotspots-ba018a0.json).

### Attention-only quantization comparison

This separate paired run keeps MLP/QKV FP16 and changes only attention to
INT8 (with sequence-major output). It also uses two processes, five warmups,
15 samples/process and reversed order:

| Configuration | Wall median (ms) | p95 (ms) | Vision mean (ms) |
|---|---:|---:|---:|
| FP16 model | 516.67 | 526.86 | 390.34 |
| INT8 attention only | **483.73** | **496.95** | **359.19** |

This is a 6.37% wall reduction. It is slower than the combined INT8 path,
but clears the new cases' original FP16 agreement gates. These are separate
timing series: do not interpret 483.73 versus 377.67 ms as a controlled direct
comparison. Drivers: `experiments/run_kitchen_precision_timing.py` and
`summarize_kitchen_precision.py` (in commit `7d7fddb`).
[Raw paired timings](kitchen-precision-ba018a0.json).
