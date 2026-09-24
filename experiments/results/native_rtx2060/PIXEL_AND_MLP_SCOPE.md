# Pixel-decoder copies and selective MLP INT8

2026-09-24, RTX 2060 Max-Q / Windows / CUDA 13.0 / LibTorch 2.10.
Base `ba018a0` plus the existing opt-in experiments. Defaults remain unchanged.

## Finding the expensive copy

New NVTX ranges isolate source selection, resize, addition, convolution,
GroupNorm and ReLU in each pixel-decoder layer. Two diagnostic traces preserve
the previous outputs byte for byte. The second GroupNorm accounts for about
18 ms of GPU kernel time. In the first trace, its components are:

| Kernel family | Time per image (ms) |
|---|---:|
| Strided direct copy | 15.046 |
| Row moments | 1.987 |
| GroupNorm output | .721 |
| Unrolled copy | .583 |
| Scale/bias parameters | .002 |

This is primarily a layout/conversion cost. The convolution output is FP16
channels-last; autocast GroupNorm converts it to FP32 and requires contiguous
channel-major storage. A tiled shared-memory transpose can do the FP16-to-FP32
conversion and layout change in one pass, with coalesced reads and writes.
The GroupNorm computation and its FP32 reduction are left intact.

`SAM3_EXPERIMENT_PIXEL` selects these experiments:

- `exact`: previous behavior, plus inactive-by-default profiling ranges.
- `borrow`: avoid cloning the single-source pyramid feature; later addition
  is out of place, and multiple-source image selection is unchanged.
- `borrow_relu`: also reuse the fresh GroupNorm output for in-place ReLU.
- `nchw`: borrow source; make FP16 contiguous before conversion to FP32.
- `fused_nchw`: borrow source; use the tiled conversion/transpose kernel.

The CUDA adapter accepts nonempty channels-last FP16 tensors, guards device
and launch/index bounds, and runs on the caller's stream. Operator checks use
both production feature sizes (256 channels, 144x144 and 288x288) and batched
tails (2x40x33x35). Conversion and complete autocast GroupNorm outputs match
the original path exactly on the tested values.

### Performance and output agreement

All four changed modes preserve masks, scores, boxes and query IDs byte for
byte in the original five image cases; the newly built `exact` control also
matches the previous sequence-layout build. The following timing series uses
the existing all-MLP/QKV INT8 plus all-vision INT8 attention configuration,
sequence-major attention output, two fresh processes per mode, five warmups,
15 timed samples/process and reversed mode order for the second pass:

| Pixel mode | Wall median (ms) | p95 (ms) | Vision mean (ms) | Detector mean (ms) |
|---|---:|---:|---:|---:|
| exact | 377.74 | 392.36 | 256.38 | 117.31 |
| borrow_relu | 378.19 | 391.11 | 257.08 | 116.84 |
| nchw | 372.43 | 385.66 | 257.66 | 112.32 |
| fused_nchw | **365.51** | 389.37 | 257.65 | **106.23** |

`fused_nchw` cuts median wall latency 3.24% and the detector event interval
11.08 ms. It includes source borrowing as well as conversion fusion. No
consistent whole-model gain is established for borrowing/in-place ReLU alone.
The fused mode's process medians are 363.79/366.43 ms, versus exact
374.57/382.13 ms. Small unrelated stage changes and host gaps also affect the
wall result; do not attribute every millisecond to one kernel.

Isolated conversion + GroupNorm event medians across four alternating-order
passes in each of two processes (20 warmups, 20 samples per measurement):

| Feature size | Original (ms) | Reordered conversion (ms) | Fused conversion (ms) |
|---|---:|---:|---:|
| 256x144x144 | 1.3636 | 1.2616 | .9496 |
| 256x288x288 | 16.9668 | 11.0679 | 3.3831 |

The large operator is about 80.1% shorter. Isolated and model clocks differ;
this ratio must not be used to predict whole-model speedup. These are exact
conversion optimizations, not a relaxation of the quantization quality gates.
They do not fix pre-existing MLP INT8 disagreement with FP16.

Raw [model comparison](pixel-ba018a0.json),
[micro run 1](pixel-micro-1.json), [micro run 2](pixel-micro-2.json).

The two instrumented model traces confirm the expensive copy is gone. Mean
kernel time for the final normalization region drops from 16.90 ms to
.613 ms conversion + 2.707 ms unchanged GroupNorm. The earlier region drops
from 1.281 ms to .159 ms conversion + .686 ms GroupNorm. Traces preserve all
output bytes. These diagnostic times are not added to, or substituted for,
the controlled wall benchmark above.
[Fused trace summary](pixel-fused-hotspots-ba018a0.json).

## Quantization isolation

The three cases that failed the previous broad FP16 agreement checks were
used for diagnosis, not as an independent accuracy validation set:

| Case | MLP INT8 only | QKV INT8 only |
|---|---|---|
| dance0-person | Pass | Pass |
| dance50-shirt | Fail: count 2 -> 1 | Pass |
| truck-window | Fail: count 6 -> 5 | Pass |

The combined MLP/QKV path had also failed dance0-person, illustrating that
errors from separate approximations can accumulate even when each alone
passes a threshold. Gate values remain unchanged: same count, finite,
mask IoU >= .98, score error <= .02 and box-coordinate error <= 1 pixel.
Raw [isolation results](precision-isolation-ba018a0.json).

`SAM3_EXPERIMENT_MLP_SCOPE` limits the INT8 MLP blocks while keeping other
blocks on their original FP16 implementation. It accepts `all` (default),
`first8`, `first16`, `first24`, `last8`, `last16`, `last24`, `global` and `local`.
Global blocks have zero-based indices 7,15,23,31; local selects the other 28.
Only selected blocks receive quantized MLP weight copies. Existing QKV,
attention and FC2/next-normalization options still apply.

The first screen retains INT8 QKV and all-vision INT8 attention, with the
sequence-major output optimization. Results on the three known failures:

| INT8 MLP scope | Passing / 3 |
|---|---:|
| First 24 blocks | 0 |
| Last 24 blocks | 1 |
| Local blocks (28) | 1 |
| Global blocks (4) | 3 |

This selects a candidate for broader checks; it is not evidence that a
four-block configuration generalizes to an independent dataset.
Raw [scope results](mlp-scope-ba018a0.json).

### Broader selected-scope result

The four-global-block candidate passes all 12 extended cases and all five
original cases against FP16: 17/17, including four empty cases (13 nonempty).
Detection counts are preserved, including the problematic shirt/window cases.
Worst matched mask IoU is .99478, maximum score error .0078125 and maximum
box-coordinate error .87635 px. This candidate includes INT8 QKV and INT8
attention; it is not merely a standalone MLP test. The same cases were used
for development, so this does not establish unseen-dataset accuracy. Fewer
quantized blocks are a confound: these results do not prove global blocks
are intrinsically less sensitive than every alternative four-block subset.

A separate paired timing series compares this candidate with attention-only
INT8. Both use sequence-major attention output and the **original** pixel
path, two processes/mode, five warmups and 15 samples/process, reversed order:

| Configuration | Wall median (ms) | p95 (ms) | Vision mean (ms) | Detector mean (ms) |
|---|---:|---:|---:|---:|
| Attention INT8; QKV/MLP FP16 | 478.63 | 497.15 | 357.76 | 118.05 |
| Attention/QKV INT8; only four global MLPs INT8 | **453.31** | **475.73** | **328.33** | 123.02 |

Median wall latency falls 5.29%, despite a noisier detector interval. This
comparison includes both QKV and selective MLP quantization; it does not
attribute the entire gain to four MLPs. The process medians are 450.23/455.19
ms for the candidate and 481.09/476.63 ms for attention-only. The pixel-fusion
and selective-precision gains are measured in different configurations and
must not be added as a measured combined gain.

Original-case comparisons and timing samples are in
[the updated attention results](kitchen-ba018a0.json), mode `global_mlp`.
They were produced with `experiments/run_kitchen_native.py quality global_mlp`
and `timing global_mlp attention_only` (driver in commit `7d7fddb`).

## Kernel safety checks

Compute Sanitizer memcheck reports zero errors for conversion plus GroupNorm
checks on a non-default stream, including production dimensions and batched
tails. Racecheck, filtered to `pixel_transpose`, reports zero hazards, errors
or warnings. Logs: [memcheck](pixel-memcheck.log), [racecheck](pixel-racecheck.log).
The full existing CTest suite also passes 51/51 with the updated library
(141.51 seconds): [validation log](windows-validation-pixel.log).

## Reproduce

Build `sam3_pixel_bench` and `sam3_image_latency`. The model runs used
`experiments/run_pixel_native.py quality|timing` with explicit modes
(`exact borrow borrow_relu nchw fused_nchw`) and `summarize_pixel_native.py`.
Timing ran in forward/reverse order, five warmups and 15 samples per process;
cold quality-run times are not used as speed claims.

`experiments/run_precision_isolation.py` produced the diagnostic split, and
`experiments/run_mlp_scope.py first24 last24 global local` / `global --all-cases`
screened scopes and extended the selected candidate to the fixed 12 cases.
All of these drivers were removed after the experiment and remain in commit
`7d7fddb`.
