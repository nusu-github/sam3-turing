# CUB sweep of native CUDA code

2026-09-24. Reviewed all 16 repository CUDA translation units plus the shared
EDT/ROI helper headers. Target: Windows/MSVC, CUDA 13.0 / CCCL 3.0.1, SM75.
This pass extends CUB from block reductions to device-wide transformations.

## Inventory and decisions

| File(s) | Operation | Decision |
| --- | --- | --- |
| `src/ops_cuda.cu` | Mask packing/unpacking | Replace two grid-stride kernels with DeviceTransform over counting iterators; preserve bit order and tail padding. |
| `src/ops_cuda.cu` | Greedy NMS | Keep sequential dependency and synchronization; scan/select cannot directly express greedy suppression. |
| `src/components_cuda.cu` | Component size lookup | DeviceTransform; preserve 64-bit labels, background handling and histogram access. |
| `src/components_cuda.cu` | Parent initialization | Trial reverted after isolating performance; retain the original initialization kernel. |
| `src/components_cuda.cu` | Union-find, root labeling/counting | Keep atomic union and canonical root rules. Generic sort/histogram adds temporary storage and does not replace connectivity. |
| `src/vision_position_cuda.cu` | Axis table generation | Adopt DeviceTransform with unchanged arithmetic. |
| `src/vision_position_cuda.cu` | Full position expansion | CUB trial passed exactness but regressed performance; retain existing expansion kernel. |
| `src/vision_fusion_cuda.cu` | Welford and residual normalization | Keep current reduction order; CUB conversion would require separate numerical/model validation. |
| `src/rotary_cuda.cu`, `src/rotary_pair_cuda.cu` | Strided/pair RoPE | Keep fused pair writes, platform-specific FMA rounding and layout logic. DeviceTransform is not automatically shorter for multiple destinations. |
| `src/roi_align_cuda.cu`, `roi_align_impl.h` | Interpolation/sampling | No collective to replace. Mapping the outer loop alone has low expected maintenance benefit; not benchmarked. |
| `src/edt_cuda.cu`, `edt_impl.h` | Parabola-envelope EDT | Not a standard associative scan. NPP feasibility is documented separately. |
| `tools/approx_kernels.cu`, `tools/approx_boundary.cu` | INT8 extrema | CUB already adopted. Restore/RoPE remains specialized fused arithmetic. |
| `tools/int4_experiment.cu` | INT4 extrema/MSE/Hadamard | Symmetric extrema already CUB. Keep affine extrema after measured regression; preserve MSE addition order and rotation shuffles. |
| `tools/pixel_transform.cu` | Coalesced tiled transpose | Keep padded shared-memory layout; an elementwise transform does not preserve that access pattern. |
| `tools/kitchen_attention.cu` | External Attention adapter | Keep donor kernels/layout contracts intact; no local generic collective to substitute. |

## DeviceTransform implementation contract

The installed header provides a no-scratch-buffer overload accepting input
iterators, output, count, functor and explicit stream. The calls use the existing
device guard/current stream, retain int64 indexing and empty-output guards, and
check returned CUDA status. Counting iterators do not materialize index tensors.
Size lookup uses labels as direct input. The number of algorithm stages and intermediate tensors is
unchanged. No CMake dependency, runtime mode, or launch policy was added.

The change removes custom launch/grid bookkeeping rather than primarily reducing
physical line count. Small named functors keep per-element semantics visible.
Source: [CUB DeviceTransform](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceTransform.html),
cross-checked with CUDA 13.0's installed `cub/device/device_transform.cuh`.

On Windows, include CUB after c10 CUDA headers: placing it before CUDAGuard exposed
the Windows `small` macro to a LibTorch allocator parameter and failed compilation.
Reordering the includes fixed the build without patching dependency headers.

## Tests and measurements

`experiments/check_cub_ops.py LIBRARY OUTPUT.json [--check-only]` (removed after this
review; kept in commit `7d7fddb`) checked CPU parity,
byte-boundary tails, arbitrary packed input bytes, empty batches, multi-batch
inputs, strided components inputs, negative/multiple component values, and
nondefault CUDA streams. CUDA Graph capture/replay is verified with changed inputs.
SHA-256 over recorded outputs matches between isolated old/new libraries.

Timings use 30 batches of 20 graph replays after warmup; allocations are excluded.
They measure the public operator graph, not isolated CUB kernels. First
old/new/new/old process comparison (median of two process medians):

| Operator | Size/input | Old ms | CUB ms |
| --- | --- | ---: | ---: |
| Pack masks | 1008x1008 | 0.016617 | 0.014398 |
| Unpack masks | 1008x1008 | 0.049883 | 0.032376 |
| Components | 256x256, random | 0.089865 | 0.098195 |
| Components | 256x256, zero | 0.028334 | 0.024995 |
| Components | 256x256, one | 0.392638 | 0.388942 |

Random components was about 9% slower in this short series and regressed again in
a reversed-order confirmation. Parent initialization was restored, retaining only
the CUB size lookup. Union-find remains unchanged; its timings also vary across
processes, so these experiments do not prove which low-level cache/scheduling
effect caused the difference. No model-level speedup is claimed.

The full position-transform trial passed the existing 54 bit-exact cases,
including FP16/BF16/FP32/FP64, layouts, empty/strided inputs and invalid arguments.
However, at 288x288, FP16 increased from 0.631 to 0.901 ms and FP32 from 0.587 to
0.856 ms. The expansion conversion was reverted, with the rejected source saved
under `.cache/cub-wide/position-full-cub-trial.cu` for diagnosis. Position timing
initially used eager CUDA-event timing. The optional `--bench` mode of
`sam3_vision_position_stream_test` was then changed to graph replay to reduce
host launch variability, producing `POSITION_GRAPH_MS` records.

Raw results: `experiments/results/native_rtx2060/cub-wide-*.json` and
`.cache/cub-wide/position-*/process.log`. Earlier raw timings named `cub-ops-*`
used eager launches and should not be mixed with graph timings.

## Final accepted configuration

Four custom kernels are replaced across three files: pack, unpack, size lookup,
and axis table. Parent initialization, union/count, and full position expansion
retain their original algorithms. There are no duplicate production modes.

Final old/new/new/old comparison with isolated old/new native DLLs:

| Operator graph | Input | Old ms | Final ms |
| --- | --- | ---: | ---: |
| Pack | 1008x1008 | 0.015836 | 0.015382 |
| Unpack | 1008x1008 | 0.048899 | 0.032302 |
| Components | 256x256 random | 0.098698 | 0.074538 |
| Components | 256x256 zero | 0.020766 | 0.023121 |
| Components | 256x256 one | 0.391730 | 0.398242 |
| Position FP16 | 72x72 | 0.042865 | 0.042785 |
| Position FP16 | 144x144 | 0.128000 | 0.135825 |
| Position FP16 | 288x288 | 0.469702 | 0.468371 |
| Position FP32 | 72x72 | 0.067042 | 0.065784 |
| Position FP32 | 144x144 | 0.149850 | 0.150249 |
| Position FP32 | 288x288 | 0.487729 | 0.490452 |

Each number is the median of two process medians, each based on 30 batches of 20
replays. Results include both improvements and small regressions (zero components
about +2.4 microseconds, FP16 144x144 about +7.8 microseconds). Treat them as
operator-level observations, not a uniform performance guarantee. Keep the large
position expansion out of CUB: its regression was much larger and repeatable.

Final operator results: `cub-wide-isolate-*.json`; final position logs:
`.cache/cub-wide/graph-position-*/process.log`. All recorded operator output hashes
match `2cdf93d7663962e687777a78fc912a6bd2c6a9e5d34e22bc702714c52836fa63`.
Position tests passed all 54 exact cases in each process. Both benchmark harnesses
exercise nondefault-stream graph capture/replay, while the ordinary test entry
points remain unchanged unless `--bench` is explicitly supplied.

Compute Sanitizer memcheck: zero errors for the operator parity/graph harness and
for the position test filtered to `AxisValue`. Final summarized measurements are
in `experiments/results/native_rtx2060/cub-wide-final-summary.json`.
The full existing CTest suite passed 51/51 after these changes; see
`experiments/results/native_rtx2060/cub-wide-ctest.log`.
