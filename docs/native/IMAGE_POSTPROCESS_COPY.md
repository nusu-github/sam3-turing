# Direct image-result interpolation

Image postprocessing now writes later bilinear-resize chunks directly into the
final probability tensor. Previously each chunk allocated a resized tensor and
then copied it into the result. The first ordinary interpolation still determines
the correct output dtype; subsequent inputs explicitly convert to that dtype
before calling ATen's `upsample_bilinear2d_out`. The complete result still receives
one sigmoid operation, preserving the source CPU vector/tail behavior.

The attached `pytorch/2.14/amp.md`, “Op Eligibility”, explains that `out=` calls do
not pass through autocasting. Explicit conversion is therefore essential; merely
replacing the interpolation call would not preserve mixed-precision semantics.
Actual behavior was tested with the deployed official LibTorch 2.10.0 CPU/cu130
builds. This uses the existing ATen C++ API, without a new platform dependency or
kernel architecture. Windows/Turing physical tests remain with the user.

No confidence threshold, query count, output dimensions, prompt behavior or
probability output was reduced. No model or weight file changed. The chunk size
still controls scheduling rather than the number of returned detections.

## Correctness evidence

[image-postprocess-copy-validation.json](image-postprocess-copy-validation.json)
records the results and baseline/optimized library hashes.

- A one-shot source-formula oracle compares all boxes, scores, probability maps,
  binary masks and query indices. Each CPU/CUDA backend passes 2,304 cases:
  FP32/FP64/FP16/BF16 inputs, all three precision modes, 0/1/17/201 queries,
  noncontiguous input, ragged two-item batches, up/same/down resizing,
  empty/full/partial selections and chunks 1/3/13/512. Zero, near-zero, infinities
  and NaNs are included; finite values match exactly and NaN positions agree.
- Complete CTest suites pass: 18 tests with CPU LibTorch, 31 with CUDA LibTorch.
- All 672 cases from the prior fixed COCO slice were rerun with SAM3/SAM3.1 and
  FP16/BF16. The JSON candidates and packed masks are byte-identical to the
  previously scored outputs, including their aggregate SHA256 fingerprints.
  Thus the prior small-slice AP results remain unchanged; this is not a new
  claim about full COCO or broader quality.
- Both models and precisions additionally compare full float32 probability maps
  on image 190307, prompt `person`: 276,480,000 values across four comparisons,
  all byte-identical. The `--save-probabilities` audit option saves raw contiguous
  `[200,1,H,W]` values and records dtype/byte count in the JSON. It is optional
  diagnostic output, not a change to inference return semantics.
- Eight offline evaluator/comparator tests pass, including a single changed
  probability bit, missing cases and matching-but-truncated files.

## Local timing

The same test executable loads either the preserved baseline native library or
the optimized library, with identical installed SDK dependencies. Three process
pairs alternate their order. Each shape/chunk uses three warmups and nine timed
calls per process; the table is the median of the three process medians. Timing
uses wall clock with CUDA synchronization on both sides, four CPU threads,
FP16 low-resolution masks and all 200 full probability/binary outputs.

RTX PRO 4500 Blackwell, driver 580.159.04, LibTorch 2.10.0+cu130, chunk size 8:

| Output H×W | Before | After | Latency reduction |
|---|---:|---:|---:|
| 640×540 | 2.754 ms | 2.524 ms | 8.4% |
| 1080×1920 | 15.508 ms | 12.179 ms | 21.5% |
| 2160×3840 | 63.795 ms | 47.136 ms | 26.1% |

These are **postprocessing-only synthetic timings**, not model throughput or
Turing performance. Full-frame vision/text/detector execution is outside this
microbenchmark. Chunks 1/32/200 and per-process ranges are retained in the report;
the unchanged single-chunk path varies only about 0.1–0.4% between measurements.

Overall peak PyTorch allocation is unchanged. Although a resized temporary is
removed from each later interpolation, the peak occurs while retaining the full
probability result and binary masks. The measurement excludes driver/context and
non-PyTorch allocations; it does not establish a Turing memory-fit guarantee.

## Reproduction and persistence

The test is a normal native CMake target and is also built by the installed-SDK
consumer in `native/eval`, alongside `sam3_image_benchmark`:

```bash
sam3_image_results_test cpu
sam3_image_results_test cuda
sam3_image_results_test cuda --benchmark
```

Build instructions are in [IMAGE_PRECISION_AUDIT.md](IMAGE_PRECISION_AUDIT.md).
For CUDA SDK consumers, set `SAM3_BENCHMARK_CUDA=ON` for synchronized timing and
allocator accounting. Full-probability comparisons use:

```bash
sam3_image_benchmark STORE sam3 cuda fp16 MANIFEST.tsv BPE.gz OUTPUT --save-probabilities
python native/eval/compare_image_runs.py MANIFEST.tsv BEFORE AFTER --output comparison.json
```

Only the separate offline comparison uses Python. Installed CPU/CUDA test clients
pass with Python absent from PATH; installed and reconstructed CUDA SDK inference
also preserves the SAM3.1 FP16 full probability output exactly.

The private bucket's `image-postprocess-sdk-overlay/overlays.json` pins the
previous preprocessing SDK recipe and the new CPU/CUDA archives. Each archive
contains the updated native library, image audit executable and result test;
there are no additional weights, LibTorch, FFmpeg or OpenCV copies. The preceding
`image-audit-sdk-tools` layer is not required because these tools replace it.
Complete reconstructed SDK file hashes/symlinks are checked against staging.

`native-foundation/image-postprocess-linux/` preserves libraries, sources,
repeated timing logs, correctness comparisons and the new full probability
fixtures. Byte-identical COCO mask/candidate outputs reuse the prior
`native-foundation/image-precision-linux/` archives instead of uploading another
copy. No GitHub Actions were used; the overall development goal remains active.
