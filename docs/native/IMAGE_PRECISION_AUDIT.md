# Native image FP16 quality audit

The native image path now has a reproducible annotated-image audit. It retains
all 200 queries, runs full 1008-pixel encoding and accepts arbitrary UTF-8 prompt
files. The dataset selection and prompt list are test inputs, not runtime limits.
No production arithmetic, threshold or model weight was changed for this audit.

## Selection and metric

COCO val2017 images were sorted by ID, sampled with Python
`random.Random(20260923).sample(images, 8)`, then sorted again. Selection happened
before inference. All 38 annotations, crowd flags and image attribution/license
metadata were retained. The union of 21 annotated categories was queried on **all
8 images**, including category/image pairs with no matching annotation: 168 cases
per model/precision, 672 total. Literal category names are used without prompt
tuning. This is a small deterministic audit, not representative or full COCO AP.

Inference threshold is -1, retaining even zero-score queries. The offline
[COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
uses its standard maxDets `[1, 10, 100]`; this metric convention does not cap model
output. Segmentation predictions omit `bbox` when passed to `loadRes`, so their
area comes from the mask. AP includes IoU thresholds 0.50 through 0.95.
[pycocotools 2.0.11](https://pypi.org/project/pycocotools/2.0.11/), its evaluator
file hash and NumPy version are recorded. The preinstalled NVIDIA evaluator
produced exactly the same 96 aggregate metric values; its reports are retained
privately. Four evaluator tests cover asymmetric non-byte-aligned masks,
truncation, nonfinite/duplicate candidates and a known perfect AP result.

## Results

AP below is expressed on a 0–100 scale. The machine-readable report is
[image-precision-coco-slice.json](image-precision-coco-slice.json).

| Model | Precision | Box AP | Mask AP |
|---|---|---:|---:|
| SAM3 | BF16 | 77.0163 | 57.3187 |
| SAM3 | FP16 | 77.0142 | 57.1521 |
| SAM3.1 | BF16 | 77.0734 | 57.8476 |
| SAM3.1 | FP16 | 76.9289 | 57.3737 |

FP16 minus BF16 mask AP is -0.1666 points for SAM3 and -0.4739 for SAM3.1 on
this slice. All 134,400 emitted candidates pass finite checks on raw logits,
boxes, mask logits, presence logits and final scores/boxes. These are native
FP16-versus-native-BF16 comparisons with ground-truth scoring, not a new claim of
parity with an unmodified source predictor. The sample is too small to establish
general precision equivalence or quality for video, spatial prompts or tracking.

Local hardware is RTX PRO 4500 Blackwell, driver 580.159.04, official standalone
LibTorch 2.10.0+cu130; TF32 is disabled. Median detector-plus-postprocess time is
29.4–30.3 ms; median decode/preprocess/vision time is 56.7–60.6 ms. These are
single-run diagnostic timings with cached text and image features, two detector
warmups and four CPU threads, not end-to-end throughput or a Turing speed claim.
Image times include the cold first image. Validation, packing and output I/O are
outside the detector timing.

Peak live PyTorch CUDA allocation is 3,945,482,240 bytes (SAM3) and 3,966,251,008
bytes (SAM3.1), the same maxima for both precisions. Peaks include resident model,
cached text/image features, full results and finite-check temporaries. The first
vision peak is preserved across the detector warmup reset. Driver/context and
non-PyTorch allocations are excluded; this is not total VRAM required. Reserved
allocator memory is reported separately. No Turing fit/performance conclusion
is drawn from Blackwell allocation measurements.

## Reproduction

Prepare the optional offline evaluator separately from the inference runtime:

```bash
python -m venv eval-env
eval-env/bin/python -m pip install -r native/eval/requirements.txt
eval-env/bin/python native/eval/prepare_coco_slice.py /absolute/path/to/coco-slice
eval-env/bin/python -m unittest discover -s native/eval -p 'test_*.py'
```

The C++ executable is available as the development target
`sam3_image_benchmark`, or can be built against an installed SDK:

```bash
cmake -S native/eval -B build/image-audit \
  '-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch;/absolute/path/to/sdk' \
  -DSAM3_BENCHMARK_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/image-audit --config Release --parallel 4
cmake --install build/image-audit --config Release --prefix /absolute/path/to/sdk
```

The matching LibTorch prefix must precede the SDK prefix during C++ compilation.
CPU builds omit `SAM3_BENCHMARK_CUDA`; CUDA benchmarking requires it for correct
synchronization and allocator accounting. Use the corresponding Windows LibTorch
and SDK paths with MSVC; Windows execution remains user-owned. The inference
command itself needs neither Python nor the evaluation packages:

```bash
env -u LD_LIBRARY_PATH PATH=/nonexistent /absolute/path/to/sdk/bin/sam3_image_benchmark \
  /absolute/path/to/native-weights-v1 sam3 cuda fp16 \
  /absolute/path/to/coco-slice/manifest.tsv \
  /absolute/path/to/sdk/share/sam3-native/bpe_simple_vocab_16e6.txt.gz \
  /absolute/path/to/coco-slice/sam3-fp16
```

Repeat with `sam3.1` and/or `bf16_reference` in separate output directories.
BF16 is the local reference mode; it is not the Turing deployment mode. Then run
`evaluate_coco_slice.py DATASET --run OUTPUT_DIR [--run OUTPUT_DIR ...] --output REPORT.json`.
JSON detections retain original query IDs; packed masks use little-endian bits
over each flattened row-major H×W mask, with end padding per mask, not per row.

Both CPU/CUDA SDK consumers compile and load using their installed libraries.
Each model/precision was additionally executed from the installed CUDA SDK on
the first selected image/prompt with Python absent from PATH: all 200 candidates
and masks match the development run exactly. CPU model inference was not rerun.

## Persistence and limits

The existing private bucket stores `native-foundation/image-precision-linux/`
(all outputs, selected data/attribution, reports and logs) and
`image-audit-sdk-tools/` (small CPU/CUDA tool-only archives). Apply each tool to
the matching SDK already restored through `video-preprocess-sdk-overlay`.
No new LibTorch, OpenCV or weight copy is required. Public Git contains code and
aggregate evidence, not dataset images or weight-bearing outputs.

No GitHub Actions or Windows/Turing physical tests were used. Wider annotated
coverage, video precision, source parity across library builds and repeated
performance measurements remain necessary; the overall development goal stays
active.
