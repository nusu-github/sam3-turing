# Controlled upstream fusion OFF/ON comparison

Measured 2026-09-24 on Windows / RTX 2060 Max-Q, using commit `908e476` for
both variants. Turning on vision normalization/residual/layout fusion and
paired Q/K rotary reduced the overall pooled median by **5.76%** and the mean
vision CUDA interval by **7.58%** on this image workload.

| Measurement | Fusion OFF | Fusion ON |
|---|---:|---:|
| Overall pooled wall median, 60 samples each | 573.98 ms | 540.91 ms |
| Median of four process medians | 573.96 ms | 539.59 ms |
| Vision mean CUDA interval | 443.72 ms | 410.09 ms |
| Detector mean CUDA interval | 125.46 ms | 127.29 ms |
| Inference peak allocated | 2,376,667,136 bytes | 2,377,715,712 bytes |

This is about 33 ms saved overall. The median-of-process-medians estimate is
5.99% shorter. The allocation peak increases by 1 MiB. Detector time did not
improve; the measured reduction is concentrated in vision.

## Paired processes

Each row pairs adjacent fresh processes. The sequence balances execution order:
OFF/ON, ON/OFF, OFF/ON, ON/OFF. Entries are each process's wall-time median.

| Pair | OFF | ON |
|---|---:|---:|
| 1 | 566.88 ms | 538.53 ms |
| 2 | 573.99 ms | 538.52 ms |
| 3 | 573.92 ms | 540.65 ms |
| 4 | 579.65 ms | 549.40 ms |

ON is faster in all four pairs, by approximately 5.0–6.2%. Clocks were not
locked: last-five-second median SM clocks ranged from 1305 to 1350 MHz; sampled
maximum temperatures rose from 61 to 71 C over the experiment. Alternating order
reduces this drift's impact but does not eliminate all measurement noise. These
are local observations, not a guarantee across datasets or hardware.

## What was controlled

- Same source, compiler, CUDA 13.0, standalone LibTorch 2.10.0+cu130, weights,
  GPU and input. Native executable SHA-256 is identical in the two directories;
  only the runtime DLL variant differs.
- OFF sets both `SAM3_FUSE_VISION_NORM` and `SAM3_FUSE_VISION_QK` to OFF; ON sets
  both to ON. Both variants retain the latest in-place exact GELU and semantic
  text reuse changes. This experiment **does not measure GELU's separate gain**
  or all improvements relative to an older commit.
- Fixed FP16 vision/text compute-weight storage, cached text, batch one,
  `truck.jpg` / `truck`, resolution 1008, threshold 0.5 and all 200 queries.
- Four CPU threads, TF32 disabled, cuDNN benchmarking disabled, default BLAS
  preference (`TORCH_BLAS_PREFER_CUBLASLT=0`), workspace config `:4096:8`.
- One cold inference, five warmups and fifteen timed samples per fresh process.
  Includes RGB transfer, preprocessing, vision, detector and postprocessing;
  excludes model loading and disk reads. Image features are recomputed.
- Same stage instrumentation (`--profile`), current-stream CUDA events and
  final device synchronization. NVML sampling runs in an external Python
  monitor; the native executable uses only system directories on PATH with
  its dependencies beside it.
- Both binaries were built and snapshotted before timing. No task builds or
  concurrent task inference runs overlapped measurement. The working build was
  restored to both fusion options ON before the measurements began.

Every measured process's masks, boxes, scores and query indices match OFF-1
byte for byte. This checks this fixture's outputs, not dataset accuracy. The
upstream runtime's preceding Windows validation passed 48/48 tests; no runtime
source edits were made for this comparison.

## Reproduction and artifacts

Configure the same build first with both fusion options ON, then with both OFF,
building `sam3_image_latency`. Copy each runtime DLL and executable into separate
variant directories with the same dependency DLLs. Restore the working build to
ON. Do not overwrite a DLL while a process using it is running.

Run the existing image-latency command with `5 15 OUTPUT --profile`, using fresh
processes in the sequence above. The local driver is
`.cache/native-perf/fusion-ab/run.py`; immutable runtime snapshots, SHA-256
manifests, raw outputs and telemetry are beside it in `on`, `off` and the eight
measurement directories. Dependency DLLs are hard-linked to save disk space;
the two `sam3_native.dll` snapshots are independent copies.

[fusion-ab-908e476.json](fusion-ab-908e476.json) retains all timing samples,
variant hashes, per-stage intervals, configuration, output checks and telemetry
summaries. Do not compare the absolute times here directly with the previous
500 ms smoke run or 515 ms prototype experiment: their contemporaneous controls
were different. This OFF/ON experiment is the supported fusion comparison.
