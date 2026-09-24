# Native image stage profile, 2026-09-23

Vision dominates this workload: about 77% of elapsed CUDA-stream time, followed
by the detector at 22%. Transfer, preprocessing and final mask postprocessing
together account for about 1%. This identifies where to investigate next; it
does not distinguish compute throughput, memory bandwidth and kernel launch
overhead within vision.

Same RTX 2060 Max-Q, SAM3 truck image/prompt, resolution 1008, threshold 0.5,
FP16 vision/text compute storage and cached text as the updated benchmark.
Base commit is 19f8bf9 with local Windows fixes and the latency profiling tool.
Two independent profiled processes, each with one cold inference, two warmups
and ten measured inferences; a ten-sample unprofiled process ran between them.
No concurrent inference or builds ran during measurement.

| Stage | Mean CUDA interval, 20 samples | Share of summed means |
|---|---:|---:|
| CPU-to-GPU upload | 1.40 ms | 0.25% |
| Preprocessing | 2.06 ms | 0.37% |
| Vision encoder, necks and selected position map | 430.03 ms | 77.05% |
| Grounding detector, including mask heads | 122.67 ms | 21.98% |
| Final mask/box postprocessing | 1.94 ms | 0.35% |

CUDA events are recorded on the current stream without added synchronization
between stages. Intervals can include host launch gaps and are not sums of
kernel execution times. Wall-clock medians were 554.35 / 560.04 ms for the
profiled processes and 561.55 ms for the unprofiled control. There is no obvious
profiling slowdown in these samples, but run order, clocks and noise prevent an
exact overhead estimate. Means are used for additive percentage attribution.

Host call durations are recorded separately and must not be interpreted as
isolated stage compute time: grounding validates CUDA image/text IDs with
`.item<bool>()`, which waits for earlier queued work. Vision's short host call
therefore does not mean vision is cheap; part of its wait appears in the next
host call. Postprocessing also contains data-dependent selection operations.

All three runs produced byte-identical masks, scores, boxes and query indices
to the pre-profile native truck result. The executable rebuilt successfully;
only the benchmark tool was changed, not the runtime operators. The previous
41/41 runtime test result applies to the same runtime build.

## Reproduce

Append `--profile` to the existing `sam3_image_latency` invocation described in
[README.md](README.md), using e.g. `2 10 OUTPUT --profile`. It emits `stages.json`
with every CUDA-event interval and host-call interval in milliseconds, alongside
the existing `metrics.json`. Omit the option for a control run. Keep each run's
output directory distinct.

[stage-profile.json](stage-profile.json) retains all samples, memory metrics,
aggregate statistics and output comparisons. Full binary outputs and telemetry
are local under `.cache/native-perf/{profile-1,control-1,profile-2}`.

The next useful breakdown is inside vision: the 32 transformer blocks
(attention and MLP), patch embedding and pyramid necks. A kernel-level trace
would then be needed to determine the hardware bottleneck. Eliminating only
transfer/preprocessing/final postprocessing cannot materially change the
approximately 560 ms latency on this workload.
