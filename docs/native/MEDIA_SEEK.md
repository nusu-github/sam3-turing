# Verified indexed video reads

Native media now reuses a byte-bounded RGB window and seeks using the complete
initial decode index. Every seek-produced native frame is checked by integer
presentation timestamp and SHA-256 of packed pixels/color properties against
that scan. A mismatch or failed seek restarts from the beginning and disables
further seeking for that source. Repeated/missing timestamps use sequential
replay with the same window. See MEDIA_IO.md for API and memory accounting.

This changes decoding work, not prompts, frame accessibility, detections,
resolution, model weights or image/video policy. C ABI 1 is extended with
`sam3_media_set_cache_bytes` and owning-result `sam3_media_stats`; existing
structures remain unchanged. The default window is 64 MiB. Both APIs and the
underlying FFmpeg interfaces are portable; no Windows/Turing execution or Actions
were performed.

## Evidence

`native/tests/media_seek.py` checks **3,856 individual reads across 16 cases**
against sequential FFmpeg RGB and ffprobe timestamps. It includes closed/open
H.264 GOPs and B-frames, VFR FFV1, duplicate timestamps, raw H.264 without
timestamps, inaccurate MP4 sync-sample metadata, MPEG-2 TS, random/reverse/forward
orders, zero/sub-frame cache budgets and four decoder threads. Each returned
frame is checked, including repeated indices. MPEG-2 TS triggers the rejected-seek
fallback and still matches all 360 frames. The inaccurate MP4 index is recovered
by FFmpeg without needing our fallback; it is not evidence of that branch.

For a 360-frame 128x96 H.264 fixture and a 589,824-byte (16-frame) window:

| Read order | Previous read decode count | Indexed read decode count |
| --- | ---: | ---: |
| Complete reverse | 64,980 | 544 |
| Complete forward | 360 | 360 |

Each additionally performs a 360-frame initial scan; the new scan also hashes
pixels. Previous reverse count follows the prior implementation's exact prefix
replay behavior. New counts are measured by the C API. VFR reverse needs 448
frames; duplicate/missing-time reverse needs 4,408 with seeking disabled.

Three separate process runs using installed C-only CPU SDK clients give median
reverse wall times **15.225 s before / 1.798 s after** (about 8.5x on this small
fixture). Forward medians are 1.608 / 1.470 s, with enough variance that no
forward speedup is claimed. Timings include process startup, full initial scan
and per-frame output files. They do not establish model inference, long-video,
high-resolution, Windows or Turing speed. Raw measurements: media-seek-benchmark.json.

CPU/CUDA CTests pass 17/29; all 23 existing codec parity cases pass. Real SAM3 and
SAM3.1 image/video file-owned predictors still match RGB callbacks in all 20
result comparisons. These are native composition regressions, not new upstream
preprocessing/quality claims. Complete per-read evidence and rejected-seek logs
are retained privately; public counters are in media-seek.json.

## Distribution and remaining costs

`media-seek-sdk-overlay` in the existing private bucket adds common headers/C
clients and CPU/CUDA native-library deltas to the prior `media-sdk-overlay`, which
it pins in its manifest. Apply the standalone base, matching media overlays, then
matching seek overlays. It contains no repeated LibTorch/FFmpeg libraries or
weights. Full reconstruction is checked against the installed SDK file hashes
and symlink targets, and the recovered clients are exercised.

Indexing still decodes and hashes the whole video before first use. The index is
linear in frame count. Ambiguous timestamps and rejected seeks retain prefix
replay costs outside the window; large GOPs can also reduce seek benefits. The
cache budget excludes the last frame, decoder/temporary memory, returned clones
and index. Wider codecs/preprocessing, video writing, multi-GPU, CPU long-run
stability and broader quality/performance remain open. The project goal remains
active.
