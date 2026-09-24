# Native media input and PNG output

The optional `SAM3_WITH_MEDIA=ON` component adds local image files, video files
and image folders to the C/C++ runtime. It calls the shared FFmpeg and libjpeg C
libraries directly and launches neither Python nor an `ffmpeg` executable.
Decoding runs on the CPU; inference still uses LibTorch/CUDA. The FFmpeg and
UTF-8 filesystem interfaces used here exist on both Windows and Linux: Windows
paths enter as UTF-8 and are opened through `filesystem::u8path` and custom
FFmpeg AVIO callbacks.

## Build

Validation pins FFmpeg **6.1.1** (built LGPL 2.1-or-later, GPL/nonfree and
external-library autodetection disabled), libjpeg-turbo **2.1.5** and standalone
LibTorch **2.10.0**. Configuration, source hash and URL:
[media-ffmpeg-source.json](evidence/media-ffmpeg-source.json).

```sh
cmake -S native -B build/native -DSAM3_WITH_MEDIA=ON \
  -DSAM3_FFMPEG_ROOT=/absolute/path/to/ffmpeg \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
cmake --build build/native --parallel 4
```

JPEG headers/libraries come from the normal CMake search paths.
`SAM3_AV_INCLUDE` and `SAM3_{avformat,avcodec,avutil,swscale}_LIBRARY` override
cached/system locations. On Windows use a shared FFmpeg distribution with import
libraries. `BundleRuntime.cmake` stages the FFmpeg build given by
`SAM3_MEDIA_ROOT` with its dependency closure ([SDK.md](SDK.md)).

Builds without media keep the C symbols: `sam3_media_available()` returns 0 and
open/write report `SAM3_UNSUPPORTED`.

## C and C++ API

`sam3_media_options_init/open/info/read_frame/release` expose a counted source.
Frame results own `rgb` U8 `[3,H,W]`, `seconds` F64 and `duration` F64. Info
contains `frames`, `height`, `width`, nominal `fps` and the requested
`image_only` mode. Unknown timestamps are NaN; images and folders report zero
rather than an invented frame rate. Reads are serialized; a result outlives its
source, and C++ `MediaSource::read` returns independent mutable pixels.

`sam3_predictor_create_from_media(context, media, options, &predictor)` takes the
frame count and dimensions from the source and retains it, so the caller can
release the media and context handles afterwards. The predictor's `image_only`
option stays explicit: a one-frame video must not silently get the SAM3.1 image
birth threshold.

```c
sam3_media_options input;
sam3_media_options_init(&input);
input.image_only = 1; /* one image; use 0 for video/folder */
sam3_media *media = NULL;
sam3_media_open("photo.jpg", &input, &media);
sam3_predictor_options options;
sam3_predictor_options_init(&options, SAM3_MODEL_31);
options.image_only = 1;
sam3_predictor *predictor = NULL;
sam3_predictor_create_from_media(context, media, &options, &predictor);
sam3_media_release(media);
/* Check every returned status in application code. */
```

`sam3_media_write_png(path, rgb_view)` writes RGB8 PNG (C++ `write_rgb_png` in
`sam3/media.h`). There is no video encoder. For OpenCV-compatible color
conversion open the source with `sam3_media_open_with_color(...,
SAM3_MEDIA_COLOR_OPENCV, ...)`; see [VIDEO_PREPROCESS.md](VIDEO_PREPROCESS.md).

## Decoding semantics

- JPEG uses libjpeg RGB/CMYK decoding, matching the tested Pillow conversions.
  Other images and all videos use FFmpeg. Known image extensions match the
  original folder loader; video containers are probed by content.
- Image mode takes the first frame and does not apply EXIF orientation, like the
  original Pillow `convert("RGB")` path. Video right-angle display rotations and
  reflections are applied. YUV conversion honors the stream colorspace/range.
  Alpha is dropped.
- Folders use numeric order when every stem parses as a signed 64-bit integer,
  otherwise UTF-8 lexicographic order (this differs from Python's unbounded
  integer parsing for exceptional names).
- The predictor requires uniform frame dimensions. Variable-resolution video,
  16-bit/HDR/ICC policies and arbitrary-angle display matrices are not handled.
- Input files must not change while open.

### Indexing, seeking and the reverse-read window

Opening a video decodes it completely, draining delayed B-frames to get the
actual frame count, and records integer timestamps, keyframes and a SHA-256 of
each frame's packed pixels and color properties. This takes one full decode and
O(frames) metadata but does not keep the pixels.

Sequential reads continue the decoder. For backward or distant reads on streams
with strictly increasing timestamps, the decoder seeks to a preceding keyframe;
every frame produced after a seek is checked against the index before it reaches
the cache or the caller. A failed seek, missing frame or differing fingerprint
disables seeking for that source and replays from the start. Streams with
missing/duplicate/non-monotone timestamps never seek.

A byte-bounded RGB window keeps preceding frames for reverse traversal (default
64 MiB; `sam3_media_set_cache_bytes` / `MediaSource::set_cache_bytes`; 0
disables it). The budget excludes the last-frame cache, returned clones, the
index and decoder memory. `sam3_media_stats` / `stats()` report decoded-frame
counts, seek attempts and fallbacks, cache hits and bytes, and whether seeking is
still enabled. Streams with ambiguous timestamps still replay prefixes outside
the window, and very large GOPs limit the benefit of seeking.

For a 360-frame 128×96 H.264 fixture with a 16-frame window, a complete reverse
read decodes 544 frames instead of 64,980 (plus the 360-frame initial scan in
both cases); installed C clients took 1.798 s instead of 15.225 s (median of
three runs, including process start and output files). Forward reads are
unchanged.

## Validation

- 23 codec cases match Pillow RGB or FFmpeg RGB24 exactly: baseline, progressive,
  grayscale and CMYK JPEG; RGB/alpha/palette/grayscale PNG; BMP, TIFF, lossless
  WebP; Unicode/numeric folders; FFV1; H.264 B-frames; BT.709; all eight
  right-angle orientations; variable frame rate. Timestamps agree with ffprobe
  within 1e-6 s.
- `native/tests/media_seek.py` checks 3,856 reads in 16 cases against sequential
  FFmpeg decoding: open/closed GOPs, VFR, duplicate or missing timestamps, raw
  H.264, inaccurate MP4 indexes, an MPEG-2 TS stream that triggers the
  rejected-seek fallback, zero/sub-frame budgets and decoder threading.
- File-owned SAM3/SAM3.1 C predictors match RGB-callback predictors in all 20
  image/video result comparisons, including reset and retained results.

Evidence: [media-validation.json](evidence/media-validation.json),
[media-parity.json](evidence/media-parity.json),
[media-neural.json](evidence/media-neural.json),
[media-seek-validation.json](evidence/media-seek-validation.json),
[media-seek-benchmark.json](evidence/media-seek-benchmark.json).
