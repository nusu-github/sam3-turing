# Native media input and PNG output

The optional `SAM3_WITH_MEDIA=ON` component adds local image/video files and image
folders to the C/C++ runtime. It calls shared FFmpeg and libjpeg libraries directly;
it launches neither Python nor an `ffmpeg` executable. The model remains the same
shared modular store, with all queries/prompts and image/video policies intact.

## Platform and dependency basis

The attached `torchcodec/stable/index.md` identifies FFmpeg as its media backend
and links its installation instructions. The attached image-decoding example
also documents U8 CHW image tensors and separate image codecs. The linked
[upstream TorchCodec instructions](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)
include Windows CPU support; [FFmpeg's platform documentation](https://ffmpeg.org/platform.html)
describes native Windows shared libraries and MSVC linking. This implementation
uses those portable C library interfaces and C++17 filesystem/streams. It adopts
no Linux-only decoder API. Windows paths enter as UTF-8 and are opened through
`filesystem::u8path` plus custom FFmpeg AVIO callbacks.

Validation pins FFmpeg **6.1.1**, libjpeg-turbo **2.1.5**, and standalone LibTorch
**2.10.0**. The built FFmpeg reports `LGPL version 2.1 or later`; GPL/nonfree and
external-library autodetection are disabled. Configuration, source hash and URL
are in `media-ffmpeg-source.json`. The private artifact retains the source archive
and matching notices. This is a reproducible dependency build, not validation of
all FFmpeg versions, formats or platforms. Windows/Turing execution remains the
user's task; no Actions were used.

```sh
cmake -S native -B build/native -DSAM3_WITH_MEDIA=ON \
  -DSAM3_FFMPEG_ROOT=/absolute/path/to/ffmpeg \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
cmake --build build/native --parallel 4
```

Supply JPEG development headers/libraries through normal CMake search paths.
`SAM3_AV_INCLUDE` and `SAM3_{avformat,avcodec,avutil,swscale}_LIBRARY` can override
cached/system locations. Use the matching Windows shared FFmpeg distribution
with import libraries, ICU/zlib/libjpeg and the existing SDK build instructions.
`BundleRuntime.cmake` accepts `SAM3_MEDIA_ROOT`, stages that selected FFmpeg build
before dependency resolution and includes its dependency closure. Duplicate
library paths are accepted only when their SHA256 values agree; conflicting
versions still fail. Preserve source/configuration and upstream notices with the
runtime. GPU inference still uses LibTorch/CUDA; this media decoder executes on CPU.

## C and C++ ownership

`sam3_media_options_init/open/info/read_frame/release` expose a counted source.
Frame results contain owning `rgb` U8 `[3,H,W]`, `seconds` F64 and `duration` F64.
Info contains `frames`, `height`, `width`, nominal `fps` and the requested
`image_only` mode. Unknown timestamps are NaN; image/folder timing is zero rather
than an invented frame rate. Reads are serialized. A result survives source
release, and the C++ `MediaSource::read` returns mutable independent pixels.

`sam3_predictor_create_from_media(context, media, options, &predictor)` takes its
frame count/dimensions from the media source and retains it internally. The
caller may release both the media and context handles after creation. Existing
predictor prompt/edit/propagation/reset calls apply unchanged. The predictor's
`image_only` option remains explicit: a one-frame video must not silently acquire
the SAM3.1 image birth threshold.

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

`sam3_media_write_png(path, rgb_view)` writes RGB8 PNG. Model masks/results remain
available through the existing C result API. A video encoder/visualization writer
is not yet integrated. The C++ equivalents are `MediaSource` and `write_rgb_png`
in `sam3/media.h`. Builds with media disabled retain C symbols and report
`sam3_media_available()==0` / `SAM3_UNSUPPORTED` on open/write. Existing ABI 1
structures are unchanged.

## Decode semantics and current costs

JPEG uses libjpeg RGB/CMYK decoding, preserving the tested Pillow conversions.
Other images and videos use FFmpeg. Known image extensions match the original
folder loader; content probing handles video containers without relying on their
filename extension. Image mode takes the first image frame and does not apply
EXIF orientation, matching the original Pillow `convert("RGB")` path. Video
right-angle display rotation/reflection is applied. YUV conversion honors the
recorded colorspace/range and produces RGB8. Tested alpha conversion drops alpha.

Folders use numeric stems when all names parse as signed 64-bit integers,
otherwise UTF-8 lexicographic ordering. This currently differs from Python's
unbounded/unicode integer parsing for exceptional filenames. Empty inputs, bad
indices and decoder failures return errors. Input files must remain unchanged
while open. The predictor requires uniform frame dimensions; variable-resolution
video, unusual TIFF/16-bit/HDR/ICC policies and arbitrary-angle display matrices
still need broader treatment. These are recorded open cases, not claimed coverage.

Video open scans decoded timestamps and drains delayed B-frames to determine the
actual frame count; it does not trust only container estimates. Pixels are not
all retained in RAM. Sequential reads continue a decoder; backward/random reads
currently reopen and replay from the start, retaining one decoded RGB frame.
There is no frame/detection/object cap, but long reverse traversal can be quadratic.
Seek indexing and bounded caching remain performance work.

Decoded RGB enters the existing native Pillow-style preprocessing, whose prior
image-folder references remain applicable. This is **not** bit-exact validation
of the original default OpenCV video loader: that path uses cubic resizing and,
in the checked source, normalizes float 0–255 values without division by 255.
TorchCodec/OpenCV resize, color and normalization variants require a separate
policy audit before claiming full video-loader parity.

## Validation and distribution

- 23 codec cases match Pillow RGB or matching FFmpeg RGB24 exactly: ordinary,
  progressive, grayscale and CMYK JPEG; RGB/alpha/palette/grayscale PNG; BMP,
  TIFF, lossless WebP; Unicode/numeric folders; FFV1; H.264 B-frames; BT.709;
  all eight right-angle/reflection orientations; and variable frame rate.
- Frame counts, reordered/repeated/reverse reads, timestamps and PNG round trips
  are checked. Timestamp error against ffprobe's decimal serialization is below
  1e-6 seconds. C results survive source release.
- CPU/CUDA CTests pass 17/29. The media ownership/error/concurrency test also passes
  after the orientation fix. The media-disabled capability test passes.
- SAM3 and SAM3.1 file-owned C predictors match RGB-callback predictors in 20
  result comparisons across image/video, reset and retained-result checks. This
  is a composition/lifetime regression on real neural execution, not a new broad
  upstream quality claim.

Reports and scope are in `media-validation.json`, `media-parity.json` and
`media-neural.json`. C-only consumers of the relocated SDK exercise decoding and
actual SAM3.1 image inference with no Python/ffmpeg on PATH; loader traces audit
that model and codec libraries originate in the SDK.

The private `media-sdk-overlay` artifacts contain common media dependencies and
separate CPU/CUDA native updates for the previously retained standalone SDK.
Their manifests pin the base archives and hash each changed file. Apply the common
and chosen platform overlay into that base SDK directory. This avoids repeating
LibTorch libraries or weights; image/video use the same assembled SDK and modular
weight store. Full decode tensors, test inputs and build/recovery logs are private.

The full project goal remains open: wider codec/preprocess coverage, efficient
seeking, video encoding, multi-GPU transport, CPU long-run stability, broader
quality/performance and Windows/Turing physical validation remain.
