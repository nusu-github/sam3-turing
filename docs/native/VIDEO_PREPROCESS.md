# Video preprocessing policies

The original source has several distinct video-input transforms. Native C/C++
callers select one without changing weights, queries, prompts or image/video
birth thresholds; predictors default to the image-folder path. These are runtime
policies in one SDK, not model variants.

| Policy | Resize and numeric order | Resize device |
| --- | --- | --- |
| `IMAGE_FOLDER` (0) | Pillow byte bilinear; F32 /255; half storage, subtract .5 and divide .5 with half rounding | CPU |
| `PIL_LIST` (1) | Pillow byte bicubic; NumPy-like F64 /255; half storage and normalization | CPU |
| `TORCHCODEC_CPU` (2) | F32 bicubic, no antialias; half, /255, -.5, /.5 with half rounding | CPU |
| `TORCHCODEC_CUDA` (3) | Same operation sequence as TorchCodec's CUDA transform | Predictor CUDA device |
| `CV2_SOURCE` (4) | OpenCV byte cubic; F32 -.5, /.5; **no /255**, as in the checked source | CPU |

All produce the model's full 1008x1008 input, losslessly represented as F32 where
the source stores half. Float bicubic overshoots are retained; no extra clamp is
introduced. Cv2Source preserves `sam3/model/io_utils.py` literally, including its
byte-scale normalization: values can span -1 to 509. It is not silently corrected.
`TORCHCODEC_*` names identify transform semantics; the runtime uses ATen and does
not depend on the Python TorchCodec package or its GPU decoder.

## C/C++ usage

Call `sam3_predictor_set_preprocess(predictor, policy)` before the first frame
encoding or after explicit reset. Changing policy on an encoded session fails;
reset retains the selected policy and permits changing it. Prompt replacement
continues to use the selected policy. `sam3_predictor_info` includes
`preprocess_policy`. C++ exposes `VideoPreprocess`, `set_preprocess` and
`preprocess_policy` on `VideoPredictor`.

`sam3_preprocess_video_rgb(rgb_view, policy, device, &result)` exposes the same
transform separately, returning owning `image` F32 `[1,3,1008,1008]`. C++ has an
overload of `preprocess_video_rgb`. `sam3_video_preprocess_available` reports build
support; CUDA execution also needs a valid CUDA device. C ABI 1 layouts are
unchanged. OpenCV support disabled at build time reports `SAM3_UNSUPPORTED`.

OpenCV file compatibility also needs its color-conversion policy. The source's
OpenCV 4.10 FFmpeg wrapper converts to BGR24 using SWS_BICUBIC without explicitly
applying stream color metadata. Native default decoding instead honors that
metadata. These differ substantially for the BT.709 test input; preprocessing
alone cannot repair that difference.

```c
sam3_media_options input;
sam3_media_options_init(&input);
sam3_media *media = NULL;
sam3_media_open_with_color(path, &input, SAM3_MEDIA_COLOR_OPENCV, &media);
sam3_predictor *predictor = NULL;
sam3_predictor_create_from_media(context, media, &options, &predictor);
sam3_predictor_set_preprocess(predictor, SAM3_PREPROCESS_CV2_SOURCE);
sam3_media_release(media);
/* Check returned statuses. Existing prompt/edit/propagation APIs apply. */
```

Color policy is immutable per source. `SAM3_MEDIA_COLOR_STREAM` and the original
`sam3_media_open` retain the metadata-aware converter. `SAM3_MEDIA_COLOR_OPENCV`
emulates the OpenCV BGR conversion and then returns RGB. Both keep the existing
frame indexing, fingerprint verification and bounded reverse cache. Image/folder
inputs retain Pillow-style decoding. C++ offers a `MediaSource` constructor taking
`MediaColorPolicy`; no mutable shared-source setting can change a live predictor's
pixels. Codec/version/display-metadata differences beyond the tested inputs remain
possible. The reference color behavior is visible in OpenCV 4.10's
[FFmpeg implementation](https://github.com/opencv/opencv/blob/4.10.0/modules/videoio/src/cap_ffmpeg_impl.hpp).

## Validation and arithmetic boundaries

- Actual upstream loader functions on identical decoded RGB match all 32 CPU
  cases with the **same deployed LibTorch CPU binary** on both sides. Eight shapes
  include singleton dimensions, 1008 square, 1080x1920 and 2001x9. The CUDA
  transform matches all 8 shapes against the NVIDIA Python reference.
- Default Python wheels differ. The NVIDIA 2.10a0 build differs in the CPU
  bicubic transform on 6/8 shapes, with max normalized error 0.001953125. The
  official 2.10 Python CPU wheel also differs on 6/8 shapes at the same maximum,
  despite having the same source commit as standalone LibTorch (the wheel uses
  GCC 13.3, standalone LibTorch GCC 9.4). Matching version strings do not prove
  bit-identical CPU interpolation.
- Stage capture localizes the NVIDIA difference to F32 interpolation before half
  conversion. A C++ executable linked to the Python wheel's own libraries matches
  its Python output exactly. Running the original Python loader with standalone
  `libtorch_cpu`/`libc10` and the wheel's Python wrapper produces the32 exact cases.
  `/proc/self/maps` paths and library SHA-256 values are captured in the report.
  The wrapper is reference-only and is never distributed with the native SDK.
- Native file decoding plus `CV2_SOURCE` matches the unmodified source OpenCV
  loader on 6 files / 44 frames: lossless FFV1, real 720x1280 input, H.264
  B-frames, BT.709, 90-degree display rotation and a real BT.709 encode. With the
  default stream color policy, 9 BT.709 frames differ by up to 82 in normalized
  byte-scale values while 31 other frames match.
- The five policies pass 90 native file-owned versus RGB-callback neural result
  comparisons across SAM3/SAM3.1 (including 14 on a real BT.709 video with the
  OpenCV color path), covering full queries, reset, policy mutation rejection and
  lifetimes. These compare native composition, not whole-source model outputs.
  Wider accuracy effects of cross-build CPU interpolation are unquantified.

Evidence: [video-preprocess-matched.json](evidence/video-preprocess-matched.json),
[video-preprocess-nvidia.json](evidence/video-preprocess-nvidia.json),
[video-preprocess-wheel.json](evidence/video-preprocess-wheel.json),
[video-preprocess-files.json](evidence/video-preprocess-files.json),
[video-preprocess-cpu-build-audit.json](evidence/video-preprocess-cpu-build-audit.json),
[video-loader-validation.json](evidence/video-loader-validation.json).

## Build and distribution

Use `-DSAM3_WITH_OPENCV=ON -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4`.
Only OpenCV `core` and `imgproc` are needed. The pinned 4.10.0 build, source/archive
hashes and configuration are in [preprocess-opencv-source.json](evidence/preprocess-opencv-source.json).
It disables GUI, videoio, Python bindings and OpenCV CUDA; media decoding remains
the FFmpeg C interface. IPP 2021.11 is included in this reference build and its
notices are preserved. A different OpenCV build/backend may round differently.

The attached TorchVision transforms documentation distinguishes PIL/tensor
transforms, and its model documentation requires matching preprocessing. This
implementation uses the existing portable LibTorch interfaces plus OpenCV's
[resize API](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html).
OpenCV documents [Windows builds](https://docs.opencv.org/4.10.0/d3/d52/tutorial_windows_install.html);
no Linux-only inference or decoder interface is used.

`BundleRuntime.cmake` accepts `SAM3_OPENCV_ROOT`, stages the selected shared runtime
before resolving dependencies, hashes it and preserves supplied notices. C-only
SDK consumers need no Torch or OpenCV headers.
