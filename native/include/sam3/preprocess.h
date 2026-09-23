#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
// Match Sam3Processor's RGB ToDtype(uint8)->antialiased Resize(1008)->
// ToDtype(float32,scale)->Normalize pipeline on the tensor's device.
// Input [3,H,W] or [B,3,H,W], output [B,3,1008,1008].
SAM3_NATIVE_EXPORT at::Tensor preprocess_rgb(const at::Tensor& pixels);
// Original tracker JPEG-sequence path: Pillow-compatible byte bicubic on CPU,
// byte / 255 rounded to F32, then Normalize(.5,.5); contiguous NCHW output.
// Input is decoded RGB uint8 [3,H,W]. This is distinct from image-mode resize.
SAM3_NATIVE_EXPORT at::Tensor resize_tracking_rgb(const at::Tensor&,int64_t height,int64_t width);
// High-level video predictor image-folder path: Pillow byte bilinear, F32 /255,
// F16 storage and each normalization step, then lossless F32 for neural input.
SAM3_NATIVE_EXPORT at::Tensor preprocess_video_rgb(const at::Tensor&);
// Upstream video input policies are deliberately distinct. Cv2Source preserves
// the checked source's byte-scale F32 normalization (no division by 255).
enum class VideoPreprocess : int32_t { ImageFolder=0, PilList=1, TorchCodecCpu=2, TorchCodecCuda=3, Cv2Source=4 };
SAM3_NATIVE_EXPORT bool video_preprocess_available(VideoPreprocess) noexcept;
SAM3_NATIVE_EXPORT at::Tensor preprocess_video_rgb(const at::Tensor&,VideoPreprocess,at::Device device=at::kCPU);
SAM3_NATIVE_EXPORT at::Tensor preprocess_tracking_rgb(const at::Tensor&);
}
