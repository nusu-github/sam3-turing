#include "sam3/preprocess.h"
#include "sam3/autocast.h"
#include <ATen/Version.h>
#include <c10/core/InferenceMode.h>
namespace sam3 {
at::Tensor preprocess_rgb(const at::Tensor& pixels) {
  c10::InferenceMode inference;
  TORCH_CHECK((pixels.dim() == 3 || pixels.dim() == 4) && pixels.size(-3) == 3 &&
    pixels.size(-2) > 0 && pixels.size(-1) > 0, "RGB pixels must be [3,H,W] or [B,3,H,W]");
  TORCH_CHECK(pixels.device().is_cpu() || pixels.is_cuda(), "RGB preprocessing supports CPU or CUDA");
  AutocastGuard autocast(pixels.device().type(), false, at::kFloat);
  auto image = pixels.dim() == 3 ? pixels.unsqueeze(0) : pixels;
  if (image.scalar_type() == at::kByte) {
    // Already in the required intermediate representation.
  } else if (image.is_floating_point()) {
    image = image.mul(256.0 - 1e-3).to(at::kByte);
  } else {
    int bits = 0;
    switch (image.scalar_type()) {
      case at::kChar: bits = 7; break;
      case at::kShort: bits = 15; break;
      case at::kInt: bits = 31; break;
      case at::kLong: bits = 63; break;
      case at::kUInt16: bits = 16; break;
      default: TORCH_CHECK(false, "unsupported pixel dtype");
    }
    if (image.scalar_type() == at::kUInt16) image = (image / 256).to(at::kByte);
    else if (bits > 8) image = image.bitwise_right_shift(bits-8).to(at::kByte);
    else image = image.to(at::kByte).bitwise_left_shift_(8-bits);
  }
  if (!image.numel()) return at::empty({image.size(0),3,1008,1008}, image.options().dtype(at::kFloat));
  if (image.size(2) != 1008 || image.size(3) != 1008) {
    const auto capability = at::get_cpu_capability();
    const bool native_byte = image.device().is_cpu() && (capability == "AVX2" || capability == "AVX512");
    if (image.is_contiguous(at::MemoryFormat::ChannelsLast) && image.size(0) == 1 && image.numel() != image.stride(0)) {
      auto strides = image.strides().vec();
      strides[0] = image.numel();
      image = image.as_strided(image.sizes(),strides);
    }
    auto resized = at::_upsample_bilinear2d_aa(native_byte ? image : image.to(at::kFloat), {1008,1008}, false);
    image = native_byte ? resized : resized.round_().to(at::kByte);
  }
  // Preserve the source operation order; do not replace it with a fused scale.
  return image.to(at::kFloat).mul_(1.0/255).sub_(0.5).div_(0.5);
}
}
