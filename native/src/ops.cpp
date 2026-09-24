#include "sam3/ops.h"
#include <ATen/Parallel.h>
#include <limits>
#include <vector>

namespace sam3 {
#ifdef SAM3_WITH_CUDA
at::Tensor pack_masks_cuda(const at::Tensor& masks);
at::Tensor unpack_masks_cuda(const at::Tensor& packed, int64_t height, int64_t width);
at::Tensor nms_keep_cuda(const at::Tensor& suppression);
#endif
namespace {
int64_t pixels_checked(int64_t height, int64_t width) {
  TORCH_CHECK(height > 0 && width > 0 && height <= std::numeric_limits<int64_t>::max() / width,
              "height and width must be positive with a representable product");
  return height * width;
}
void device_checked(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cpu() || tensor.is_cuda(), "only CPU and CUDA tensors are supported");
#ifndef SAM3_WITH_CUDA
  TORCH_CHECK(!tensor.is_cuda(), "this native library was built without CUDA kernels");
#endif
}
}

at::Tensor pack_masks(const at::Tensor& masks) {
  TORCH_CHECK(masks.dim() == 3 && masks.scalar_type() == at::kBool,
              "masks must be bool [N,H,W]");
  device_checked(masks);
  const auto pixels = pixels_checked(masks.size(1), masks.size(2));
  const auto contiguous = masks.contiguous();
#ifdef SAM3_WITH_CUDA
  if (masks.is_cuda()) return pack_masks_cuda(contiguous);
#endif
  const auto bytes = pixels / 8 + (pixels % 8 != 0);
  auto output = at::empty({masks.size(0), bytes}, masks.options().dtype(at::kByte));
  const auto* src = contiguous.const_data_ptr<bool>();
  auto* dst = output.mutable_data_ptr<uint8_t>();
  at::parallel_for(0, output.numel(), 1024, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; ++i) {
      const auto row = i / bytes;
      const auto offset = (i % bytes) * 8;
      uint8_t value = 0;
      for (int bit = 0; bit < 8 && offset + bit < pixels; ++bit)
        value |= static_cast<uint8_t>(src[row * pixels + offset + bit]) << bit;
      dst[i] = value;
    }
  });
  return output;
}

at::Tensor unpack_masks(const at::Tensor& packed, int64_t height, int64_t width) {
  const auto pixels = pixels_checked(height, width);
  const auto bytes = pixels / 8 + (pixels % 8 != 0);
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte && packed.size(1) == bytes,
              "packed must be uint8 [N,ceil(H*W/8)]");
  device_checked(packed);
  const auto contiguous = packed.contiguous();
#ifdef SAM3_WITH_CUDA
  if (packed.is_cuda()) return unpack_masks_cuda(contiguous, height, width);
#endif
  auto output = at::empty({packed.size(0), 1, height, width}, packed.options().dtype(at::kBool));
  const auto* src = contiguous.const_data_ptr<uint8_t>();
  auto* dst = output.mutable_data_ptr<bool>();
  at::parallel_for(0, output.numel(), 4096, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; ++i) {
      const auto p = i % pixels;
      dst[i] = (src[(i / pixels) * bytes + p / 8] >> (p % 8)) & 1;
    }
  });
  return output;
}

at::Tensor resize_and_pack_masks(const at::Tensor& logits, int64_t height, int64_t width, int64_t chunk_size) {
  TORCH_CHECK(logits.dim() == 3 && logits.is_floating_point(), "logits must be floating [N,H,W]");
  device_checked(logits);
  pixels_checked(logits.size(1), logits.size(2));
  const auto pixels = pixels_checked(height, width);
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  auto result = at::empty({logits.size(0), pixels / 8 + (pixels % 8 != 0)}, logits.options().dtype(at::kByte));
  for (int64_t start = 0; start < logits.size(0);) {
    const auto end = start + std::min(chunk_size, logits.size(0) - start);
    const auto resized = at::upsample_bilinear2d(logits.slice(0, start, end).unsqueeze(1), {height, width}, false);
    // Preserve interpolation dtype and sigmoid rounding, including FP16 near 0.5.
    result.slice(0, start, end).copy_(pack_masks(resized.sigmoid().gt(0.5).squeeze(1)));
    start = end;
  }
  return result;
}

at::Tensor generic_nms(const at::Tensor& ious, const at::Tensor& scores, double threshold) {
  TORCH_CHECK(ious.dim() == 2 && scores.dim() == 1 && ious.size(0) == ious.size(1) &&
              ious.size(0) == scores.size(0), "expected IoUs [N,N] and scores [N]");
  TORCH_CHECK(ious.device() == scores.device(), "IoUs and scores must share a device");
  TORCH_CHECK(ious.is_floating_point() && scores.is_floating_point(), "IoUs and scores must be floating point");
  device_checked(ious);
  const auto n = scores.numel();
  const auto order = at::argsort(scores, true, 0, true);
  if (n == 0) return order;
  const auto suppression = ious.gt(threshold).index_select(0, order).index_select(1, order).contiguous();
  at::Tensor keep;
#ifdef SAM3_WITH_CUDA
  if (ious.is_cuda()) {
    keep = nms_keep_cuda(suppression);
  } else
#endif
  {
    keep = at::ones({n}, scores.options().dtype(at::kBool));
    auto* kept = keep.mutable_data_ptr<bool>();
    const auto* suppress = suppression.const_data_ptr<bool>();
    for (int64_t i = 0; i < n; ++i) {
      if (!kept[i]) continue;
      for (int64_t j = i + 1; j < n; ++j)
        if (suppress[i * n + j]) kept[j] = false;
    }
  }
  return order.masked_select(keep);
}
}
