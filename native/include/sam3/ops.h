#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"

namespace sam3 {
// One independently padded, little-endian bit row per [H,W] boolean mask.
SAM3_NATIVE_EXPORT at::Tensor pack_masks(const at::Tensor& masks);
SAM3_NATIVE_EXPORT at::Tensor unpack_masks(const at::Tensor& packed, int64_t height, int64_t width);
SAM3_NATIVE_EXPORT at::Tensor resize_and_pack_masks(
    const at::Tensor& logits, int64_t height, int64_t width, int64_t chunk_size = 8);
// Stable decreasing score order; suppress iff IoU > threshold. No detection cap.
SAM3_NATIVE_EXPORT at::Tensor generic_nms(
    const at::Tensor& ious, const at::Tensor& scores, double threshold);
}
