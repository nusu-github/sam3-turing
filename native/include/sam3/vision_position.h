#pragma once
#include "sam3_native_export.h"
#include <ATen/ATen.h>
namespace sam3 {
// Exact vision sine/cosine positions for a floating [B,C,H,W] reference.
// Values depend on shape/device/dtype, not image contents. Returns
// independently owned [B,256,H,W] with the reference implementation's
// channel-last strides. CUDA computes per-axis tables for Float/Half/BFloat16/
// Double and broadcasts them. CPU, empty inputs and other floating dtypes
// retain the ATen expression. No persistent cache or resolution restriction.
SAM3_NATIVE_EXPORT at::Tensor
vision_position_encoding(const at::Tensor &reference);
} // namespace sam3
