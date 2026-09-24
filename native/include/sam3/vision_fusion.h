#pragma once
#include "sam3_native_export.h"
#include <ATen/ATen.h>
namespace sam3 {
// LayerNorm(1024, eps=1e-5), optional 24x24 window partition, then dtype cast.
// Input is FP32 [B,H,W,1024], affine parameters are FP32 [1024]. Windowed
// dimensions must be multiples of 24. CUDA fuses the contiguous/aligned case;
// all other supported layouts/devices follow the equivalent ATen expression.
SAM3_NATIVE_EXPORT at::Tensor vision_norm_projection(const at::Tensor &input,
                                                     const at::Tensor &gamma,
                                                     const at::Tensor &beta,
                                                     at::ScalarType output_type,
                                                     bool windowed);
// Inverse window partition (when requested) + FP32 residual addition +
// LayerNorm + projection cast. Returns {residual_sum, normalized_projection}.
SAM3_NATIVE_EXPORT std::tuple<at::Tensor, at::Tensor>
vision_residual_norm(const at::Tensor &residual, const at::Tensor &attention,
                     const at::Tensor &gamma, const at::Tensor &beta,
                     at::ScalarType output_type, bool windowed);
// Residual inputs share logical [B,H,W,1024] layout. The normalized output
// may be partitioned for the next block's QKV projection; the sum is logical.
SAM3_NATIVE_EXPORT std::tuple<at::Tensor, at::Tensor>
vision_residual_norm_projection(const at::Tensor &residual,
                                const at::Tensor &projection,
                                const at::Tensor &gamma, const at::Tensor &beta,
                                at::ScalarType output_type,
                                bool partition_output);

} // namespace sam3
