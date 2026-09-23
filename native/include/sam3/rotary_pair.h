#pragma once
#include "sam3/rotary.h"
namespace sam3 {
// Two independent RoPE outputs; inputs are unchanged. CUDA fuses standard
// interleaved QKV projections (16 heads, 64 channels); other layouts follow
// rotary_embedding twice. No cache or shape/prompt restriction is introduced.
SAM3_NATIVE_EXPORT std::tuple<at::Tensor, at::Tensor>
rotary_embedding_pair(const at::Tensor &q, const at::Tensor &k,
                      const at::Tensor &frequencies);
} // namespace sam3
