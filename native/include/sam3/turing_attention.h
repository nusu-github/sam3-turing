#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
// Local experiment: SM75 FP16 dense self-attention, head dimension 64.
// Accepts BHND views; packs only operands whose token-major view needs it.
SAM3_NATIVE_EXPORT at::Tensor turing_attention(const at::Tensor& q,
    const at::Tensor& k,const at::Tensor& v,bool reserve64=false);
SAM3_NATIVE_EXPORT std::vector<int64_t> turing_attention_resources(bool reserve64=false);
}
