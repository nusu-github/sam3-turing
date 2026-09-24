#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
SAM3_NATIVE_EXPORT at::Tensor kitchen_attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,bool rotation,bool sequence_output=false);
}
