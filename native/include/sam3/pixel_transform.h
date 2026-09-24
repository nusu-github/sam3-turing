#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
SAM3_NATIVE_EXPORT at::Tensor pixel_nchw_float(const at::Tensor& input);
}
