#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
// Match Sam3Processor's RGB ToDtype(uint8)->antialiased Resize(1008)->
// ToDtype(float32,scale)->Normalize pipeline on the tensor's device.
// Input [3,H,W] or [B,3,H,W], output [B,3,1008,1008].
SAM3_NATIVE_EXPORT at::Tensor preprocess_rgb(const at::Tensor& pixels);
}
