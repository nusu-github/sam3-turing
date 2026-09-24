#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"

namespace sam3 {
// Inference RoPE: floating [B,H,N,D] values, complex64 [N,D/2] frequencies.
// Arithmetic uses FP32 complex multiplication, then restores the input dtype.
// CUDA uses a precompiled kernel where supported; other layouts/devices retain
// the ATen implementation. Inputs are never modified.
SAM3_NATIVE_EXPORT at::Tensor rotary_embedding(
    const at::Tensor& value, const at::Tensor& frequencies);
}
