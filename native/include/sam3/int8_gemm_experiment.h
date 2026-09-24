#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
// Local SM75 experiment. A[M,K], W[N,K] -> INT32[M,N], no floating epilogue.
SAM3_NATIVE_EXPORT at::Tensor int8_gemm_experiment(const at::Tensor& a,const at::Tensor& w,int tile);
SAM3_NATIVE_EXPORT std::vector<int64_t> int8_gemm_resources(int tile);
SAM3_NATIVE_EXPORT at::Tensor int8_gemm_restore_experiment(const at::Tensor& a,const at::Tensor& w,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias);
}
