#pragma once
#include <ATen/ATen.h>
#include "sam3_native_export.h"
namespace sam3 {
SAM3_NATIVE_EXPORT std::tuple<at::Tensor,at::Tensor> int4_quant(const at::Tensor& x,bool mse=false);
SAM3_NATIVE_EXPORT std::tuple<at::Tensor,at::Tensor> int4_boundary(const at::Tensor& acc,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias);
SAM3_NATIVE_EXPORT std::tuple<at::Tensor,at::Tensor,at::Tensor> int4_quant_affine(const at::Tensor& x,bool mse=false);
SAM3_NATIVE_EXPORT std::tuple<at::Tensor,at::Tensor,at::Tensor> int4_boundary_affine(const at::Tensor& acc,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias,bool mse=false);
SAM3_NATIVE_EXPORT void int4_correct(at::Tensor& acc,const at::Tensor& offset,const at::Tensor& sums);
SAM3_NATIVE_EXPORT std::tuple<at::Tensor,at::Tensor> int4_quant_rht(const at::Tensor& x,int group);
SAM3_NATIVE_EXPORT std::tuple<at::Tensor,at::Tensor> int4_boundary_rht(const at::Tensor& acc,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias,int group);
SAM3_NATIVE_EXPORT at::Tensor int4_mm(const at::Tensor& a,const at::Tensor& w,int tile=0);
}
