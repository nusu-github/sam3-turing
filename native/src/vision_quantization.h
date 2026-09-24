#pragma once
#include "approx_kernels.h"
#include "profile_range.h"
#include "vision_experiments.h"

namespace sam3::detail {
#ifdef SAM3_EXPERIMENT_INT8_GEMM
inline at::Tensor unpack_int4_diagnostic(const at::Tensor& p) {
  auto lo=p.to(at::kInt)&15,hi=at::bitwise_right_shift(p.to(at::kInt),4);
  lo=at::where(lo>=8,lo-16,lo);hi=at::where(hi>=8,hi-16,hi);
  return at::stack({lo,hi},-1).reshape({p.size(0),p.size(1)*2}).to(at::kChar);
}
#endif

#ifdef SAM3_WITH_CUDA
// The caller retains the flattened FP16 tensor until its restoration completes.
inline std::pair<at::Tensor, at::Tensor> quantize_rows(
    const at::Tensor& flat, const std::string& profile_label) {
  auto quantized = at::empty(flat.sizes(), flat.options().dtype(at::kChar));
  auto scales = at::empty({flat.size(0)}, flat.options().dtype(at::kFloat));
  profile_call(profile_label + ".quant", [&] {
    approx_quant(flat, quantized, scales);
    return 0;
  });
  return {std::move(quantized), std::move(scales)};
}

// Shared unfused INT8 projection; restore/GELU runs in FP32 before FP16 output.
inline at::Tensor quantized_linear(const at::Tensor& value, const at::Tensor& weight,
    const at::Tensor& weight_scales, const at::Tensor& bias, bool gelu,
    const std::string& profile_label) {
  const auto flat = value.to(at::kHalf).reshape({-1, value.size(-1)}).contiguous();
  auto [quantized, scales] = quantize_rows(flat, profile_label);
  auto accum = profile_call(profile_label + ".int8_gemm", [&] {
    return at::_int_mm(quantized, weight.t());
  });
  auto result = at::empty(accum.sizes(), flat.options());
  profile_call(profile_label + ".restore", [&] {
    approx_restore(accum, scales, weight_scales, bias, result, gelu);
    return 0;
  });
  return result;
}
#endif
} // namespace sam3::detail
