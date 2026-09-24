#pragma once
#include <ATen/ATen.h>
#include <tuple>

// Internal CUDA experiment interfaces, shared by the runtime and local probes.
// Keep the existing linkage: probes compile these kernels directly as well.
void approx_quant(const at::Tensor& input, at::Tensor& quantized, at::Tensor& scales);
void approx_restore(const at::Tensor& accum, const at::Tensor& input_scales,
    const at::Tensor& weight_scales, const at::Tensor& bias, at::Tensor& output, bool gelu);
void approx_restore_quant(const at::Tensor& accum, const at::Tensor& input_scales,
    const at::Tensor& weight_scales, const at::Tensor& bias,
    at::Tensor& quantized, at::Tensor& scales);
void approx_restore_quant_affine(const at::Tensor& accum, const at::Tensor& input_scales,
    const at::Tensor& weight_scales, const at::Tensor& bias,
    const at::Tensor& channel_scale, const at::Tensor& channel_shift,
    at::Tensor& quantized, at::Tensor& scales);
void approx_gelu_quant(const at::Tensor& fp16_input,
    const at::Tensor& channel_scale, const at::Tensor& channel_shift,
    at::Tensor& quantized, at::Tensor& scales);
at::Tensor approx_restore_rope(const at::Tensor& accum, const at::Tensor& input_scales,
    const at::Tensor& weight_scales, const at::Tensor& bias,
    const at::Tensor& frequencies, int64_t batch);

namespace sam3 {
at::Tensor int8_lt_cached(const at::Tensor& input, const at::Tensor& weight);
std::tuple<at::Tensor, at::Tensor> approx_fc2_norm(const at::Tensor& input,
    const at::Tensor& accum, const at::Tensor& input_scales, const at::Tensor& weight_scales,
    const at::Tensor& bias, const at::Tensor& gamma, const at::Tensor& beta, bool partition);
} // namespace sam3
