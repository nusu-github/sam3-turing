#include "sam3/vision_fusion.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
#ifdef SAM3_WITH_CUDA
void vision_residual_norm_cuda(const at::Tensor &, const at::Tensor &,
                               const at::Tensor &, const at::Tensor &,
                               at::Tensor &, at::Tensor &, bool, bool);
void vision_norm_projection_cuda(const at::Tensor &, const at::Tensor &,
                                 const at::Tensor &, at::Tensor &, bool);
#endif
at::Tensor vision_norm_projection(const at::Tensor &input,
                                  const at::Tensor &gamma,
                                  const at::Tensor &beta, at::ScalarType type,
                                  bool windowed) {
  c10::InferenceMode inference;
  TORCH_CHECK(input.dim() == 4 && input.size(3) == 1024 &&
                  input.scalar_type() == at::kFloat,
              "vision norm requires FP32 [B,H,W,1024]");
  TORCH_CHECK(
      gamma.sizes() == at::IntArrayRef({1024}) &&
          beta.sizes() == gamma.sizes() && gamma.scalar_type() == at::kFloat &&
          beta.scalar_type() == at::kFloat &&
          gamma.device() == input.device() && beta.device() == input.device(),
      "vision norm affine mismatch");
  TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16,
              "vision norm output dtype");
  const auto b = input.size(0), h = input.size(1), w = input.size(2);
  TORCH_CHECK(!windowed || (h % 24 == 0 && w % 24 == 0),
              "vision windows require dimensions divisible by 24");
#ifdef SAM3_WITH_CUDA
  const auto aligned = [](const at::Tensor &x) {
    return reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0;
  };
  // Degenerate window grids can remain noncontiguous views in ATen.
  // Preserve their exact layout through the reference expression.
  if ((!windowed || (h > 24 && w > 24)) && input.is_cuda() &&
      input.numel() > 0 && input.numel() / 1024 <= 2147483647 &&
      input.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous() &&
      aligned(input) && aligned(gamma) && aligned(beta) && !input.is_neg() &&
      !gamma.is_neg() && !beta.is_neg()) {
    auto out = at::empty(
        windowed ? std::vector<int64_t>{b * (h / 24) * (w / 24), 24, 24, 1024}
                 : input.sizes().vec(),
        input.options().dtype(type));
    vision_norm_projection_cuda(input, gamma, beta, out, windowed);
    return out;
  }
#endif
  auto out = at::layer_norm(input, {1024}, gamma, beta, 1e-5);
  if (windowed)
    out = out.view({b, h / 24, 24, w / 24, 24, 1024})
              .permute({0, 1, 3, 2, 4, 5})
              .reshape({-1, 24, 24, 1024});
  return out.to(type);
}
static std::tuple<at::Tensor, at::Tensor>
vision_residual_norm_impl(const at::Tensor &input, const at::Tensor &attention,
                          const at::Tensor &gamma, const at::Tensor &beta,
                          at::ScalarType type, bool windowed,
                          bool partition_output) {
  c10::InferenceMode inference;
  TORCH_CHECK(input.dim() == 4 && input.size(3) == 1024 &&
                  input.scalar_type() == at::kFloat,
              "vision residual norm requires FP32 [B,H,W,1024]");
  const auto b = input.size(0), h = input.size(1), w = input.size(2);
  TORCH_CHECK(!(windowed || partition_output) || (h % 24 == 0 && w % 24 == 0),
              "vision windows require dimensions divisible by 24");
  const auto shape =
      windowed ? std::vector<int64_t>{b * (h / 24) * (w / 24), 24, 24, 1024}
               : input.sizes().vec();
  TORCH_CHECK(attention.sizes() == at::IntArrayRef(shape) &&
                  attention.device() == input.device() &&
                  (attention.scalar_type() == at::kFloat ||
                   attention.scalar_type() == at::kHalf ||
                   attention.scalar_type() == at::kBFloat16),
              "vision attention shape/device/type mismatch");
  TORCH_CHECK(
      gamma.sizes() == at::IntArrayRef({1024}) &&
          beta.sizes() == gamma.sizes() && gamma.scalar_type() == at::kFloat &&
          beta.scalar_type() == at::kFloat &&
          gamma.device() == input.device() && beta.device() == input.device(),
      "vision norm affine mismatch");
  TORCH_CHECK(type == at::kFloat || type == at::kHalf || type == at::kBFloat16,
              "vision norm output dtype");
#ifdef SAM3_WITH_CUDA
  const auto aligned = [](const at::Tensor &x) {
    return reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0;
  };
  if ((!partition_output || (h > 24 && w > 24)) && input.is_cuda() &&
      input.numel() > 0 && input.numel() / 1024 <= 2147483647 &&
      input.is_contiguous() && attention.is_contiguous() &&
      gamma.is_contiguous() && beta.is_contiguous() && aligned(input) &&
      aligned(attention) && aligned(gamma) && aligned(beta) &&
      !input.is_neg() && !attention.is_neg() && !gamma.is_neg() &&
      !beta.is_neg()) {
    auto sum = at::empty_like(input);
    auto out = at::empty(
        partition_output
            ? std::vector<int64_t>{b * (h / 24) * (w / 24), 24, 24, 1024}
            : input.sizes().vec(),
        input.options().dtype(type));
    vision_residual_norm_cuda(input, attention, gamma, beta, sum, out, windowed,
                              partition_output);
    return {sum, out};
  }
#endif
  auto a = attention;
  if (windowed)
    a = a.reshape({b, h / 24, w / 24, 24, 24, 1024})
            .permute({0, 1, 3, 2, 4, 5})
            .reshape({b, h, w, 1024});
  auto sum = input + a;
  auto normalized = at::layer_norm(sum, {1024}, gamma, beta, 1e-5);
  if (partition_output)
    normalized = normalized.view({b, h / 24, 24, w / 24, 24, 1024})
                     .permute({0, 1, 3, 2, 4, 5})
                     .reshape({-1, 24, 24, 1024});
  return {sum, normalized.to(type)};
}

std::tuple<at::Tensor, at::Tensor>
vision_residual_norm(const at::Tensor &x, const at::Tensor &a,
                     const at::Tensor &g, const at::Tensor &b,
                     at::ScalarType type, bool windowed) {
  return vision_residual_norm_impl(x, a, g, b, type, windowed, false);
}
std::tuple<at::Tensor, at::Tensor>
vision_residual_norm_projection(const at::Tensor &x, const at::Tensor &a,
                                const at::Tensor &g, const at::Tensor &b,
                                at::ScalarType type, bool partition_output) {
  return vision_residual_norm_impl(x, a, g, b, type, false, partition_output);
}

} // namespace sam3
