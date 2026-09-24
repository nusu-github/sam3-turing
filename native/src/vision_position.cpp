#include "sam3/vision_position.h"
#include <c10/core/InferenceMode.h>
#include <cmath>
namespace sam3 {
#ifdef SAM3_WITH_CUDA
at::Tensor vision_position_cuda(const at::Tensor &);
#endif
at::Tensor vision_position_encoding(const at::Tensor &x) {
  c10::InferenceMode inference;
  TORCH_CHECK(x.dim() == 4 && x.is_floating_point(),
              "position encoding requires floating [B,C,H,W]");
#ifdef SAM3_WITH_CUDA
  const auto type = x.scalar_type();
  if (x.is_cuda() && x.size(0) > 0 && x.size(2) > 0 && x.size(3) > 0 &&
      (type == at::kFloat || type == at::kHalf || type == at::kBFloat16 ||
       type == at::kDouble))
    return vision_position_cuda(x);
#endif

  const auto b = x.size(0), h = x.size(2), w = x.size(3);
  const auto options = x.options().dtype(at::kFloat);
  auto y = at::arange(1, h + 1, options).view({1, h, 1}).repeat({b, 1, w});
  auto xx = at::arange(1, w + 1, options).view({1, 1, w}).repeat({b, h, 1});
  y = y / (y.slice(1, h - 1, h) + 1e-6) * (2 * std::acos(-1.0));
  xx = xx / (xx.slice(2, w - 1, w) + 1e-6) * (2 * std::acos(-1.0));
  auto dim = at::arange(128, options);
  dim = at::pow(10000.0, 2 * dim.floor_divide(2) / 128);
  const auto encode = [&](const at::Tensor &coordinate) {
    const auto angle = coordinate.unsqueeze(-1) / dim;
    return at::stack({angle.slice(3, 0, 128, 2).sin(),
                      angle.slice(3, 1, 128, 2).cos()},
                     4)
        .flatten(3);
  };
  return at::cat({encode(y), encode(xx)}, 3)
      .permute({0, 3, 1, 2})
      .to(x.scalar_type());
}
} // namespace sam3
