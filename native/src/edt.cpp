#include "sam3/ops.h"
#include "edt_impl.h"
#include <ATen/Parallel.h>

namespace sam3 {
#ifdef SAM3_WITH_CUDA
at::Tensor edt_cuda(const at::Tensor& values);
#endif
at::Tensor euclidean_distance_transform(const at::Tensor& masks) {
  TORCH_CHECK(masks.dim() == 3 && masks.size(1) > 0 && masks.size(2) > 0,
              "EDT expects [B,H,W] with positive height and width");
  TORCH_CHECK(masks.device().is_cpu() || masks.is_cuda(), "EDT supports CPU and CUDA");
  // Match upstream's finite infinity, including the all-foreground case.
  const auto values = masks.ne(0).to(at::kFloat).mul_(1e18).contiguous();
#ifdef SAM3_WITH_CUDA
  if (masks.is_cuda()) return edt_cuda(values);
#else
  TORCH_CHECK(!masks.is_cuda(), "library was built without CUDA kernels");
#endif
  auto intermediate = at::empty_like(values);
  auto result = at::empty_like(values);
  auto locations = at::empty_like(values, values.options().dtype(at::kLong));
  auto boundaries = at::empty_like(values, values.options().dtype(at::kDouble));
  const auto b = masks.size(0), h = masks.size(1), w = masks.size(2);
  const auto* input = values.const_data_ptr<float>();
  auto* tmp = intermediate.mutable_data_ptr<float>();
  auto* output = result.mutable_data_ptr<float>();
  auto* loc = locations.mutable_data_ptr<int64_t>();
  auto* bound = boundaries.mutable_data_ptr<double>();
  at::parallel_for(0, b * h, 1, [&](int64_t start, int64_t end) {
    for (auto row = start; row < end; ++row) {
      const auto base = row * w;
      edt_line(input + base, tmp + base, loc + base, bound + base, w, 1);
    }
  });
  at::parallel_for(0, b * w, 1, [&](int64_t start, int64_t end) {
    for (auto col = start; col < end; ++col) {
      const auto base = (col / w) * h * w + col % w;
      edt_line(tmp + base, output + base, loc + base, bound + base, h, w);
    }
  });
  return result.sqrt_();
}
}
