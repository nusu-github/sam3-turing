#include "sam3/ops.h"
#include <ATen/Parallel.h>

namespace sam3 {
#ifdef SAM3_WITH_CUDA
std::tuple<at::Tensor, at::Tensor> components_cuda(const at::Tensor& values);
#endif
std::tuple<at::Tensor, at::Tensor> connected_components(const at::Tensor& masks) {
  TORCH_CHECK((masks.dim() == 3 || (masks.dim() == 4 && masks.size(1) == 1)),
              "connected components expects [B,H,W] or [B,1,H,W]");
  TORCH_CHECK(at::isIntegralType(masks.scalar_type(), true), "connected components expects integer or bool values");
  TORCH_CHECK(masks.device().is_cpu() || masks.is_cuda(), "connected components supports CPU and CUDA");
  const auto values = masks.to(at::kLong).contiguous();
#ifdef SAM3_WITH_CUDA
  if (masks.is_cuda()) return components_cuda(values);
#else
  TORCH_CHECK(!masks.is_cuda(), "library was built without CUDA kernels");
#endif
  auto parents = at::empty_like(values);
  auto labels = at::empty_like(values);
  auto histogram = at::zeros_like(values);
  auto sizes = at::empty_like(values);
  const auto* input = values.const_data_ptr<int64_t>();
  auto* parent = parents.mutable_data_ptr<int64_t>();
  auto* label = labels.mutable_data_ptr<int64_t>();
  auto* hist = histogram.mutable_data_ptr<int64_t>();
  auto* size = sizes.mutable_data_ptr<int64_t>();
  const auto n = values.numel(), h = values.size(-2), w = values.size(-1);
  // Components never cross batch boundaries; rows are processed in raster order.
  at::parallel_for(0, values.size(0), 1, [&](int64_t start, int64_t end) {
    const auto root = [&](int64_t i) {
      while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
      return i;
    };
    for (auto b = start; b < end; ++b) {
      const auto begin = b * h * w, finish = begin + h * w;
      for (auto i = begin; i < finish; ++i) parent[i] = input[i] != 0 ? i : -1;
      for (int64_t y = 0; y < h; ++y) for (int64_t x = 0; x < w; ++x) {
        const auto i = begin + y * w + x;
        if (!input[i]) continue;
        const auto merge = [&](int64_t j) {
          if (input[i] != input[j]) return;
          auto a = root(i), c = root(j);
          if (a > c) std::swap(a, c);
          parent[c] = a;
        };
        if (x) merge(i - 1);
        if (y) {
          if (x) merge(i - w - 1);
          merge(i - w);
          if (x + 1 < w) merge(i - w + 1);
        }
      }
      for (auto i = begin; i < finish; ++i) {
        label[i] = input[i] ? root(i) + 1 : 0;
        if (label[i]) ++hist[label[i] - 1];
      }
    }
  });
  at::parallel_for(0, n, 4096, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; ++i) size[i] = label[i] ? hist[label[i] - 1] : 0;
  });
  return {labels, sizes};
}
}
