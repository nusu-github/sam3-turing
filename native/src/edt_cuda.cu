#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "edt_impl.h"
#include <algorithm>
namespace sam3 {
namespace {
__global__ void edt_pass(const float* input, float* output, int64_t* locations,
                          double* boundaries, int64_t lines, int64_t h, int64_t w,
                          bool vertical) {
  for (int64_t line = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       line < lines; line += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const auto base = vertical ? (line / w) * h * w + line % w : line * w;
    edt_line(input + base, output + base, locations + base, boundaries + base,
             vertical ? h : w, vertical ? w : 1);
  }
}
}
at::Tensor edt_cuda(const at::Tensor& values) {
  const c10::cuda::CUDAGuard guard(values.device());
  const auto b = values.size(0), h = values.size(1), w = values.size(2);
  auto tmp = at::empty_like(values);
  auto output = at::empty_like(values);
  if (!b) return output;
  auto locations = at::empty_like(values, values.options().dtype(at::kLong));
  auto boundaries = at::empty_like(values, values.options().dtype(at::kDouble));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const auto blocks = [](int64_t lines) { return static_cast<int>(std::min<int64_t>((lines - 1) / 64 + 1, 4096)); };
  edt_pass<<<blocks(b * h), 64, 0, stream>>>(values.const_data_ptr<float>(), tmp.mutable_data_ptr<float>(),
      locations.mutable_data_ptr<int64_t>(), boundaries.mutable_data_ptr<double>(), b * h, h, w, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  edt_pass<<<blocks(b * w), 64, 0, stream>>>(tmp.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
      locations.mutable_data_ptr<int64_t>(), boundaries.mutable_data_ptr<double>(), b * w, h, w, true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.sqrt_();
}
}
