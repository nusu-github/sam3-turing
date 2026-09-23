#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace sam3 {
namespace {
__global__ void axes(const float *dim, float *table, int64_t h, int64_t w) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < (h + w) * 128; i += int64_t(blockDim.x) * gridDim.x) {
    auto row = i / 128, channel = i % 128;
    auto extent = row < h ? h : w;
    auto coordinate = row < h ? row : row - h;
    const float normalized = (float(coordinate + 1) / (float(extent) + 1e-6f)) *
                             float(6.2831853071795864769);
    const float angle = normalized / dim[channel];
    table[i] = (channel % 2 == 0) ? ::sinf(angle) : ::cosf(angle);
  }
}
template <typename T>
__global__ void expand(const float *table, T *out, int64_t h, int64_t w,
                       int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    const auto channel = i % 256, pixel = i / 256, x = pixel % w,
               y = (pixel / w) % h;
    out[i] = T(table[(channel < 128 ? y : h + x) * 128 + channel % 128]);
  }
}
} // namespace
at::Tensor vision_position_cuda(const at::Tensor &input) {
  c10::cuda::CUDAGuard guard(input.device());
  const auto b = input.size(0), h = input.size(2), w = input.size(3);
  auto result = at::empty({b, h, w, 256}, input.options());
  if (result.numel() == 0)
    return result.permute({0, 3, 1, 2});
  const auto options = input.options().dtype(at::kFloat);
  auto dim = at::arange(128, options);
  dim = at::pow(10000.0, 2 * dim.floor_divide(2) / 128);
  auto table = at::empty({h + w, 128}, options);
  axes<<<std::min<int64_t>(((h + w) * 128 + 255) / 256, 4096), 256, 0,
         c10::cuda::getCurrentCUDAStream()>>>(
      dim.const_data_ptr<float>(), table.mutable_data_ptr<float>(), h, w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "position_expand", [&] {
        expand<scalar_t>
            <<<std::min<int64_t>((result.numel() + 255) / 256, 4096), 256, 0,
               c10::cuda::getCurrentCUDAStream()>>>(
                table.const_data_ptr<float>(),
                result.mutable_data_ptr<scalar_t>(), h, w, result.numel());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result.permute({0, 3, 1, 2});
}

} // namespace sam3
