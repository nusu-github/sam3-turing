#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

namespace sam3 {
namespace {
constexpr int threads = 256;
int blocks(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements - 1) / threads + 1, 4096));
}
__global__ void pack_kernel(const bool* src, uint8_t* dst, int64_t total, int64_t pixels, int64_t bytes) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const auto row = i / bytes;
    const auto offset = (i % bytes) * 8;
    uint8_t value = 0;
    for (int bit = 0; bit < 8 && offset + bit < pixels; ++bit)
      value |= static_cast<uint8_t>(src[row * pixels + offset + bit]) << bit;
    dst[i] = value;
  }
}
__global__ void unpack_kernel(const uint8_t* src, bool* dst, int64_t total, int64_t pixels, int64_t bytes) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const auto p = i % pixels;
    dst[i] = (src[(i / pixels) * bytes + p / 8] >> (p % 8)) & 1;
  }
}
__global__ void nms_kernel(const bool* suppress, bool* keep, int64_t n) {
  __shared__ int active;
  for (int64_t i = 0; i < n; ++i) {
    if (threadIdx.x == 0) active = keep[i];
    __syncthreads();
    if (active) {
      for (int64_t j = i + 1 + threadIdx.x; j < n; j += blockDim.x)
        if (suppress[i * n + j]) keep[j] = false;
    }
    __syncthreads();
  }
}
}
at::Tensor pack_masks_cuda(const at::Tensor& masks) {
  const c10::cuda::CUDAGuard guard(masks.device());
  const auto pixels = masks.size(1) * masks.size(2);
  const auto bytes = pixels / 8 + (pixels % 8 != 0);
  auto output = at::empty({masks.size(0), bytes}, masks.options().dtype(at::kByte));
  if (output.numel()) {
    pack_kernel<<<blocks(output.numel()), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        masks.const_data_ptr<bool>(), output.mutable_data_ptr<uint8_t>(), output.numel(), pixels, bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
at::Tensor unpack_masks_cuda(const at::Tensor& packed, int64_t height, int64_t width) {
  const c10::cuda::CUDAGuard guard(packed.device());
  const auto pixels = height * width;
  const auto bytes = pixels / 8 + (pixels % 8 != 0);
  auto output = at::empty({packed.size(0), 1, height, width}, packed.options().dtype(at::kBool));
  if (output.numel()) {
    unpack_kernel<<<blocks(output.numel()), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        packed.const_data_ptr<uint8_t>(), output.mutable_data_ptr<bool>(), output.numel(), pixels, bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
at::Tensor nms_keep_cuda(const at::Tensor& suppression) {
  const c10::cuda::CUDAGuard guard(suppression.device());
  const auto n = suppression.size(0);
  auto keep = at::ones({n}, suppression.options());
  if (n) {
    nms_kernel<<<1, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        suppression.const_data_ptr<bool>(), keep.mutable_data_ptr<bool>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return keep;
}
}
