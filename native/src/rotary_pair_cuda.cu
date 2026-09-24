#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace sam3 {
namespace {
template <typename T>
__device__ void rotate_pair(const T *value,
                            const c10::complex<float> &frequency, T *out) {
  const auto input = c10::complex<float>(float(value[0]), float(value[1]));
  const auto result = input * frequency;
  out[0] = T(result.real());
#ifdef _WIN32
  out[1] = T(__fmaf_rn(input.real(), frequency.imag(),
                       __fmul_rn(input.imag(), frequency.real())));
#else
  // Match the Linux standalone LibTorch product order explicitly.
  out[1] = T(__fmaf_rn(input.imag(), frequency.real(),
                       __fmul_rn(input.real(), frequency.imag())));
#endif
}
template <typename T>
__global__ void pair_kernel(const T *q, const T *k,
                            const c10::complex<float> *f, T *a, T *b,
                            int64_t length, int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(blockDim.x) * gridDim.x) {
    const auto row = i / 512, within = i % 512;
    const auto source = row * 3072 + within * 2;
    const auto frequency = f[(row % length) * 32 + (within % 32)];
    rotate_pair(q + source, frequency, a + i * 2);
    rotate_pair(k + source, frequency, b + i * 2);
  }
}
} // namespace
void rotary_pair_cuda(const at::Tensor &q, const at::Tensor &k,
                      const at::Tensor &f, at::Tensor &a, at::Tensor &b) {
  const c10::cuda::CUDAGuard guard(q.device());
  const auto count = q.numel() / 2;
  const int blocks = int(std::min<int64_t>((count + 255) / 256, 4096));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, q.scalar_type(), "rotary_pair", [&] {
        pair_kernel<scalar_t>
            <<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
                q.const_data_ptr<scalar_t>(), k.const_data_ptr<scalar_t>(),
                f.const_data_ptr<c10::complex<float>>(),
                a.mutable_data_ptr<scalar_t>(), b.mutable_data_ptr<scalar_t>(),
                q.size(2), count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace sam3
