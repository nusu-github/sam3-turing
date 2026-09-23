#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

namespace sam3 {
namespace {
struct Layout { int64_t sizes[4], input[4], output[4]; };
template <typename scalar_t>
__global__ void rotary_kernel(const scalar_t* input,
    const c10::complex<float>* frequencies, scalar_t* output,
    Layout layout, int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(blockDim.x) * gridDim.x) {
    auto index = i;
    int64_t source = 0, destination = 0, frequency = 0;
    for (int k = 3; k >= 0; --k) {
      const auto size = k == 3 ? layout.sizes[k] / 2 : layout.sizes[k];
      const auto coordinate = index % size;
      index /= size;
      source += coordinate * layout.input[k] * (k == 3 ? 2 : 1);
      destination += coordinate * layout.output[k] * (k == 3 ? 2 : 1);
      if (k == 3) frequency = coordinate;
      if (k == 2) frequency += coordinate * layout.sizes[3] / 2;
    }
    // Preserve the reference complex arithmetic (including FMA rounding).
    const auto value = c10::complex<float>(float(input[source]),
                                         float(input[source + layout.input[3]]));
    const auto freq = frequencies[frequency];
    const auto rotated = value * freq;
    output[destination] = scalar_t(rotated.real());
#ifdef _WIN32
    // Windows LibTorch 2.10 cu130 contracts a*d + b*c as fma(a,d,round(b*c)).
    // NVCC can instead contract b*c in this fused kernel. Specify the rounding
    // points so its imaginary component stays bitwise equal to the reference.
    output[destination + layout.output[3]] = scalar_t(__fmaf_rn(
        value.real(), freq.imag(), __fmul_rn(value.imag(), freq.real())));
#else
    output[destination + layout.output[3]] = scalar_t(rotated.imag());
#endif
  }
}
}
void rotary_embedding_cuda(const at::Tensor& value, const at::Tensor& frequencies, at::Tensor& output) {
  const c10::cuda::CUDAGuard guard(value.device());
  const auto count = value.numel() / 2;
  if (!count) return;
  Layout layout;
  for (int i = 0; i < 4; ++i) {
    layout.sizes[i] = value.size(i);
    layout.input[i] = value.stride(i);
    layout.output[i] = output.stride(i);
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((count - 1) / threads + 1, 4096));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, value.scalar_type(), "rotary_embedding", [&] {
    rotary_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        value.const_data_ptr<scalar_t>(), frequencies.const_data_ptr<c10::complex<float>>(),
        output.mutable_data_ptr<scalar_t>(), layout, count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
