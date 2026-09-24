#include "approx_kernels.h"
// Welford helpers adapted from PyTorch layer_norm_kernel.cu.
// Copyright PyTorch contributors; see ../third_party/LICENSE-PyTorch.
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>
namespace sam3 {
namespace {
using at::acc_type;
constexpr int vec_size = 4;
template <typename T, int N> struct alignas(sizeof(T) * N) aligned_vector {
  T val[N];
};
struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;
  C10_HOST_DEVICE WelfordDataLN() : mean(0.f), sigma2(0.f), count(0.f) {}
  C10_HOST_DEVICE WelfordDataLN(float mean, float sigma2, float count)
      : mean(mean), sigma2(sigma2), count(count) {}
};

template <typename U, bool rms_norm>
__device__ WelfordDataLN cuWelfordOnlineSum(const U val,
                                            const WelfordDataLN &curr_sum) {
  if constexpr (!rms_norm) {
    U delta = val - curr_sum.mean;
    U new_count = curr_sum.count + 1.f;
    U new_mean =
        curr_sum.mean +
        delta * (1.f / new_count); // proper division is slow, this is less
                                   // accurate but noticeably faster
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
  } else {
    return {0.f, curr_sum.sigma2 + val * val, 0};
  }
}

template <bool rms_norm>
__device__ WelfordDataLN cuWelfordCombine(const WelfordDataLN dataB,
                                          const WelfordDataLN dataA) {
  if constexpr (!rms_norm) {
    using U = decltype(dataB.count);
    U delta = dataB.mean - dataA.mean;
    U count = dataA.count + dataB.count;
    U mean, sigma2;
    if (count > decltype(dataB.count){0}) {
      auto coef =
          1.f / count; // NB we don't use --use_fast_math, but this is
                       // emulation, 1./count goes to intrinsic, `* coef` is
                       // multiplication, instead of slow fp division
      auto nA = dataA.count * coef;
      auto nB = dataB.count * coef;
      mean = nA * dataA.mean + nB * dataB.mean;
      sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
      mean = U(0);
      sigma2 = U(0);
    }
    return {mean, sigma2, count};
  } else {
    return {0.f, dataB.sigma2 + dataA.sigma2, 0};
  }
}

template <typename T, bool rms_norm = false>
__device__ WelfordDataLN compute_stats(const T *__restrict__ X, const int N,
                                       float *buf) {
  // X points to the row to read
  using vec_t = aligned_vector<T, vec_size>;
  using acc_t = acc_type<T, true>;
  const vec_t *X_vec = reinterpret_cast<const vec_t *>(X);
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int n_vec_to_read = N / vec_size;
  WelfordDataLN wd(0.f, 0.f, 0.f);
  // no tail, we check that N is multiple of vec_size
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
#pragma unroll
    for (int ii = 0; ii < vec_size; ii++) {
      wd = cuWelfordOnlineSum<acc_t, rms_norm>(static_cast<acc_t>(data.val[ii]),
                                               wd);
    }
  }
  // intra-warp reduction
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                      WARP_SHFL_DOWN(wd.sigma2, offset),
                      WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine<rms_norm>(wd, wdB);
  }
  // threadIdx.x == 0 has correct values for each warp
  // inter-warp reductions
  if (blockDim.y > 1) {
    float *meansigmabuf = buf;
    float *countbuf = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
      // upper half of warps write to shared
      if (threadIdx.x == 0 && threadIdx.y >= offset &&
          threadIdx.y < 2 * offset) {
        const int wrt_y = threadIdx.y - offset;
        meansigmabuf[2 * wrt_y] = wd.mean;
        meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
        countbuf[wrt_y] = wd.count;
      }
      __syncthreads();
      // lower half merges
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        WelfordDataLN wdB{meansigmabuf[2 * threadIdx.y],
                          meansigmabuf[2 * threadIdx.y + 1],
                          countbuf[threadIdx.y]};
        wd = cuWelfordCombine<rms_norm>(wd, wdB);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      meansigmabuf[0] = wd.mean;
      meansigmabuf[1] = wd.sigma2 / float(N);
    }
    __syncthreads();
    return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.f};

  } else {
    return WelfordDataLN{WARP_SHFL(wd.mean, 0),
                         WARP_SHFL(wd.sigma2, 0) / float(N), 0.f};
  }
}

__device__ WelfordDataLN reduce_residual_stats(WelfordDataLN wd, float *buf) {
  // intra-warp reduction
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                      WARP_SHFL_DOWN(wd.sigma2, offset),
                      WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine<false>(wd, wdB);
  }
  // threadIdx.x == 0 has correct values for each warp
  // inter-warp reductions
  if (blockDim.y > 1) {
    float *meansigmabuf = buf;
    float *countbuf = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
      // upper half of warps write to shared
      if (threadIdx.x == 0 && threadIdx.y >= offset &&
          threadIdx.y < 2 * offset) {
        const int wrt_y = threadIdx.y - offset;
        meansigmabuf[2 * wrt_y] = wd.mean;
        meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
        countbuf[wrt_y] = wd.count;
      }
      __syncthreads();
      // lower half merges
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        WelfordDataLN wdB{meansigmabuf[2 * threadIdx.y],
                          meansigmabuf[2 * threadIdx.y + 1],
                          countbuf[threadIdx.y]};
        wd = cuWelfordCombine<false>(wd, wdB);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      meansigmabuf[0] = wd.mean;
      meansigmabuf[1] = wd.sigma2 / float(1024);
    }
    __syncthreads();
    return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.f};

  } else {
    return WelfordDataLN{WARP_SHFL(wd.mean, 0),
                         WARP_SHFL(wd.sigma2, 0) / float(1024), 0.f};
  }
}
template <typename Out>
__global__ void norm_projection_kernel(int N, const float *input,
                                       const float *gamma, const float *beta,
                                       Out *output, int64_t height,
                                       int64_t width, bool windowed) {
  extern __shared__ float buffer[];
  const int64_t row = blockIdx.x;
  const float *source = input + row * N;
  const auto stats = compute_stats<float, false>(source, N, buffer);
  const float rstd = c10::cuda::compat::rsqrt(stats.sigma2 + 1e-5f);
  int64_t dest = row;
  if (windowed) {
    const auto x = row % width, y = (row / width) % height,
               batch = row / (height * width);
    dest = (((batch * (height / 24) + y / 24) * (width / 24) + x / 24) * 24 +
            y % 24) *
               24 +
           x % 24;
  }
  const int lane = threadIdx.x + threadIdx.y * blockDim.x;
  const auto *in = reinterpret_cast<const aligned_vector<float, 4> *>(source);
  const auto *g = reinterpret_cast<const aligned_vector<float, 4> *>(gamma);
  const auto *b = reinterpret_cast<const aligned_vector<float, 4> *>(beta);
  auto *out = reinterpret_cast<aligned_vector<Out, 4> *>(output + dest * N);
  for (int i = lane; i < N / 4; i += blockDim.x * blockDim.y) {
    auto data = in[i];
    aligned_vector<Out, 4> values;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float value =
          g[i].val[j] * (rstd * (data.val[j] - stats.mean)) + b[i].val[j];
      values.val[j] = Out(value);
    }
    out[i] = values;
  }
}

template <typename Attention, typename Out>
__global__ void
residual_norm_kernel(const float *input, const Attention *attention,
                     const float *gamma, const float *beta, float *sum,
                     Out *output, int64_t height, int64_t width, bool windowed,
                     bool partition_output) {
  extern __shared__ float buffer[];
  const int64_t row = blockIdx.x;
  int64_t other = row;
  if (windowed || partition_output) {
    const auto x = row % width, y = (row / width) % height,
               batch = row / (height * width);
    other = (((batch * (height / 24) + y / 24) * (width / 24) + x / 24) * 24 +
             y % 24) *
                24 +
            x % 24;
  }
  const int lane = threadIdx.x + threadIdx.y * blockDim.x;
  const auto *in =
      reinterpret_cast<const aligned_vector<float, 4> *>(input + row * 1024);
  const auto *attn = reinterpret_cast<const aligned_vector<Attention, 4> *>(
      attention + (windowed ? other : row) * 1024);
  aligned_vector<float, 4> values[2];
  WelfordDataLN stats(0.f, 0.f, 0.f);
  for (int part = 0; part < 2; ++part) {
    const int i = lane + part * 128;
    auto a = in[i];
    auto b = attn[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      values[part].val[j] = a.val[j] + float(b.val[j]);
      stats = cuWelfordOnlineSum<float, false>(values[part].val[j], stats);
    }
  }
  stats = reduce_residual_stats(stats, buffer);
  const float rstd = c10::cuda::compat::rsqrt(stats.sigma2 + 1e-5f);
  const auto *g = reinterpret_cast<const aligned_vector<float, 4> *>(gamma);
  const auto *b = reinterpret_cast<const aligned_vector<float, 4> *>(beta);
  auto *residual =
      reinterpret_cast<aligned_vector<float, 4> *>(sum + row * 1024);
  auto *out = reinterpret_cast<aligned_vector<Out, 4> *>(
      output + (partition_output ? other : row) * 1024);
  for (int part = 0; part < 2; ++part) {
    const int i = lane + part * 128;
    aligned_vector<Out, 4> normalized;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float value =
          g[i].val[j] * (rstd * (values[part].val[j] - stats.mean)) +
          b[i].val[j];
      normalized.val[j] = Out(value);
    }
    residual[i] = values[part];
    out[i] = normalized;
  }
}
__global__ void
fc2_residual_norm_kernel(const float *input, const int *accum,
                     const float *xs,const float *ws,const c10::Half *bias,
                     const float *gamma, const float *beta, float *sum,
                     c10::Half *output, int64_t height, int64_t width,
                     bool partition_output) {
  extern __shared__ float buffer[];
  const int64_t row = blockIdx.x;
  int64_t other = row;
  if (partition_output) {
    const auto x = row % width, y = (row / width) % height,
               batch = row / (height * width);
    other = (((batch * (height / 24) + y / 24) * (width / 24) + x / 24) * 24 +
             y % 24) *
                24 +
            x % 24;
  }
  const int lane = threadIdx.x + threadIdx.y * blockDim.x;
  const auto *in =
      reinterpret_cast<const aligned_vector<float, 4> *>(input + row * 1024);
  const auto *acc = reinterpret_cast<const aligned_vector<int,4> *>(accum+row*1024);
  aligned_vector<float, 4> values[2];
  WelfordDataLN stats(0.f, 0.f, 0.f);
  for (int part = 0; part < 2; ++part) {
    const int i = lane + part * 128;
    auto a = in[i];
    auto raw = acc[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col=i*4+j;
      const float restored=float(raw.val[j])*xs[row]*ws[col]+float(bias[col]);
      const float rounded=float(c10::Half(restored));
      values[part].val[j] = a.val[j] + rounded;
      stats = cuWelfordOnlineSum<float, false>(values[part].val[j], stats);
    }
  }
  stats = reduce_residual_stats(stats, buffer);
  const float rstd = c10::cuda::compat::rsqrt(stats.sigma2 + 1e-5f);
  const auto *g = reinterpret_cast<const aligned_vector<float, 4> *>(gamma);
  const auto *b = reinterpret_cast<const aligned_vector<float, 4> *>(beta);
  auto *residual =
      reinterpret_cast<aligned_vector<float, 4> *>(sum + row * 1024);
  auto *out = reinterpret_cast<aligned_vector<c10::Half, 4> *>(
      output + (partition_output ? other : row) * 1024);
  for (int part = 0; part < 2; ++part) {
    const int i = lane + part * 128;
    aligned_vector<c10::Half, 4> normalized;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float value =
          g[i].val[j] * (rstd * (values[part].val[j] - stats.mean)) +
          b[i].val[j];
      normalized.val[j] = c10::Half(value);
    }
    residual[i] = values[part];
    out[i] = normalized;
  }
}
template <typename Attention>
void launch_residual(const at::Tensor &input, const at::Tensor &attention,
                     const at::Tensor &gamma, const at::Tensor &beta,
                     at::Tensor &sum, at::Tensor &output, bool windowed,
                     bool partition_output) {
  const dim3 threads(32, 4);
  const auto blocks = input.numel() / 1024;
  auto stream = c10::cuda::getCurrentCUDAStream();
#define RUN(T)                                                                 \
  residual_norm_kernel<Attention, T>                                           \
      <<<blocks, threads, 6 * sizeof(float), stream>>>(                        \
          input.const_data_ptr<float>(),                                       \
          attention.const_data_ptr<Attention>(),                               \
          gamma.const_data_ptr<float>(), beta.const_data_ptr<float>(),         \
          sum.mutable_data_ptr<float>(), output.mutable_data_ptr<T>(),         \
          input.size(1), input.size(2), windowed, partition_output)
  if (output.scalar_type() == at::kHalf) {
    RUN(c10::Half);
  } else if (output.scalar_type() == at::kBFloat16) {
    RUN(c10::BFloat16);
  } else {
    RUN(float);
  }
#undef RUN
}
} // namespace
void vision_norm_projection_cuda(const at::Tensor &input,
                                 const at::Tensor &gamma,
                                 const at::Tensor &beta, at::Tensor &output,
                                 bool windowed) {
  const c10::cuda::CUDAGuard guard(input.device());
  const dim3 threads(32, 4);
  const auto blocks = input.numel() / 1024;
  auto stream = c10::cuda::getCurrentCUDAStream();
#define RUN(T)                                                                 \
  norm_projection_kernel<T><<<blocks, threads, 6 * sizeof(float), stream>>>(   \
      1024, input.const_data_ptr<float>(), gamma.const_data_ptr<float>(),      \
      beta.const_data_ptr<float>(), output.mutable_data_ptr<T>(),              \
      input.size(1), input.size(2), windowed)
  if (output.scalar_type() == at::kHalf) {
    RUN(c10::Half);
  } else if (output.scalar_type() == at::kBFloat16) {
    RUN(c10::BFloat16);
  } else {
    RUN(float);
  }
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void vision_residual_norm_cuda(const at::Tensor &input,
                               const at::Tensor &attention,
                               const at::Tensor &gamma, const at::Tensor &beta,
                               at::Tensor &sum, at::Tensor &output,
                               bool windowed, bool partition_output) {
  const c10::cuda::CUDAGuard guard(input.device());
  if (attention.scalar_type() == at::kHalf)
    launch_residual<c10::Half>(input, attention, gamma, beta, sum, output,
                               windowed, partition_output);
  else if (attention.scalar_type() == at::kBFloat16)
    launch_residual<c10::BFloat16>(input, attention, gamma, beta, sum, output,
                                   windowed, partition_output);
  else
    launch_residual<float>(input, attention, gamma, beta, sum, output, windowed,
                           partition_output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Opt-in FC2 restoration + residual + next-layer norm; keep Welford order.
std::tuple<at::Tensor,at::Tensor> approx_fc2_norm(const at::Tensor& input,
    const at::Tensor& accum,const at::Tensor& xs,const at::Tensor& ws,
    const at::Tensor& bias,const at::Tensor& gamma,const at::Tensor& beta,bool partition) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat && input.dim()==4 && input.size(3)==1024 && input.is_contiguous() && input.numel()>0,"FC2 norm requires contiguous CUDA FP32 BHWC");
  const auto rows=input.numel()/1024;
  TORCH_CHECK(rows<=2147483647 && accum.scalar_type()==at::kInt && accum.sizes()==at::IntArrayRef({rows,1024}) && xs.scalar_type()==at::kFloat && xs.numel()==rows && ws.scalar_type()==at::kFloat && ws.numel()==1024 && bias.scalar_type()==at::kHalf && bias.numel()==1024 && gamma.scalar_type()==at::kFloat && gamma.numel()==1024 && beta.scalar_type()==at::kFloat && beta.numel()==1024,"FC2 norm operands mismatch");
  for(const auto& t:{input,accum,xs,ws,bias,gamma,beta})
    TORCH_CHECK(t.device()==input.device() && t.is_contiguous() && !t.is_neg() && reinterpret_cast<uintptr_t>(t.const_data_ptr())%16==0,"FC2 norm alignment/device/layout mismatch");
  const auto b=input.size(0),h=input.size(1),w=input.size(2);
  TORCH_CHECK(!partition || (h%24==0 && w%24==0),"FC2 norm window dimensions");
  const c10::cuda::CUDAGuard guard(input.device());
  auto sum=at::empty_like(input);
  auto out=at::empty(partition?std::vector<int64_t>{b*(h/24)*(w/24),24,24,1024}:input.sizes().vec(),input.options().dtype(at::kHalf));
  fc2_residual_norm_kernel<<<rows,dim3(32,4),6*sizeof(float),c10::cuda::getCurrentCUDAStream()>>>(input.const_data_ptr<float>(),accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),bias.const_data_ptr<at::Half>(),gamma.const_data_ptr<float>(),beta.const_data_ptr<float>(),sum.mutable_data_ptr<float>(),out.mutable_data_ptr<at::Half>(),h,w,partition);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {sum,out};
}

} // namespace sam3
