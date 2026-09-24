#include "../src/approx_kernels.h"
// Experimental FC1 restore/GELU -> FC2 quantization without an FP16 global tensor.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <climits>

constexpr int boundary_threads=256;
template<int Items,bool Affine=false,bool HalfInput=false>
__global__ void restore_quant_rows(const int* accum, const float* xs, const float* ws,
    const half* bias, signed char* q, float* scales, int n,
    const float* channel_scale=nullptr,const float* channel_shift=nullptr,const half* input=nullptr) {
  const int row=blockIdx.x;
  float values[Items], mx=0;
  [[maybe_unused]] const float xscale=HalfInput?1.f:xs[row];
  #pragma unroll
  for(int j=0;j<Items;++j) {
    const int col=threadIdx.x+j*boundary_threads;
    float v=0;
    if(col<n) {
      if constexpr(HalfInput) v=__half2float(input[row*n+col]);
      else v=float(accum[row*n+col])*xscale*ws[col]+__half2float(bias[col]);
      v=.5f*v*(1.f+erff(v*.7071067811865475f));
      // Preserve the existing two-kernel INT8 path's rounding before absmax.
      v=__half2float(__float2half_rn(v));
      if constexpr(Affine) {
        // The calibrated FC2 consumes rounded FP16 GELU, then a rounded affine input.
        v=__half2float(__float2half_rn(__fsub_rn(__fdiv_rn(v,channel_scale[col]),channel_shift[col])));
      }
    }
    values[j]=v; mx=fmaxf(mx,fabsf(v));
  }
  using Reduce = cub::BlockReduce<float,boundary_threads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float scale;
  // Publish CUB's thread-0 result before all threads quantize their cached values.
  mx=Reduce(storage).Reduce(mx,cuda::maximum<>{});
  if(threadIdx.x==0) { scale=fmaxf(mx/127.f,1e-12f);scales[row]=scale; }
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Items;++j) {
    const int col=threadIdx.x+j*boundary_threads;
    if(col<n) q[row*n+col]=(signed char)__float2int_rn(fminf(127.f,fmaxf(-127.f,values[j]/scale)));
  }
}

void approx_gelu_quant(const at::Tensor& input,const at::Tensor& r,const at::Tensor& shift,
    at::Tensor& q,at::Tensor& scales) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kHalf && input.dim()==2 &&
      input.is_contiguous() && input.size(0)>0 && input.numel()<INT_MAX &&
      input.size(1)>0 && input.size(1)<=8192,"invalid FP16 GELU boundary input");
  const auto n=input.size(1),m=input.size(0);
  TORCH_CHECK(q.device()==input.device() && scales.device()==input.device() &&
      q.is_contiguous() && scales.is_contiguous() && q.scalar_type()==at::kChar &&
      scales.scalar_type()==at::kFloat && q.sizes()==input.sizes() && scales.numel()==m,
      "invalid FP16 GELU quantization output");
  TORCH_CHECK(r.defined()==shift.defined(),"scale and shift must be provided together");
  if(r.defined()) for(const auto& t:{r,shift})
    TORCH_CHECK(t.device()==input.device() && t.is_contiguous() && t.scalar_type()==at::kFloat &&
        t.numel()==n,"invalid FP16 GELU calibration vector");
  const c10::cuda::CUDAGuard guard(input.device());
  auto stream=c10::cuda::getCurrentCUDAStream();
#define GELU_LAUNCH(I,A) restore_quant_rows<I,A,true><<<m,256,0,stream>>>(nullptr,nullptr,nullptr,nullptr,q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),int(n),r.defined()?r.const_data_ptr<float>():nullptr,shift.defined()?shift.const_data_ptr<float>():nullptr,reinterpret_cast<const half*>(input.const_data_ptr<at::Half>()))
  if(r.defined()) {
    if(n==4736) {GELU_LAUNCH(19,true);} else {GELU_LAUNCH(32,true);}
  } else {
    if(n==4736) {GELU_LAUNCH(19,false);} else {GELU_LAUNCH(32,false);}
  }
#undef GELU_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void approx_restore_quant_affine(const at::Tensor& accum,const at::Tensor& xs,
    const at::Tensor& ws,const at::Tensor& bias,const at::Tensor& r,const at::Tensor& shift,
    at::Tensor& q,at::Tensor& scales) {
  TORCH_CHECK(accum.is_cuda() && accum.scalar_type()==at::kInt && accum.dim()==2 &&
      accum.is_contiguous() && accum.size(0)>0 && accum.numel()<INT_MAX &&
      accum.size(1)>0 && accum.size(1)<=8192,"invalid calibrated boundary accumulator");
  const auto n=accum.size(1),m=accum.size(0);
  for(const auto& t:{xs,ws,bias,r,shift,q,scales})
    TORCH_CHECK(t.device()==accum.device() && t.is_contiguous(),"calibrated boundary device/layout mismatch");
  TORCH_CHECK(xs.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat &&
      r.scalar_type()==at::kFloat && shift.scalar_type()==at::kFloat && scales.scalar_type()==at::kFloat &&
      bias.scalar_type()==at::kHalf && q.scalar_type()==at::kChar && q.sizes()==accum.sizes() &&
      xs.numel()==m && scales.numel()==m && ws.numel()==n && bias.numel()==n && r.numel()==n && shift.numel()==n,
      "calibrated boundary operand type/shape mismatch");
  // Positive finite calibration vectors are validated once when loaded.
  const c10::cuda::CUDAGuard guard(accum.device());
  auto stream=c10::cuda::getCurrentCUDAStream();
#define AFFINE_LAUNCH(I) restore_quant_rows<I,true><<<m,256,0,stream>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),int(n),r.const_data_ptr<float>(),shift.const_data_ptr<float>())
  if(n==4736) {AFFINE_LAUNCH(19);} else {AFFINE_LAUNCH(32);}
#undef AFFINE_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void approx_restore_quant(const at::Tensor& accum,const at::Tensor& xs,
    const at::Tensor& ws,const at::Tensor& bias,at::Tensor& q,at::Tensor& scales) {
  TORCH_CHECK(accum.dim()==2 && accum.is_cuda() && accum.is_contiguous() &&
              accum.scalar_type()==at::kInt,"restore_quant requires contiguous CUDA INT32 matrix");
  const int n=int(accum.size(1));
  TORCH_CHECK(n>0 && n<=8192,"restore_quant width must be in 1..8192");
  TORCH_CHECK(q.sizes()==accum.sizes() && q.scalar_type()==at::kChar && q.is_contiguous(),"invalid quantized output");
  TORCH_CHECK(xs.numel()==accum.size(0) && scales.numel()==accum.size(0) &&
              ws.numel()==n && bias.numel()==n,"restore_quant scale/bias shape mismatch");
  for(const auto& t : {xs,ws,bias,q,scales})
    TORCH_CHECK(t.device()==accum.device() && t.is_contiguous(),"restore_quant device/layout mismatch");
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(n==4736)
    restore_quant_rows<19><<<accum.size(0),256,0,stream>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),n);
  else
    restore_quant_rows<32><<<accum.size(0),256,0,stream>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
