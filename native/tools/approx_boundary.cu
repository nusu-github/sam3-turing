#include "../src/approx_kernels.h"
// Experimental FC1 restore/GELU -> FC2 quantization without an FP16 global tensor.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cstdlib>
#include <string>

template<int Items,int Threads=256,bool ReadOnly=false>
__global__ void restore_quant_rows(const int* accum, const float* xs, const float* ws,
    const half* bias, signed char* q, float* scales, int n) {
  const int row=blockIdx.x;
  float values[Items], mx=0;
  const float xscale=xs[row];
  #pragma unroll
  for(int j=0;j<Items;++j) {
    const int col=threadIdx.x+j*Threads;
    float v=0;
    if(col<n) {
      const float w=ReadOnly ? __ldg(ws+col) : ws[col];
      const half b=ReadOnly ? __ldg(bias+col) : bias[col];
      v=float(accum[row*n+col])*xscale*w+__half2float(b);
      v=.5f*v*(1.f+erff(v*.7071067811865475f));
      // Preserve the existing two-kernel INT8 path's rounding before absmax.
      v=__half2float(__float2half_rn(v));
    }
    values[j]=v; mx=fmaxf(mx,fabsf(v));
  }
  using Reduce = cub::BlockReduce<float,Threads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float scale;
  // Publish CUB's thread-0 result before all threads quantize their cached values.
  mx=Reduce(storage).Reduce(mx,cuda::maximum<>{});
  if(threadIdx.x==0) { scale=fmaxf(mx/127.f,1e-12f);scales[row]=scale; }
  __syncthreads();
  #pragma unroll
  for(int j=0;j<Items;++j) {
    const int col=threadIdx.x+j*Threads;
    if(col<n) q[row*n+col]=(signed char)__float2int_rn(fminf(127.f,fmaxf(-127.f,values[j]/scale)));
  }
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
  static const std::string mode=[] {const auto* p=std::getenv("SAM3_EXPERIMENT_BOUNDARY");return std::string(p?p:"exact");}();
  TORCH_CHECK(mode=="exact" || mode=="threads128" || mode=="threads512" || mode=="readonly","invalid boundary experiment: ",mode);
#define LAUNCH(I,T,R) restore_quant_rows<I,T,R><<<accum.size(0),T,0,stream>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),n)
  if(n==4736 && mode!="exact") {
    if(mode=="threads128") {LAUNCH(37,128,false);}
    else if(mode=="threads512") {LAUNCH(10,512,false);}
    else {LAUNCH(19,256,true);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return;
  }
#undef LAUNCH
  if(n==4736)
    restore_quant_rows<19><<<accum.size(0),256,0,stream>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),n);
  else
    restore_quant_rows<32><<<accum.size(0),256,0,stream>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
