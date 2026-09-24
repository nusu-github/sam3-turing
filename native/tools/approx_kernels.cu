#include "../src/approx_kernels.h"
// Experimental kernels for diagnostics and the opt-in INT8 MLP path.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cstdlib>
#include <cstring>
#include <climits>

__global__ void quant_rows(const half* x, signed char* q, float* scales, int k) {
  const int row=blockIdx.x;
  float mx=0;
  for(int j=threadIdx.x;j<k;j+=blockDim.x) mx=fmaxf(mx,fabsf(__half2float(x[row*k+j])));
  using Reduce = cub::BlockReduce<float,256>;
  __shared__ Reduce::TempStorage storage;
  __shared__ float scale;
  // Local absmax values are nonnegative and NaN-free. Only thread 0 owns the result.
  mx=Reduce(storage).Reduce(mx,cuda::maximum<>{});
  if(threadIdx.x==0) { scale=fmaxf(mx/127.f,1e-12f);scales[row]=scale; }
  __syncthreads();
  for(int j=threadIdx.x;j<k;j+=blockDim.x)
    q[row*k+j]=(signed char)__float2int_rn(fminf(127.f,fmaxf(-127.f,__half2float(x[row*k+j])/scale)));
}
__global__ void restore(const int* accum,const float* xs,const float* ws,const half* bias,half* out,int m,int n,bool gelu) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=m*n)return;
  float v=float(accum[i])*xs[i/n]*ws[i%n]+__half2float(bias[i%n]);
  if(gelu)v=.5f*v*(1.f+erff(v*.7071067811865475f));
  out[i]=__float2half_rn(v);
}
__global__ void restore_rows(const int* accum,const float* xs,const float* ws,const half* bias,half* out,int n,bool gelu) {
  const int row=blockIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;
  if(col>=n)return;
  const int i=row*n+col;
  float v=float(accum[i])*xs[row]*ws[col]+__half2float(bias[col]);
  if(gelu)v=.5f*v*(1.f+erff(v*.7071067811865475f));
  out[i]=__float2half_rn(v);
}
void approx_quant(const at::Tensor& x,at::Tensor& q,at::Tensor& scales) {
  quant_rows<<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),q.mutable_data_ptr<int8_t>(),scales.mutable_data_ptr<float>(),int(x.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void approx_restore(const at::Tensor& accum,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias,at::Tensor& out,bool gelu) {
  static const bool rows=[] {
    const auto* p=std::getenv("SAM3_EXPERIMENT_RESTORE");
    return p && std::strcmp(p,"rows")==0;
  }();
  if(rows && out.size(0)<=65535) {
    restore_rows<<<dim3((out.size(1)+255)/256,out.size(0)),256,0,c10::cuda::getCurrentCUDAStream()>>>(
      accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),reinterpret_cast<half*>(out.mutable_data_ptr<at::Half>()),int(out.size(1)),gelu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return;
  }
  restore<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),reinterpret_cast<half*>(out.mutable_data_ptr<at::Half>()),int(out.size(0)),int(out.size(1)),gelu);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Preserve the original restore's Half rounding before the complex multiply.
__global__ void restore_rope(const int* accum,const float* xs,const float* ws,
    const half* bias,const c10::complex<float>* freq,half* out,int rows,int length) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*512)return;
  const int row=i/512,within=i%512,part=blockIdx.y,col=part*1024+within*2;
  const int source=row*3072+col;
  float a=float(accum[source])*xs[row]*ws[col]+__half2float(bias[col]);
  float b=float(accum[source+1])*xs[row]*ws[col+1]+__half2float(bias[col+1]);
  a=__half2float(__float2half_rn(a));b=__half2float(__float2half_rn(b));
  if(part<2) {
    const auto f=freq[(row%length)*32+within%32];
    const auto result=c10::complex<float>(a,b)*f;
#ifdef _WIN32
    b=__fmaf_rn(a,f.imag(),__fmul_rn(b,f.real()));
#else
    b=__fmaf_rn(b,f.real(),__fmul_rn(a,f.imag()));
#endif
    a=result.real();
  }
  const int target=part*rows*1024+i*2;
  out[target]=__float2half_rn(a);out[target+1]=__float2half_rn(b);
}
at::Tensor approx_restore_rope(const at::Tensor& accum,const at::Tensor& xs,
    const at::Tensor& ws,const at::Tensor& bias,const at::Tensor& freq,int64_t batch) {
  TORCH_CHECK(accum.is_cuda() && accum.scalar_type()==at::kInt && accum.dim()==2 &&
              accum.size(1)==3072 && accum.is_contiguous() && batch>0 &&
              accum.size(0)>0 && accum.size(0)%batch==0 && accum.numel()<INT_MAX,
              "restore/RoPE requires contiguous INT32 [B*N,3072]");
  const auto rows=accum.size(0),length=rows/batch;
  TORCH_CHECK(xs.device()==accum.device() && ws.device()==accum.device() &&
              bias.device()==accum.device() && freq.device()==accum.device() &&
              xs.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat &&
              bias.scalar_type()==at::kHalf && freq.scalar_type()==at::kComplexFloat &&
              xs.numel()==rows && ws.numel()==3072 && bias.numel()==3072 &&
              freq.sizes()==at::IntArrayRef({length,32}) && xs.is_contiguous() &&
              ws.is_contiguous() && bias.is_contiguous() && freq.is_contiguous() &&
              !freq.is_conj() && !freq.is_neg(),"invalid restore/RoPE operands");
  const c10::cuda::CUDAGuard guard(accum.device());
  auto out=at::empty({3,batch,length,16,64},bias.options());
  restore_rope<<<dim3((rows*512+255)/256,3),256,0,c10::cuda::getCurrentCUDAStream()>>>(
      accum.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),
      reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),
      freq.const_data_ptr<c10::complex<float>>(),reinterpret_cast<half*>(out.mutable_data_ptr<at::Half>()),int(rows),int(length));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.permute({0,1,3,2,4});
}
