// Adapter for external ComfyKitchen PR #103, pinned in the experiment report.
// The external Apache-2.0 kernels are included without modifying their math.
#include "sam3/kitchen_attention.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "sage_attention/qk_int_sv_i8_cuda.cuh"
#include <climits>
extern "C" void launch_quant_qk_per_thread_int8(const void*,void*,void*,const void*,void*,void*,int,void*,void*,int,int,int,int,int,int,int,int,int,int,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,int,void*,cudaStream_t);
extern "C" void launch_quant_v_int8_kernel(const void*,void*,void*,int,int,int,int,int,int64_t,int64_t,int64_t,int,cudaStream_t);
namespace sam3 {
at::Tensor kitchen_attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,bool rotation,bool sequence_output){
  TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.size(3)==64 && q.size(0)>0 && q.size(1)>0 && q.size(2)>0 && q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.numel()<INT_MAX,"kitchen adapter requires equal nonempty BHND, D64");
  for(const auto& t:{q,k,v}){TORCH_CHECK(t.device()==q.device() && t.scalar_type()==at::kHalf && t.stride(3)==1,"kitchen attention requires CUDA FP16, unit channel stride");for(int i=0;i<3;++i)TORCH_CHECK(t.stride(i)>0 && t.stride(i)<INT_MAX && t.stride(i)%8==0,"unsupported kitchen attention stride");}
  c10::cuda::CUDAGuard guard(q.device());cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,q.get_device()));TORCH_CHECK(prop.major==7 && prop.minor==5,"kitchen experiment requires SM75");
  const int b=int(q.size(0)),h=int(q.size(1)),n=int(q.size(2)),d=64,padded=((n+63)/64)*64;
  TORCH_CHECK(int64_t(b)*h*d*padded<INT_MAX,"kitchen padded layout too large");
  auto qi=at::empty(q.sizes(),q.options().dtype(at::kChar)),ki=at::empty_like(qi),vi=at::empty({b,h,d,padded},qi.options());
  auto qs=at::empty({b,h,((n+127)/128)*32},q.options().dtype(at::kFloat)),ks=at::empty({b,h,((n+63)/64)*4},qs.options()),vs=at::empty({b,h,d},qs.options());
  auto anchor=at::empty({b,h},q.options().dtype(at::kInt));
  // Keep the BHND logical interface, but optionally write directly into BNHD
  // storage. The caller's head concatenation then becomes a view, not a copy.
  auto out=sequence_output ? at::empty({b,n,h,d},q.options()).transpose(1,2) : at::empty(q.sizes(),q.options());
  auto stream=c10::cuda::getCurrentCUDAStream();
  launch_quant_qk_per_thread_int8(q.const_data_ptr(),qi.mutable_data_ptr(),qs.mutable_data_ptr(),k.const_data_ptr(),ki.mutable_data_ptr(),ks.mutable_data_ptr(),0,nullptr,nullptr,b,h,n,h,n,d,128,32,64,64,q.stride(0),q.stride(1),q.stride(2),k.stride(0),k.stride(1),k.stride(2),1,rotation?1:0,1,anchor.mutable_data_ptr(),stream);
  launch_quant_v_int8_kernel(v.const_data_ptr(),vi.mutable_data_ptr(),vs.mutable_data_ptr(),b,h,n,d,padded,v.stride(0),v.stride(1),v.stride(2),1,stream);
  auto kernel=qk_int_sv_i8_attn_kernel<128,64,32,64,64,DataType::kInt8,QuantGranularity::kPerThread,QuantGranularity::kPerThread,float,false,half,ComputeUnit::kCudaCore,MaskMode::kNone,false,true,false,false,true>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,16384));
  kernel<<<dim3((n+127)/128,h,b),dim3(32,4),16384,stream>>>(qi.mutable_data_ptr<int8_t>(),ki.mutable_data_ptr<int8_t>(),vi.mutable_data_ptr<int8_t>(),reinterpret_cast<half*>(out.mutable_data_ptr<at::Half>()),nullptr,qs.mutable_data_ptr<float>(),ks.mutable_data_ptr<float>(),vs.mutable_data_ptr<float>(),nullptr,nullptr,0,0,0,0,0,n,n,1,h*n*d,d,n*d,h*n*d,d,n*d,h*d*padded,d*padded,padded,int(out.stride(0)),int(out.stride(2)),int(out.stride(1)),.125f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
