#include "sam3/turing_attention.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "flash_fwd_kernel.h"

namespace sam3 {
namespace {
using Traits=Flash_fwd_kernel_traits<64,128,128,8>;
constexpr size_t q_elements=cute::cosize(typename Traits::SmemLayoutQ{});
constexpr size_t k_elements=cute::cosize(typename Traits::SmemLayoutK{});
constexpr size_t v_elements=cute::cosize(typename Traits::SmemLayoutV{});
static_assert(q_elements==128*64 && k_elements==128*64 && v_elements==128*64);
constexpr int required_smem=int((q_elements+k_elements)*sizeof(half_t));
static_assert(required_smem==32768);
template<bool Even>
__global__ __launch_bounds__(256) void forward64(half_t* q,half_t* k,half_t* v,
    half_t* out,float* lse,int batch,int length,int heads) {
  compute_attn<Traits,false,Even>(q,k,v,out,lse,nullptr,nullptr,batch,length,length,
                                heads,heads,1,64,0.125f,0);
}
}
at::Tensor turing_attention(const at::Tensor& q,const at::Tensor& k,
    const at::Tensor& v,bool reserve64) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kHalf && q.dim()==4 && q.size(3)==64,
              "Turing experiment requires CUDA FP16 BHND, D=64");
  TORCH_CHECK(q.size(0)>0 && q.size(1)>0 && q.size(2)>0 && q.size(0)<=65535 && q.size(1)<=65535,
              "invalid attention shape");
  TORCH_CHECK(k.sizes()==q.sizes() && v.sizes()==q.sizes() && k.device()==q.device() &&
              v.device()==q.device() && k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type(),
              "Turing experiment requires matching dense self-attention operands");
  const c10::cuda::CUDAGuard device(q.device());
  cudaDeviceProp props;C10_CUDA_CHECK(cudaGetDeviceProperties(&props,q.get_device()));
  TORCH_CHECK(props.major==7 && props.minor==5,"Turing experiment supports SM75 only");
  auto qt=q.transpose(1,2).contiguous(),kt=k.transpose(1,2).contiguous(),vt=v.transpose(1,2).contiguous();
  auto out=at::empty_like(qt),lse=at::empty({q.size(0),q.size(1),q.size(2)},q.options().dtype(at::kFloat));
  const int batch=int(q.size(0)),heads=int(q.size(1)),length=int(q.size(2));
  const int smem=reserve64?65536:required_smem;
  const dim3 grid((length+127)/128,batch,heads);
  auto stream=c10::cuda::getCurrentCUDAStream(q.get_device());
  auto qp=reinterpret_cast<half_t*>(qt.data_ptr<at::Half>());
  auto kp=reinterpret_cast<half_t*>(kt.data_ptr<at::Half>());
  auto vp=reinterpret_cast<half_t*>(vt.data_ptr<at::Half>());
  auto op=reinterpret_cast<half_t*>(out.data_ptr<at::Half>());
  if(length%128==0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(forward64<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    forward64<true><<<grid,256,smem,stream>>>(qp,kp,vp,op,lse.data_ptr<float>(),batch,length,heads);
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(forward64<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    forward64<false><<<grid,256,smem,stream>>>(qp,kp,vp,op,lse.data_ptr<float>(),batch,length,heads);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.transpose(1,2);
}
std::vector<int64_t> turing_attention_resources(bool reserve64) {
  cudaFuncAttributes attrs;int blocks;
  const int smem=reserve64?65536:required_smem;
  C10_CUDA_CHECK(cudaFuncSetAttribute(forward64<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,forward64<false>));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,forward64<false>,256,smem));
  return {attrs.numRegs,int64_t(attrs.localSizeBytes),int64_t(attrs.sharedSizeBytes),smem,blocks};
}
}
