#include "sam3/pixel_transform.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
namespace sam3 {
namespace {
__global__ void pixel_transpose(const at::Half* input,float* output,int channels,int spatial) {
  __shared__ float tile[32][33];
  int c=blockIdx.x*32+threadIdx.x,s=blockIdx.y*32+threadIdx.y;
  const int base=blockIdx.z*channels*spatial;
  for(int j=0;j<32;j+=8)
    if(c<channels && s+j<spatial)tile[threadIdx.y+j][threadIdx.x]=float(input[base+(s+j)*channels+c]);
  __syncthreads();
  c=blockIdx.x*32+threadIdx.y;s=blockIdx.y*32+threadIdx.x;
  for(int j=0;j<32;j+=8)
    if(c+j<channels && s<spatial)output[base+(c+j)*spatial+s]=tile[threadIdx.x][threadIdx.y+j];
}
}
at::Tensor pixel_nchw_float(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim()==4 && input.scalar_type()==at::kHalf && input.is_contiguous(at::MemoryFormat::ChannelsLast),"pixel conversion requires channels-last CUDA FP16 NCHW tensor");
  TORCH_CHECK(!input.is_neg() && !input.is_conj(),"pixel conversion requires resolved input values");
  TORCH_CHECK(input.numel()>0 && input.numel()<INT_MAX && input.size(0)<=65535 && (input.size(2)*input.size(3)+31)/32<=65535,"pixel conversion shape exceeds launch limits");
  const c10::cuda::CUDAGuard guard(input.device());
  const int b=input.size(0),c=input.size(1),s=input.size(2)*input.size(3);
  auto output=at::empty(input.sizes(),input.options().dtype(at::kFloat));
  pixel_transpose<<<dim3((c-1)/32+1,(s-1)/32+1,b),dim3(32,8),0,c10::cuda::getCurrentCUDAStream()>>>(input.const_data_ptr<at::Half>(),output.mutable_data_ptr<float>(),c,s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
}
