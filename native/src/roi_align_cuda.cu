#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "roi_align_impl.h"
#include <algorithm>
namespace sam3 {
namespace {
template <typename T>
__global__ void roi_kernel(int64_t count, const T* input, const T* rois, T* output,
    T scale, int64_t channels, int h, int w, int ph, int pw, int ratio, bool aligned) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x)*gridDim.x)
    output[i] = roi_sample(i,input,rois,scale,channels,h,w,ph,pw,ratio,aligned);
}
}
at::Tensor roi_align_cuda(const at::Tensor& input, const at::Tensor& rois, double scale,
                         int64_t ph, int64_t pw, int64_t ratio, bool aligned) {
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty({rois.size(0),input.size(1),ph,pw},input.options());
  if (!output.numel()) return output;
  const int blocks = std::min<int64_t>((output.numel()-1)/256+1,4096);
  AT_DISPATCH_FLOATING_TYPES_AND(at::kHalf,input.scalar_type(),"roi_align_cuda",[&] {
    roi_kernel<<<blocks,256,0,c10::cuda::getCurrentCUDAStream()>>>(output.numel(),
        input.const_data_ptr<scalar_t>(),rois.const_data_ptr<scalar_t>(),output.mutable_data_ptr<scalar_t>(),
        scalar_t(scale),input.size(1),input.size(2),input.size(3),ph,pw,ratio,aligned);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}
