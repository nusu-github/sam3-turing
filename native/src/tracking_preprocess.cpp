// Bicubic coefficient/rounding behavior follows Pillow 12.2 libImaging/Resample.c.
// Copyright/permission notice: native/third_party/pillow/LICENSE.txt.
#include "sam3/preprocess.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <cmath>
#include <limits>
namespace sam3 {
namespace {
struct Kernel {int64_t first;std::vector<int32_t> weights;};
double cubic(double x) {
  x=std::abs(x);
  if(x<1.)return (1.5*x-2.5)*x*x+1.;
  if(x<2.)return (((x-5.)*x+8.)*x-4.)*(-.5);
  return 0.;
}
std::vector<Kernel> kernels(int64_t input,int64_t output) {
  const double scale=double(float(input))/output,filter_scale=std::max(1.,scale),support=2*filter_scale,reciprocal=1./filter_scale;
  std::vector<Kernel> result;result.reserve(output);
  for(int64_t i=0;i<output;++i) {
    const auto center=(i+.5)*scale;
    const auto first=std::max<int64_t>(0,int64_t(center-support+.5));
    const auto last=std::min<int64_t>(input,int64_t(center+support+.5));
    std::vector<double> values;double sum=0.;
    for(auto j=first;j<last;++j){const auto value=cubic((j-center+.5)*reciprocal);values.push_back(value);sum+=value;}
    Kernel kernel;kernel.first=first;
    for(auto value:values){if(sum!=0.)value/=sum;kernel.weights.push_back(int32_t(value*(1<<22)+(value<0?-.5:.5)));}
    result.push_back(std::move(kernel));
  }
  return result;
}
uint8_t quantize(int64_t sum) {return sum<=0?0:sum>=(int64_t(255)<<22)?255:uint8_t(sum>>22);}
}
at::Tensor resize_tracking_rgb(const at::Tensor& pixels,int64_t height,int64_t width) {
  c10::InferenceMode inference;
  TORCH_CHECK(pixels.dim()==3 && pixels.size(0)==3 && pixels.scalar_type()==at::kByte && pixels.size(1)>0 && pixels.size(2)>0,"tracking pixels require RGB uint8 [3,H,W]");
  TORCH_CHECK(height>0 && width>0 && height<=std::numeric_limits<int>::max() && width<=std::numeric_limits<int>::max() && pixels.size(1)<=std::numeric_limits<int>::max() && pixels.size(2)<=std::numeric_limits<int>::max(),"invalid resize dimensions");
  auto input=pixels.cpu().contiguous();const auto old_h=input.size(1),old_w=input.size(2);
  if(old_w!=width) {
    const auto weights=kernels(old_w,width);auto output=at::empty({3,old_h,width},input.options());
    const auto* src=input.const_data_ptr<uint8_t>();auto* dst=output.mutable_data_ptr<uint8_t>();
    at::parallel_for(0,3*old_h,16,[&](int64_t begin,int64_t end){
      for(auto row=begin;row<end;++row) for(int64_t x=0;x<width;++x) {
        const auto& k=weights[x];int64_t sum=1<<21;
        for(size_t j=0;j<k.weights.size();++j)sum+=int64_t(src[row*old_w+k.first+j])*k.weights[j];
        dst[row*width+x]=quantize(sum);
      }
    });input=std::move(output);
  }
  if(old_h!=height) {
    const auto weights=kernels(old_h,height);auto output=at::empty({3,height,width},input.options());
    const auto* src=input.const_data_ptr<uint8_t>();auto* dst=output.mutable_data_ptr<uint8_t>();
    at::parallel_for(0,3*height,16,[&](int64_t begin,int64_t end){
      for(auto row=begin;row<end;++row) {
        const auto channel=row/height,y=row%height;const auto& k=weights[y];
        for(int64_t x=0;x<width;++x) {
          int64_t sum=1<<21;for(size_t j=0;j<k.weights.size();++j)sum+=int64_t(src[(channel*old_h+k.first+j)*width+x])*k.weights[j];
          dst[row*width+x]=quantize(sum);
        }
      }
    });input=std::move(output);
  }
  return input;
}
at::Tensor preprocess_tracking_rgb(const at::Tensor& pixels) {
  // Division is deliberately not multiplication by a rounded F32 reciprocal.
  return resize_tracking_rgb(pixels,1008,1008).to(at::kFloat).div_(255.).sub_(.5).div_(.5).unsqueeze(0);
}
}
