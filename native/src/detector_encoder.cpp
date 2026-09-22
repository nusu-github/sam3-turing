#include "sam3/detector.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
DetectorEncoder::DetectorEncoder(const WeightStore& store,const std::string& model,at::Device device)
    :weights_(detail::load(store,model+"/detector.transformer.encoder.",device)),device_(device) {
  device_=detail::weight(weights_,"layers.0.linear1.weight").device();
  for (int i=0;i<6;++i) detail::weight(weights_,"layers."+std::to_string(i)+".self_attn.in_proj_weight");
}
FusionFeatures DetectorEncoder::forward(const at::Tensor& image,const at::Tensor& positions,
    const at::Tensor& prompt,const at::Tensor& prompt_padding,const at::Tensor& image_padding,const std::string& mode) const {
  c10::InferenceMode inference;
  detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16" ? at::kHalf : at::kBFloat16);
  TORCH_CHECK(image.dim()==4 && image.size(0)>0 && image.size(1)==256 && image.size(2)>0 && image.size(3)>0,
              "detector image must be [B,256,H,W]");
  const auto batch=image.size(0),h=image.size(2),w=image.size(3);
  TORCH_CHECK(image.device()==device_ && positions.device()==device_ && positions.sizes()==image.sizes(),"detector feature/position mismatch");
  TORCH_CHECK(prompt.dim()==3 && prompt.size(1)==batch && prompt.size(2)==256 && prompt.device()==device_,"detector prompt must be [L,B,256]");
  TORCH_CHECK(prompt_padding.sizes()==at::IntArrayRef({batch,prompt.size(0)}) && prompt_padding.scalar_type()==at::kBool && prompt_padding.device()==device_,
              "detector prompt padding must be bool [B,L]");
  at::Tensor mask;
  auto ratios=at::ones({batch,1,2},image.options().dtype(at::kFloat));
  if (image_padding.defined()) {
    TORCH_CHECK(image_padding.sizes()==at::IntArrayRef({batch,h,w}) && image_padding.scalar_type()==at::kBool && image_padding.device()==device_,
                "detector image padding must be bool [B,H,W]");
    mask=image_padding.flatten(1).contiguous();
    const auto rh=image_padding.select(2,0).logical_not().sum(1).to(at::kFloat)/h;
    const auto rw=image_padding.select(1,0).logical_not().sum(1).to(at::kFloat)/w;
    ratios=at::stack({rw,rh},-1).unsqueeze(1);
  }
  auto x=image.flatten(2).transpose(1,2).contiguous();
  const auto pos=positions.flatten(2).transpose(1,2).contiguous();
  for (int i=0;i<6;++i) {
    const auto name="layers."+std::to_string(i);
    const auto normalized=detail::norm(weights_,x,name+".norm1");
    const auto q=(normalized+pos).transpose(0,1);
    x=x+detail::attention(weights_,q,q,normalized.transpose(0,1),mask,name+".self_attn",detail::Projection::Separate).transpose(0,1);
    x=x+detail::attention(weights_,detail::norm(weights_,x,name+".norm2").transpose(0,1),prompt,prompt,prompt_padding,
        name+".cross_attn_image",detail::Projection::SharedKV).transpose(0,1);
    x=x+detail::linear(weights_,at::relu(detail::linear(weights_,detail::norm(weights_,x,name+".norm3"),name+".linear1")),name+".linear2");
  }
  const auto longs=image.options().dtype(at::kLong);
  return {x.transpose(0,1),mask.defined()?mask.transpose(0,1):at::Tensor(),pos.transpose(0,1),prompt,
          at::zeros({1},longs),at::tensor({h,w},longs).view({1,2}),ratios};
}
}
