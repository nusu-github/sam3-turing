#include "sam3/memory_encoder.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
namespace {
at::Tensor norm2d(const detail::Weights& weights,const at::Tensor& input,const std::string& name) {
  const auto mean=input.mean(1,true),variance=(input-mean).pow(2).mean(1,true);
  const auto normalized=(input-mean)/(variance+1e-6).sqrt();
  return detail::weight(weights,name+".weight").unsqueeze(-1).unsqueeze(-1)*normalized+
    detail::weight(weights,name+".bias").unsqueeze(-1).unsqueeze(-1);
}
at::Tensor position(const at::Tensor& x) {
  const auto b=x.size(0),h=x.size(2),w=x.size(3),dim=x.size(1)/2;
  const auto options=x.options().dtype(at::kFloat);
  auto y=at::arange(1,h+1,options).view({1,h,1}).repeat({b,1,w});
  auto xx=at::arange(1,w+1,options).view({1,1,w}).repeat({b,h,1});
  y=y/(y.slice(1,h-1,h)+1e-6)*(2*std::acos(-1.0));
  xx=xx/(xx.slice(2,w-1,w)+1e-6)*(2*std::acos(-1.0));
  auto frequencies=at::arange(dim,options);
  frequencies=at::pow(10000.,2*frequencies.floor_divide(2)/dim);
  const auto encode=[&](const at::Tensor& coordinate) {
    const auto angle=coordinate.unsqueeze(-1)/frequencies;
    return at::stack({angle.slice(3,0,dim,2).sin(),angle.slice(3,1,dim,2).cos()},4).flatten(3);
  };
  return at::cat({encode(y),encode(xx)},3).permute({0,3,1,2}).to(x.scalar_type());
}
}
MaskMemoryEncoder::MaskMemoryEncoder(const WeightStore& store,const std::string& model,at::Device device)
    :device_(device),multiplex_(model=="sam3.1") {
  TORCH_CHECK(model=="sam3" || multiplex_,"unknown memory model");
  const auto root=model+(multiplex_?"/tracker.model.":"/tracker.");
  weights_=detail::load(store,root+"maskmem_backbone.",device);
  absent_embedding_=store.read(root+"no_obj_embed_spatial",device);
  device_=absent_embedding_.device();
  c10::InferenceMode inference;
  // The source builder precomputes this fixed grid in float32. Keep it per
  // loaded module; repeat before casting to preserve source layout and dtype.
  position_=position(at::empty({1,multiplex_?256:64,72,72},absent_embedding_.options().dtype(at::kFloat))).contiguous();
}
MaskMemoryOutput MaskMemoryEncoder::forward(const at::Tensor& image,const at::Tensor& masks,bool skip_mask_sigmoid,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(masks.dim()==4 && masks.size(0)>0 && masks.size(1)==(multiplex_?32:1) && masks.size(2)>0 && masks.size(3)>0 && masks.device()==device_,"invalid mask-memory input channels or device");
  TORCH_CHECK(image.dim()==4 && (image.size(0)==1 || image.size(0)==masks.size(0)) && image.size(1)==256 && image.size(2)==72 && image.size(3)==72,"memory image features must be [1|B,256,72,72]");
  auto encoded=skip_mask_sigmoid?masks:masks.sigmoid();
  if (encoded.size(2)!=1152 || encoded.size(3)!=1152)
    encoded=at::_upsample_bilinear2d_aa(encoded.to(at::kFloat),{1152,1152},false);
  for (int layer=0;layer<4;++layer) {
    const auto name="mask_downsampler.encoder."+std::to_string(layer*3);
    encoded=at::conv2d(encoded,detail::weight(weights_,name+".weight"),detail::weight(weights_,name+".bias"),{2,2},{1,1});
    encoded=at::gelu(norm2d(weights_,encoded,"mask_downsampler.encoder."+std::to_string(layer*3+1)));
  }
  const auto conv=[&](const at::Tensor& x,const std::string& name) {
    return at::conv2d(x,detail::weight(weights_,name+".weight"),detail::weight(weights_,name+".bias"));
  };
  encoded=conv(encoded,"mask_downsampler.encoder.12");
  auto x=conv(image.to(device_),"pix_feat_proj")+encoded;
  for (int layer=0;layer<2;++layer) {
    const auto name="fuser.layers."+std::to_string(layer);
    auto branch=at::conv2d(x,detail::weight(weights_,name+".dwconv.weight"),detail::weight(weights_,name+".dwconv.bias"),{1,1},{3,3},{1,1},256);
    branch=norm2d(weights_,branch,name+".norm").permute({0,2,3,1});
    branch=detail::linear(weights_,at::gelu(detail::linear(weights_,branch,name+".pwconv1")),name+".pwconv2");
    branch=detail::weight(weights_,name+".gamma")*branch;
    x=x+branch.permute({0,3,1,2});
  }
  if (!multiplex_) x=conv(x,"out_proj");
  return {x,position_.repeat({x.size(0),1,1,1}).to(x.scalar_type())};
}
MaskMemoryOutput MaskMemoryEncoder::encode_frame(const at::Tensor& image,const at::Tensor& masks,
    const at::Tensor& object_logits,const MemoryFrameOptions& options,const at::Tensor& matrix,
    const at::Tensor& conditioning_objects,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(masks.dim()==4 && masks.size(0)>0 && masks.size(1)==1 && masks.device()==device_,"frame masks require [objects,1,H,W]");
  auto input=masks;
  if (options.non_overlap && masks.size(0)>1) {
    const auto winner=masks.argmax(0,true),ids=at::arange(masks.size(0),masks.options().dtype(at::kLong)).view({-1,1,1,1});
    input=at::where(winner==ids,masks,masks.clamp_max(-10.));
  }
  auto transformed=(!multiplex_ && options.from_points)?(input>0.).to(at::kFloat):input.sigmoid();
  transformed=transformed*(multiplex_?2.:20.);
  transformed=transformed+(multiplex_?-1.:-10.);
  if (!multiplex_) {
    TORCH_CHECK(options.object_threshold==0.,"SAM3 memory object threshold is zero");
    TORCH_CHECK(object_logits.sizes()==at::IntArrayRef({masks.size(0),1}) && object_logits.device()==device_,"invalid memory object scores");
    auto out=forward(image.view_as(image),transformed,true,mode);
    const auto present=(object_logits>0.).to(at::kFloat);
    out.features+=(1-present.unsqueeze(-1).unsqueeze(-1))*absent_embedding_.unsqueeze(-1).unsqueeze(-1).expand_as(out.features);
    return out;
  }
  TORCH_CHECK(matrix.defined() && matrix.dim()==2 && matrix.size(0)>0 && matrix.size(0)%16==0 && matrix.size(1)==masks.size(0) && matrix.device()==device_,"multiplex matrix requires [buckets*16,objects]");
  const auto buckets=matrix.size(0)/16;
  const auto mux=[&](const at::Tensor& x) {
    auto shape=x.sizes().vec();shape.erase(shape.begin());shape.insert(shape.begin(),{buckets,16});
    return at::matmul(matrix,x.reshape({masks.size(0),-1})).view(shape);
  };
  auto packed=mux(transformed).squeeze(2);
  auto conditions=at::zeros({masks.size(0)},transformed.options());
  if (conditioning_objects.defined()) {
    TORCH_CHECK(conditioning_objects.dim()==1 && conditioning_objects.scalar_type()==at::kLong,"conditioning object indices must be int64 [N]");
    conditions.index_fill_(0,conditioning_objects.to(device_),1.);
  }
  const auto embedded=conditions.view({-1,1,1,1}).expand_as(transformed);
  packed=at::cat({packed,mux(embedded).squeeze(2)},1);
  auto out=forward(image,packed,true,mode);
  TORCH_CHECK(object_logits.dim()==2 && object_logits.size(1)==1 && object_logits.device()==device_,"invalid multiplex object scores");
  auto scores=object_logits;
  if (scores.size(0)<masks.size(0)) scores=at::cat({scores,at::zeros({masks.size(0)-scores.size(0),1},scores.options())},0);
  else if (scores.size(0)>masks.size(0)) scores=scores.slice(0,0,masks.size(0));
  const auto present=(mux(scores)>options.object_threshold).to(at::kFloat);
  const auto absent=absent_embedding_.unsqueeze(0).repeat({buckets,1,1});
  const auto embedding=((1-present)*absent).sum(1);
  out.features+=embedding.unsqueeze(-1).unsqueeze(-1).expand_as(out.features);
  return out;
}
}
