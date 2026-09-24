#include "sam3/detection_heads.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
#include <cstdlib>
#ifdef SAM3_WITH_CUDA
#include "sam3/pixel_transform.h"
#endif
namespace sam3 {
namespace {
at::Tensor inverse_sigmoid(const at::Tensor& value) {
  const auto x=value.clamp(0,1);
  return (x.clamp_min(1e-3)/(1-x).clamp_min(1e-3)).log();
}
at::Tensor xyxy(const at::Tensor& boxes) {
  const auto cx=boxes.select(-1,0),cy=boxes.select(-1,1),w=boxes.select(-1,2),h=boxes.select(-1,3);
  return at::stack({cx-.5*w,cy-.5*h,cx+.5*w,cy+.5*h},-1);
}
}
DetectionHeads::DetectionHeads(const WeightStore& store,const std::string& model,at::Device device):device_(device) {
  for (const auto& module : {std::make_pair("segmentation_head.","seg."),std::make_pair("dot_prod_scoring.","score."),
                             std::make_pair("transformer.decoder.bbox_embed.","box.")}) {
    for (auto& [name,value] : detail::load(store,model+"/detector."+module.first,device))
      weights_.emplace(module.second+name,std::move(value));
  }
  device_=detail::weight(weights_,"seg.instance_seg_head.weight").device();
}
DetectionOutput DetectionHeads::forward(const std::vector<at::Tensor>& pyramid,const at::Tensor& image_ids,
    const FusionFeatures& encoded,const at::Tensor& prompt_padding,const DecoderFeatures& decoded,
    bool joint_scores,const std::string& mode) const {
  detail::ProfileRange range("detector.heads");
  c10::InferenceMode inference;
  detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16" ? at::kHalf : at::kBFloat16);
  TORCH_CHECK(!pyramid.empty() && pyramid.size()<=4,"segmentation requires 1 to 4 feature levels");
  TORCH_CHECK(encoded.memory.dim()==3 && encoded.memory.size(2)==256 && encoded.memory.device()==device_,"invalid segmentation memory");
  const auto batch=encoded.memory.size(1);
  TORCH_CHECK(decoded.hidden.sizes()==at::IntArrayRef({6,200,batch,256}) && decoded.hidden.device()==device_,"expected all six layers and 200 query features");
  TORCH_CHECK(decoded.references.sizes()==at::IntArrayRef({6,200,batch,4}) && decoded.references.device()==device_,"invalid decoder anchors");
  TORCH_CHECK(decoded.presence_logits.sizes()==at::IntArrayRef({6,1,batch}) && decoded.presence_logits.device()==device_,"invalid presence logits");
  TORCH_CHECK(encoded.prompt.dim()==3 && encoded.prompt.size(1)==batch && encoded.prompt.size(2)==256 && encoded.prompt.device()==device_,"invalid head prompt");
  TORCH_CHECK(prompt_padding.sizes()==at::IntArrayRef({batch,encoded.prompt.size(0)}) && prompt_padding.scalar_type()==at::kBool && prompt_padding.device()==device_,"invalid head prompt padding");
  TORCH_CHECK(image_ids.dim()==1 && image_ids.size(0)==batch && image_ids.scalar_type()==at::kLong,"image_ids must be int64 [B]");
  TORCH_CHECK(pyramid[0].dim()==4 && pyramid[0].size(0)>0,"empty source image batch");
  const auto sources=pyramid[0].size(0);
  TORCH_CHECK(((image_ids>=0)&(image_ids<sources)).all().item<bool>(),"image ID out of range");
  for (const auto& feature : pyramid)
    TORCH_CHECK(feature.dim()==4 && feature.size(0)==sources && feature.size(1)==256 && feature.size(2)>0 && feature.size(3)>0,"inconsistent feature pyramid");
  const auto h=pyramid.back().size(2),w=pyramid.back().size(3);
  TORCH_CHECK(encoded.memory.size(0)>=h*w,"segmentation memory is shorter than the image grid");
  const auto hs=decoded.hidden.transpose(1,2);
  const auto anchors=decoded.references.transpose(1,2);
  const auto presence=decoded.presence_logits.transpose(1,2);
  auto prompt=detail::norm(weights_,detail::mlp(weights_,encoded.prompt,"score.prompt_mlp",2)+encoded.prompt,"score.prompt_mlp.out_norm");
  const auto valid=prompt_padding.logical_not().to(at::kFloat).transpose(0,1).unsqueeze(-1);
  const auto pooled=(prompt*valid).sum(0)/valid.sum(0).clamp_min(1.);
  auto scores=at::matmul(detail::linear(weights_,hs,"score.hs_proj"),detail::linear(weights_,pooled,"score.prompt_proj").unsqueeze(-1));
  scores.mul_(1./16).clamp_(-12,12);
  if (joint_scores) scores=inverse_sigmoid(scores.sigmoid()*presence.clone().sigmoid().unsqueeze(2)).clamp(-10,10);
  const auto boxes=(inverse_sigmoid(anchors)+detail::mlp(weights_,hs,"box",3)).sigmoid();
  const auto attended=detail::attention(weights_,detail::norm(weights_,encoded.memory,"seg.cross_attn_norm"),
      encoded.prompt,encoded.prompt,prompt_padding,"seg.cross_attend_prompt",detail::Projection::SharedKV);
  const auto memory=attended+encoded.memory;
  auto pixel=memory.permute({1,2,0}).slice(2,0,h*w).reshape({batch,256,h,w});
  const auto* experiment_env=std::getenv("SAM3_EXPERIMENT_PIXEL");
  const std::string experiment=experiment_env?experiment_env:"exact";
  TORCH_CHECK(experiment=="exact" || experiment=="borrow" || experiment=="borrow_relu" || experiment=="nchw" || experiment=="fused_nchw","invalid pixel experiment: ",experiment);
  for (int64_t level=static_cast<int64_t>(pyramid.size())-2,layer=0;level>=0;--level,++layer) {
    const auto range_name="detector.heads.pixel."+std::to_string(layer);
    const auto& feature=pyramid[level];
    const auto source=detail::profile_call(range_name+".source",[&] {
      if(sources>1)return feature.index({image_ids.to(feature.device())}).to(device_);
      return experiment=="exact"?feature.clone().to(device_):feature.to(device_);
    });
    auto upsampled=detail::profile_call(range_name+".upsample",[&] {return at::upsample_nearest2d(pixel,{source.size(2),source.size(3)});});
    pixel=detail::profile_call(range_name+".add",[&] {return source+upsampled;});
    upsampled=at::Tensor();
    const auto conv="seg.pixel_decoder.conv_layers."+std::to_string(layer);
    pixel=detail::profile_call(range_name+".conv",[&] {return at::conv2d(pixel,detail::weight(weights_,conv+".weight"),detail::weight(weights_,conv+".bias"),{1,1},{1,1});});
    const auto norm="seg.pixel_decoder.norms."+std::to_string(layer);
    if(experiment=="nchw" || experiment=="fused_nchw") {
      pixel=detail::profile_call(range_name+".pre_norm",[&] {
        TORCH_CHECK(pixel.is_cuda() && pixel.scalar_type()==at::kHalf,"pixel layout experiment requires CUDA FP16 convolution output");
#ifdef SAM3_WITH_CUDA
        if(experiment=="fused_nchw")return pixel_nchw_float(pixel);
#endif
        return pixel.contiguous().to(at::kFloat);
      });
    }
    auto normalized=detail::profile_call(range_name+".group_norm",[&] {return at::group_norm(pixel,8,detail::weight(weights_,norm+".weight"),detail::weight(weights_,norm+".bias"),1e-5);});
    pixel=detail::profile_call(range_name+".relu",[&] {return experiment=="borrow_relu"?at::relu_(normalized):at::relu(normalized);});
  }
  const auto instances=detail::profile_call("detector.heads.instances",[&] {return at::conv2d(pixel,detail::weight(weights_,"seg.instance_seg_head.weight"),detail::weight(weights_,"seg.instance_seg_head.bias"));});
  const auto queries=hs[-1];
  const auto mask_queries=detail::mlp(weights_,queries,"seg.mask_predictor.mask_embed",3);
  const auto masks=detail::profile_call("detector.heads.masks",[&] {return at::einsum("bqc,bchw->bqhw",{mask_queries,instances});});
  const auto semantic=detail::profile_call("detector.heads.semantic",[&] {return at::conv2d(pixel,detail::weight(weights_,"seg.semantic_seg_head.weight"),detail::weight(weights_,"seg.semantic_seg_head.bias"));});
  return {scores[-1],boxes[-1],xyxy(boxes)[-1],masks,semantic,queries,presence[-1],decoded.presence};
}
}
