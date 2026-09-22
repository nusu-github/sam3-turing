#include "sam3/grounding.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
namespace sam3 {
GroundingDetector::GroundingDetector(const WeightStore& store,const std::string& model,at::Device device)
    :geometry_(store,model,device),encoder_(store,model,device),decoder_(store,model,device),heads_(store,model,device),device_(device) {}
GroundingOutput GroundingDetector::forward(const std::vector<at::Tensor>& pyramid,const at::Tensor& positions,
    const GroundingPrompt& request,bool joint_scores,const std::string& mode) const {
  c10::InferenceMode inference;
  detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16" ? at::kHalf : at::kBFloat16);
  TORCH_CHECK(!pyramid.empty() && pyramid.back().dim()==4,"grounding requires a feature pyramid");
  TORCH_CHECK(request.image_ids.dim()==1 && request.image_ids.scalar_type()==at::kLong && request.image_ids.numel()>0,"grounding image IDs must be int64 [B]");
  const auto batch=request.image_ids.numel();
  const auto& low=pyramid.back();
  TORCH_CHECK(positions.sizes()==low.sizes(),"grounding positional feature shape mismatch");
  TORCH_CHECK(((request.image_ids>=0)&(request.image_ids<low.size(0))).all().item<bool>(),"grounding image ID out of range");
  const auto image=low.index({request.image_ids.to(low.device())}).to(device_);
  const auto pos=positions.index({request.image_ids.to(positions.device())}).to(device_);
  auto geometry_image=image;
  if (request.previous_mask.defined()) {
    TORCH_CHECK(request.previous_mask.sizes()==at::IntArrayRef({image.size(2)*image.size(3),batch,256}),"previous mask features must be [H*W,B,256]");
    geometry_image=(image.flatten(2).permute({2,0,1})+request.previous_mask.to(device_)).permute({1,2,0}).reshape_as(image);
  }
  const auto [geo,geo_padding]=geometry_.forward(geometry_image,pos,request.geometry,mode);
  std::vector<at::Tensor> features,paddings;
  if (request.use_text) {
    TORCH_CHECK(request.text_features.dim()==3 && request.text_features.size(2)==256,"text features must be [L,Ntext,256]");
    TORCH_CHECK(request.text_ids.sizes()==at::IntArrayRef({batch}) && request.text_ids.scalar_type()==at::kLong,"text IDs must be int64 [B]");
    TORCH_CHECK(request.text_padding.sizes()==at::IntArrayRef({request.text_features.size(1),request.text_features.size(0)}) && request.text_padding.scalar_type()==at::kBool,"invalid text padding");
    TORCH_CHECK(((request.text_ids>=0)&(request.text_ids<request.text_features.size(1))).all().item<bool>(),"text ID out of range");
    features.push_back(request.text_features.index({at::indexing::Slice(),request.text_ids.to(request.text_features.device())}).to(device_));
    paddings.push_back(request.text_padding.index({request.text_ids.to(request.text_padding.device())}).to(device_));
  }
  features.push_back(geo);paddings.push_back(geo_padding);
  if (request.visual_features.defined()) {
    TORCH_CHECK(request.visual_features.dim()==3 && request.visual_features.size(1)==batch && request.visual_features.size(2)==256,"visual prompts must be [Nvisual,B,256]");
    TORCH_CHECK(request.visual_padding.defined() && request.visual_padding.sizes()==at::IntArrayRef({batch,request.visual_features.size(0)}) && request.visual_padding.scalar_type()==at::kBool,"invalid visual prompt padding");
    features.push_back(request.visual_features.to(device_));paddings.push_back(request.visual_padding.to(device_));
  } else {
    features.push_back(at::zeros({0,batch,256},geo.options().dtype(at::kFloat)));
    paddings.push_back(at::zeros({batch,0},geo_padding.options()));
  }
  const auto prompt=at::cat(features,0),padding=at::cat(paddings,1);
  auto encoded=encoder_.forward(image,pos,prompt,padding,{},mode);
  auto decoded=decoder_.forward(encoded,padding,mode);
  auto detection=heads_.forward(pyramid,request.image_ids,encoded,padding,decoded,joint_scores,mode);
  return {std::move(detection),std::move(encoded),std::move(decoded)};
}
}
