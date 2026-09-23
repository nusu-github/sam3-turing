#include "sam3/video_heads.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
namespace sam3 {
VideoInteractiveHeads::VideoInteractiveHeads(const WeightStore& store,const std::string& model,at::Device device)
    :encoder_(store,model,device),decoder_(store,model,device),device_(device),multiplex_(model=="sam3.1") {
  const auto root=model+(multiplex_?"/tracker.model.":"/tracker.");
  for (const auto& name: {std::string("obj_ptr_proj"),std::string("mask_downsample")}) {
    for (auto& [key,value]:detail::load(store,root+(multiplex_?"interactive_":"")+name+".",device))
      weights_.emplace(name+"."+key,std::move(value));
  }
  if (multiplex_) {
    for (auto& [key,value]:detail::load(store,root+"no_obj_ptr_linear.",device))
      weights_.emplace("no_obj_ptr_linear."+key,std::move(value));
  } else weights_.emplace("no_obj_ptr",store.read(root+"no_obj_ptr",device));
  device_=detail::weight(weights_,"obj_ptr_proj.layers.0.weight").device();
}
std::vector<at::Tensor> VideoInteractiveHeads::project_pyramid(const std::vector<at::Tensor>& features,const std::string& mode) const {
  return decoder_.project_pyramid(features,mode);
}
at::Tensor VideoInteractiveHeads::project_no_object_pointer(const at::Tensor& pointer,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);TORCH_CHECK(multiplex_,"no-object linear projection requires SAM3.1");
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  return detail::linear(weights_,pointer,"no_obj_ptr_linear");
}
at::Tensor VideoInteractiveHeads::gate_pointer(const at::Tensor& pointer,const at::Tensor& present) const {
  const auto probability=present.to(at::kFloat);
  if (multiplex_) return probability*pointer+(1-probability)*detail::linear(weights_,pointer,"no_obj_ptr_linear");
  return probability*pointer+(1-probability)*detail::weight(weights_,"no_obj_ptr");
}
VideoMaskOutput VideoInteractiveHeads::forward(const at::Tensor& image,const std::vector<at::Tensor>& high,
    const at::Tensor& points,const at::Tensor& labels,const at::Tensor& masks,bool multimask,const std::string& mode,double object_threshold,bool attenuate) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(image.dim()==4 && image.size(1)==256 && image.size(2)==72 && image.size(3)==72 && image.device()==device_,"video image features must be [B,256,72,72]");
  TORCH_CHECK(!multiplex_ || image.size(0)==1,"SAM3.1 interactive heads repeat one image");
  TORCH_CHECK(!multiplex_ || points.defined() || masks.defined(),"SAM3.1 without prompts requires the separate multiplex propagation head");
  TORCH_CHECK(multiplex_ || object_threshold==0.,"SAM3 object threshold is fixed at zero");
  const auto batch=multiplex_?(points.defined()?points.size(0):masks.size(0)):image.size(0);
  auto coords=points,point_labels=labels,mask_prompt=masks;
  if (coords.defined()) {
    TORCH_CHECK(coords.dim()==3 && coords.size(0)==batch && coords.size(2)==2 && point_labels.defined(),"video points require [B,N,2] coordinates and labels");
  } else {
    TORCH_CHECK(!labels.defined(),"labels require points");
    coords=at::zeros({batch,1,2},image.options().dtype(at::kFloat));
    point_labels=-at::ones({batch,1},image.options().dtype(at::kInt));
  }
  if (masks.defined()) {
    TORCH_CHECK(masks.dim()==4 && masks.size(0)==batch && masks.size(1)==1 && masks.device()==device_,"video masks require [B,1,H,W]");
    if (masks.size(2)!=288 || masks.size(3)!=288)
      mask_prompt=at::_upsample_bilinear2d_aa(masks.to(at::kFloat),{288,288},false);
  }
  const auto prompt=encoder_.forward({coords,point_labels,{},mask_prompt},mode);
  const auto output=decoder_.forward(image,prompt,high,multimask,multiplex_,mode);
  const auto present=output.object_logits>object_threshold;
  const auto low=at::where(present.unsqueeze(-1).unsqueeze(-1),output.masks,-1024.).to(at::kFloat);
  const auto full=at::upsample_bilinear2d(low,{1008,1008},false);
  auto iou=output.iou;
  if(multimask && attenuate) {
    TORCH_CHECK(multiplex_,"stability attenuation is a SAM3.1 policy");
    const auto flat=low.flatten(2),intersection=(flat>.05).sum(-1),union_area=(flat>-.05).sum(-1);
    const auto stability=at::where(union_area>0,intersection.to(at::kFloat)/union_area,at::ones_like(intersection,at::kFloat));
    iou=iou*stability;
  }
  auto selected_low=low,selected_full=full,token=output.tokens.select(1,0);
  if (multimask) {
    const auto best=iou.argmax(-1),ids=at::arange(batch,best.options());
    selected_low=low.index({ids,best}).unsqueeze(1);selected_full=full.index({ids,best}).unsqueeze(1);
    if (output.tokens.size(1)>1) token=output.tokens.index({ids,best});
  }
  const auto pointer=gate_pointer(detail::mlp(weights_,token,"obj_ptr_proj",3),present);
  return {low,full,iou,selected_low,selected_full,pointer,output.object_logits};
}
VideoMaskOutput VideoInteractiveHeads::use_mask_as_output(const at::Tensor& image,const std::vector<at::Tensor>& high,
    const at::Tensor& mask,const std::string& mode,double object_threshold) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(mask.dim()==4 && mask.size(0)>0 && mask.size(1)==1 && mask.size(2)>=4 && mask.size(3)>=4 && mask.device()==device_,"direct video mask must be [B,1,H,W]");
  const auto value=mask.to(multiplex_?image.scalar_type():at::kFloat);
  const auto full=value*20.-10.;
  // Preserve source SAM3's stride-based size, including non-default mask sizes.
  const auto h=multiplex_?full.size(2)/4:full.size(2)/14*4,w=multiplex_?full.size(3)/4:full.size(3)/14*4;
  const auto low=at::_upsample_bilinear2d_aa(full,{h,w},false);
  const auto iou=at::ones({mask.size(0),1},mask.options().dtype(multiplex_?image.scalar_type():at::kFloat));
  const auto prompt=at::conv2d(value,detail::weight(weights_,"mask_downsample.weight"),detail::weight(weights_,"mask_downsample.bias"),{4,4});
  const auto decoded=forward(image,high,{},{},prompt,false,mode,object_threshold);
  const auto present=(mask.flatten(1).to(at::kFloat)>0.).any(1).unsqueeze(-1);
  const auto object_logits=20.*present.to(at::kFloat)-10.;
  // Source applies absent-object handling a second time to direct-mask output.
  const auto pointer=gate_pointer(decoded.object_pointer,present);
  return {low,full,iou,low,full,pointer,object_logits};
}
}
