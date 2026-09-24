#include "sam3/interactive_image.h"
#include "sam3/autocast.h"
#include "sam3/preprocess.h"
#include "sam3/ops.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
at::Tensor postprocess_interactive_masks(const at::Tensor& logits,int64_t height,int64_t width,const InteractiveImageOptions& options) {
  c10::InferenceMode inference;
  TORCH_CHECK(logits.dim()==4 && height>0 && width>0,"interactive masks require [B,C,H,W] and positive output size");
  auto masks=logits.to(at::kFloat);
  // Both component passes use the original logits, matching SAM2Transforms.
  const auto flat=masks.flatten(0,1).unsqueeze(1);
  if (options.max_hole_area>0) {
    const auto [labels,areas]=connected_components((flat<=options.mask_threshold).to(at::kByte));
    const auto holes=((labels>0)&(areas<=options.max_hole_area)).reshape_as(masks);
    masks=at::where(holes,options.mask_threshold+10.,masks);
  }
  if (options.max_sprinkle_area>0) {
    const auto [labels,areas]=connected_components((flat>options.mask_threshold).to(at::kByte));
    const auto sprinkles=((labels>0)&(areas<=options.max_sprinkle_area)).reshape_as(masks);
    masks=at::where(sprinkles,options.mask_threshold-10.,masks);
  }
  return at::upsample_bilinear2d(masks,{height,width},false);
}
InteractiveImageSession::InteractiveImageSession(const WeightStore& store,const std::string& model,at::Device device)
    :encoder_(store,model,device),decoder_(store,model,device),device_(device),model_(model) {
  no_memory_=store.read(model+(model=="sam3"?"/tracker.no_mem_embed":"/tracker.model.interactivity_no_mem_embed"),device);
  device_=no_memory_.device();
}
void InteractiveImageSession::set_image(const at::Tensor& rgb,const VisionEncoder& vision,const std::string& mode) {
  c10::InferenceMode inference;
  TORCH_CHECK(rgb.dim()==3 && rgb.size(0)==3,"session image must be RGB [3,H,W]");
  const std::string head=model_=="sam3"?"sam2_convs":"interactive_convs";
  auto pyramid=vision.forward(preprocess_rgb(rgb.to(device_)),mode,{head},{}).pyramid.at(head);
  if (model_=="sam3") pyramid.pop_back();
  set_features(pyramid,{rgb.size(1)},{rgb.size(2)},mode);
}
void InteractiveImageSession::set_images(const std::vector<at::Tensor>& rgb,const VisionEncoder& vision,const std::string& mode) {
  c10::InferenceMode inference;detail::check_mode(mode);
  TORCH_CHECK(!rgb.empty(),"cannot set an empty image batch");
  std::vector<at::Tensor> normalized;std::vector<int64_t> heights,widths;
  for (const auto& image:rgb) {
    TORCH_CHECK(image.dim()==3 && image.size(0)==3,"session images must be RGB [3,H,W]");
    heights.push_back(image.size(1));widths.push_back(image.size(2));
    normalized.push_back(preprocess_rgb(image.to(device_)).squeeze(0));
  }
  // Preserve source set_image_batch stacking, including a one-image batch.
  const auto input=at::stack(normalized,0);
  const std::string head=model_=="sam3"?"sam2_convs":"interactive_convs";
  const auto output=vision.forward(input,mode,{head},{});
  auto pyramid=output.pyramid.at(head);
  if (model_=="sam3") pyramid.pop_back();
  set_features(pyramid,heights,widths,mode);
}
void InteractiveImageSession::set_features(const std::vector<at::Tensor>& pyramid,const std::vector<int64_t>& heights,
    const std::vector<int64_t>& widths,const std::string& mode,bool projected_high) {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(pyramid.size()==3 && !heights.empty() && widths.size()==heights.size(),"expected three feature levels and one original size per image");
  const auto batch=static_cast<int64_t>(heights.size());
  for (int level=0;level<3;++level) {
    const auto size=288>>level,channels=projected_high && level<2?(level==0?32:64):256;
    TORCH_CHECK(pyramid[level].sizes()==at::IntArrayRef({batch,channels,size,size}) && pyramid[level].device()==device_,"invalid interactive image pyramid");
  }
  for (int64_t i=0;i<batch;++i) TORCH_CHECK(heights[i]>0 && widths[i]>0,"invalid original image size");
  auto high=projected_high?std::vector<at::Tensor>{pyramid[0],pyramid[1]}:decoder_.project_pyramid({pyramid[0],pyramid[1]},mode);
  auto image=(pyramid[2].flatten(2).permute({2,0,1})+no_memory_).permute({1,2,0}).view({batch,256,72,72});
  high_=std::move(high);image_=std::move(image);heights_=heights;widths_=widths;mode_=mode;
}
InteractiveImageResult InteractiveImageSession::predict(int64_t index,const InteractiveImagePrompt& request,const InteractiveImageOptions& options) const {
  c10::InferenceMode inference;
  TORCH_CHECK(index>=0 && index<image_count(),"set an image before predicting; image index is out of range");
  AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  const auto transform=[&](const at::Tensor& input) {
    auto coords=input.to(device_,at::kFloat).clone();
    if (request.pixel_coordinates) {
      coords.select(-1,0).copy_(coords.select(-1,0)/widths_[index]);
      coords.select(-1,1).copy_(coords.select(-1,1)/heights_[index]);
    }
    return coords*1008;
  };
  at::Tensor points,labels,masks;
  if (request.points.defined()) {
    TORCH_CHECK(request.labels.defined(),"point labels are required");
    TORCH_CHECK((request.points.dim()==2 || request.points.dim()==3) && request.points.size(-1)==2,"points must be [N,2] or [B,N,2]");
    points=transform(request.points);labels=request.labels.to(device_,at::kInt);
    if (points.dim()==2) { points=points.unsqueeze(0);labels=labels.unsqueeze(0); }
    TORCH_CHECK(labels.sizes()==points.sizes().slice(0,2),"point and label shape mismatch");
  } else TORCH_CHECK(!request.labels.defined(),"point labels require coordinates");
  if (request.boxes.defined()) {
    TORCH_CHECK(request.boxes.numel()>0 && request.boxes.size(-1)==4,"boxes must be [4] or [B,4]");
    const auto corners=transform(request.boxes.reshape({-1,2,2}));
    const auto corner_labels=at::tensor({2,3},at::TensorOptions().device(device_).dtype(at::kInt)).reshape({1,2}).repeat({corners.size(0),1});
    points=points.defined()?at::cat({corners,points},1):corners;
    labels=labels.defined()?at::cat({corner_labels,labels},1):corner_labels;
  }
  if (request.masks.defined()) {
    masks=request.masks.to(device_,at::kFloat);
    if (masks.dim()==3) masks=masks.unsqueeze(0);
  }
  const auto prompt=encoder_.forward({points,labels,{},masks},mode_);
  const auto high=std::vector<at::Tensor>{high_[0][index].unsqueeze(0),high_[1][index].unsqueeze(0)};
  const auto output=decoder_.forward(image_[index].unsqueeze(0),prompt,high,options.multimask,prompt.sparse.size(0)>1,mode_);
  auto restored=postprocess_interactive_masks(output.masks,heights_[index],widths_[index],options);
  if (!options.return_logits) restored=restored>options.mask_threshold;
  return {restored,output.iou,output.masks.clamp(-32.,32.)};
}
std::vector<InteractiveImageResult> InteractiveImageSession::predict_batch(const std::vector<InteractiveImagePrompt>& prompts,const InteractiveImageOptions& options) const {
  TORCH_CHECK(image_count()>0 && prompts.size()==heights_.size(),"one prompt request is required per stored image");
  std::vector<InteractiveImageResult> result;
  for (size_t i=0;i<prompts.size();++i) result.push_back(predict(i,prompts[i],options));
  return result;
}
at::Tensor InteractiveImageSession::image_embedding() const {
  TORCH_CHECK(image_count()>0,"set an image before requesting its embedding");return image_;
}
void InteractiveImageSession::reset() { image_=at::Tensor();high_.clear();heights_.clear();widths_.clear();mode_.clear(); }
}
