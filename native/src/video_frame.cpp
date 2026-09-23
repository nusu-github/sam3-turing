#include "sam3/video_frame.h"
#include "sam3/preprocess.h"
#include "sam3/autocast.h"
#include "sam3/ops.h"
#include <c10/core/InferenceMode.h>
#include <cmath>
namespace sam3 {
VideoFrameEncoder::VideoFrameEncoder(std::shared_ptr<const VisionEncoder> v,std::shared_ptr<const GroundingDetector> d,std::shared_ptr<const Sam3TrackingFrame> c,at::Device device):vision_(std::move(v)),detector_(std::move(d)),sam3_(std::move(c)),device_(device){TORCH_CHECK(vision_ && detector_ && sam3_,"frame encoder requires shared vision/detector/tracker modules");}
VideoFrameEncoder::VideoFrameEncoder(std::shared_ptr<const VisionEncoder> v,std::shared_ptr<const GroundingDetector> d,std::shared_ptr<const Sam31TrackingFrame> c,at::Device device):vision_(std::move(v)),detector_(std::move(d)),sam31_(std::move(c)),device_(device){TORCH_CHECK(vision_ && detector_ && sam31_,"frame encoder requires shared vision/detector/tracker modules");}
VideoFrameFeatures VideoFrameEncoder::encode_rgb(const at::Tensor& rgb,const std::string& mode)const{return encode_preprocessed(preprocess_video_rgb(rgb),mode);}
VideoFrameFeatures VideoFrameEncoder::encode_preprocessed(const at::Tensor& input,const std::string& mode)const{
  c10::InferenceMode inference;TORCH_CHECK(input.dim()==4 && input.size(0)>0 && input.sizes().slice(1)==at::IntArrayRef({3,1008,1008}) && input.scalar_type()==at::kFloat,"video image requires normalized F32 [B,3,1008,1008]");
  auto visual=vision_->forward(input.to(device_),mode,sam31_?std::vector<std::string>{"convs","interactive_convs","propagation_convs"}:std::vector<std::string>{"convs","sam2_convs"},{2});
  VideoFrameFeatures out;out.detection_pyramid=std::move(visual.pyramid.at("convs"));if(sam3_)out.detection_pyramid.pop_back();out.detection_position=visual.positions.at(2);
  if(sam3_){const auto& p=visual.pyramid.at("sam2_convs");out.tracking.interactive={p.at(2),visual.positions.at(2),sam3_->project_pyramid({p.at(0),p.at(1)},mode)};out.tracking.propagation=out.tracking.interactive;}
  else{const auto& i=visual.pyramid.at("interactive_convs");const auto& p=visual.pyramid.at("propagation_convs");out.tracking.interactive={i.at(2),visual.positions.at(2),sam31_->project_interactive({i.at(0),i.at(1)},mode)};out.tracking.propagation={p.at(2),visual.positions.at(2),sam31_->project_propagation({p.at(0),p.at(1)},mode)};}
  return out;
}
GroundingOutput VideoFrameEncoder::detect(const VideoFrameFeatures& features,const GroundingPrompt& prompt,const std::string& mode)const{return detector_->forward(features.detection_pyramid,features.detection_position,prompt,true,mode);}
at::Tensor video_mask_nms(const at::Tensor& scores,const at::Tensor& masks,const VideoDetectionOptions& o){
  c10::InferenceMode inference;TORCH_CHECK(scores.dim()==1 && scores.is_floating_point() && masks.dim()==3 && masks.size(0)==scores.size(0) && masks.is_floating_point() && masks.device()==scores.device(),"invalid video NMS inputs");
  TORCH_CHECK(std::isfinite(o.score_threshold) && std::isfinite(o.nms_threshold),"NMS thresholds must be finite");
  TORCH_CHECK(o.nms==VideoNmsMode::Greedy || o.nms==VideoNmsMode::Sam31Perflib || o.nms==VideoNmsMode::Sam31Batched,"invalid video NMS mode");
  TORCH_CHECK(o.policy_mode=="fp32" || o.policy_mode=="fp16" || o.policy_mode=="bf16_reference","invalid video NMS arithmetic mode");
  AutocastGuard autocast(masks.device().type(),o.policy_mode!="fp32",o.policy_mode=="fp16"?at::kHalf:at::kBFloat16);
  auto valid=scores.gt(o.score_threshold);const auto n=scores.size(0);if(!n)return valid;
  const auto rows=o.nms==VideoNmsMode::Greedy?at::nonzero(valid).squeeze(1):at::arange(n,scores.options().dtype(at::kLong));if(!rows.numel())return valid;
  const auto binary=masks.index_select(0,rows).gt(0).flatten(1).to(at::kFloat),selected=scores.index_select(0,rows);
  const auto area=binary.sum(1),intersection=at::mm(binary,binary.t());at::Tensor denominator;
  if(o.use_iom){
    // Original single-frame perflib IoM takes min(area_pred, area_gt) without
    // transposing. With self-overlap this is the row area, not pairwise minimum.
    denominator=o.nms==VideoNmsMode::Sam31Perflib?area.unsqueeze(1):at::minimum(area.unsqueeze(1),area.unsqueeze(0));denominator=denominator+1e-8;
  }else{denominator=area.unsqueeze(1)+area.unsqueeze(0)-intersection;denominator=o.nms==VideoNmsMode::Sam31Batched?denominator+1e-8:denominator.clamp_min(1);}
  const auto overlap=(o.use_iom && o.nms==VideoNmsMode::Greedy?intersection.to(at::kLong):intersection)/denominator;auto keep=at::zeros_like(valid);
  if(o.nms==VideoNmsMode::Greedy){const auto selected_rows=generic_nms(overlap,selected,o.nms_threshold);keep.index_fill_(0,rows.index_select(0,selected_rows),true);}
  else{
    const auto order=at::argsort(selected,0,true),sorted=overlap.index_select(0,order).index_select(1,order);auto retained=valid.index_select(0,order);
    if(o.nms==VideoNmsMode::Sam31Perflib)retained=retained.logical_and(sorted.gt(o.nms_threshold).triu(1).any(0).logical_not());
    else{
      // Feed the source's potentially unstable tie ordering into precompiled
      // greedy NMS using unique ordinal scores; rejected rows cannot suppress.
      const auto active=at::nonzero(retained).squeeze(1);const auto matrix=sorted.index_select(0,active).index_select(1,active);
      const auto ordinal=at::arange(active.numel(),0,-1,scores.options().dtype(at::kDouble));const auto kept=generic_nms(matrix,ordinal,o.nms_threshold);
      retained=at::zeros_like(retained);retained.index_fill_(0,active.index_select(0,kept),true);
    }
    keep.index_copy_(0,order,retained);
  }return keep;
}
std::vector<VideoDetections> postprocess_video_detections(const DetectionOutput& raw,const VideoDetectionOptions& o){
  c10::InferenceMode inference;TORCH_CHECK(raw.logits.dim()==3 && raw.logits.size(2)==1 && raw.masks.dim()==4 && raw.masks.sizes().slice(0,2)==raw.logits.sizes().slice(0,2) && raw.boxes_xyxy.sizes()==at::IntArrayRef({raw.logits.size(0),raw.logits.size(1),4}),"invalid video detection shapes");
  TORCH_CHECK(o.model==AssociationPolicy::Sam3 || o.model==AssociationPolicy::Sam31,"invalid video detector model");std::vector<VideoDetections> out;
  for(int64_t b=0;b<raw.logits.size(0);++b){auto logits=raw.logits[b].squeeze(1).clone();const auto masks=raw.masks[b],boxes=raw.boxes_xyxy[b];
    if(o.nms_threshold>0){const auto keep=video_mask_nms(logits.sigmoid(),masks,o);logits.sub_(keep.logical_not().to(at::kFloat)*1e4);}
    auto scores=logits.sigmoid();if(o.model==AssociationPolicy::Sam3 && !o.allow_new_detections)scores=scores-1e8;
    auto keep=scores.gt(o.score_threshold);
    if(o.model==AssociationPolicy::Sam31 && o.boundary_filter)keep=keep.logical_and(detection_boundary_keep(boxes,o.boundary_margin));
    if(o.model==AssociationPolicy::Sam31 && !o.allow_new_detections)keep=at::zeros_like(keep);
    const auto rows=o.model==AssociationPolicy::Sam3?at::nonzero(keep).squeeze(1):at::argsort(keep,0,true);
    out.push_back({masks.index_select(0,rows),scores.index_select(0,rows),boxes.index_select(0,rows),keep.index_select(0,rows)});
  }return out;
}
}
