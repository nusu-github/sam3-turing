#include "sam3/tracking_frame.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
namespace {
TemporalState temporal_state(const TrackingHistory& history) {
  TemporalState result;
  for (const auto* frames:{&history.conditioning,&history.tracked}) {
    auto& target=frames==&history.conditioning?result.conditioning:result.tracked;
    for (const auto& frame:*frames) target.push_back({frame.index,frame.memory,frame.memory_position,frame.pointer,frame.confidence});
  }
  return result;
}
TrackingFrame* find_frame(std::vector<TrackingFrame>& frames,int64_t index) {
  for (auto& frame:frames) if (frame.index==index) return &frame;
  return nullptr;
}
void trim(TrackingFrame& frame) {
  // The source keeps only low masks, pointers and object logits when trimming.
  frame.high_mask=at::Tensor();frame.iou=at::Tensor();frame.confidence=at::Tensor();
  frame.memory=at::Tensor();frame.memory_position=at::Tensor();
}
}
Sam3TrackingFrame::Sam3TrackingFrame(const WeightStore& store,at::Device device)
    :heads_(store,"sam3",device),temporal_(store,device),memory_(store,"sam3",device) {}
std::vector<at::Tensor> Sam3TrackingFrame::project_pyramid(const std::vector<at::Tensor>& features,const std::string& mode) const {
  return heads_.project_pyramid(features,mode);
}
MaskMemoryOutput Sam3TrackingFrame::encode_memory(const TrackingFeatures& features,const at::Tensor& masks,
    const at::Tensor& logits,bool from_points,bool non_overlap,const std::string& mode) const {
  return memory_.encode_frame(features.image,masks,logits,{from_points,non_overlap,0.},{},{},mode);
}
TrackingFrame Sam3TrackingFrame::forward(const TrackingFeatures& features,const TrackingFrameRequest& request,
    TrackingHistory& history,const TrackingFrameOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  TORCH_CHECK(features.image.dim()==4 && features.image.size(0)>0 && features.image.size(1)==256 && features.image.size(2)==72 && features.image.size(3)==72,"tracking features must be [B,256,72,72]");
  TORCH_CHECK(features.position.sizes()==features.image.sizes() && features.position.device()==features.image.device(),"tracking positions must match image features");
  TORCH_CHECK(request.frame_count>0 && request.index>=0 && request.index<request.frame_count,"invalid tracking frame index");
  TORCH_CHECK(options.multimask_min_points>=0 && options.multimask_max_points>=options.multimask_min_points,"invalid multimask point policy");
  AutocastGuard autocast(features.image.device().type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  VideoMaskOutput decoded;
  if (request.mask.defined()) decoded=heads_.use_mask_as_output(features.image,features.high,request.mask,mode);
  else {
    const auto source=features.image.flatten(2).permute({2,0,1}),position=features.position.flatten(2).permute({2,0,1});
    const auto conditioned=temporal_.forward(source,position,72,72,request.index,request.frame_count,request.initial,request.reverse,
        request.use_previous,temporal_state(history),options.temporal,mode);
    if (request.previous_logits.defined()) TORCH_CHECK(request.points.defined(),"previous mask logits require point inputs");
    const auto point_count=request.points.defined()?request.points.size(1):0;
    const auto multi=options.multimask && (request.initial || options.multimask_tracking) && point_count>=options.multimask_min_points && point_count<=options.multimask_max_points;
    decoded=heads_.forward(conditioned,features.high,request.points,request.labels,request.previous_logits,multi,mode);
  }
  TrackingFrame result;
  result.index=request.index;result.low_mask=decoded.low_res_mask;result.high_mask=decoded.high_res_mask;
  result.pointer=decoded.object_pointer;result.object_logits=decoded.object_logits;
  if (options.temporal.select_by_score) {
    result.iou=std::get<0>(decoded.iou.max(-1));
    result.confidence=memory_confidence(result.object_logits,result.iou);
  }
  if (request.encode_memory && options.temporal.memory_slots>0) {
    const auto encoded=encode_memory(features,result.high_mask,result.object_logits,request.points.defined(),options.non_overlap_memory,mode);
    result.memory=encoded.features;result.memory_position=encoded.position;
  }
  if (options.offload_output) {
    result.low_mask=result.low_mask.cpu();result.high_mask=result.high_mask.cpu();
    if (result.memory.defined()) {result.memory=result.memory.cpu();result.memory_position=result.memory_position.cpu();}
    if (result.iou.defined()) {result.iou=result.iou.cpu();result.confidence=result.confidence.cpu();}
  }
  if (options.trim_history) {
    const auto past=request.index-options.temporal.stride*options.temporal.memory_slots;
    if (auto entry=find_frame(history.tracked,past)) {
      // Preserve short-circuiting: no score read/GPU synchronization is needed
      // when score-based history selection is disabled.
      if (!options.temporal.select_by_score || (entry->confidence.defined()?
          (entry->confidence<options.temporal.score_threshold).item<bool>():0.<options.temporal.score_threshold)) trim(*entry);
    }
    if (options.temporal.select_by_score && !options.offload_output)
      if (auto entry=find_frame(history.tracked,request.index-20*options.temporal.max_pointer_frames)) trim(*entry);
  }
  return result;
}
}
