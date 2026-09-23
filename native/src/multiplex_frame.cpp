#include "sam3/multiplex_frame.h"
#include "sam3/multiplex_storage.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <numeric>
namespace sam3 {
namespace {
// Source memory encoding reconstructs BCHW from sequence features. In particular,
// this sets the singleton batch stride for a channels-last shared image; passing
// its raw BCHW view can select numerically different convolution behavior.
at::Tensor memory_pixels(const TrackingFeatures& features) {
  return features.image.flatten(2).permute({2,0,1}).permute({1,2,0}).view({1,256,72,72});
}
MultiplexTemporalState temporal_state(const MultiplexFrameHistory& history) {
  MultiplexTemporalState out;
  for(const bool cond:{true,false}) for(const auto& frame:cond?history.conditioning:history.tracked)
    (cond?out.conditioning:out.tracked).push_back({frame.index,frame.memory,frame.memory_position,frame.pointer,frame.confidence,frame.image,frame.image_position});
  return out;
}
void restore_spatial(MultiplexFrameHistory& history,const MultiplexTemporalState& state) {
  for(const bool cond:{true,false}) {
    auto& frames=cond?history.conditioning:history.tracked;const auto& stored=cond?state.conditioning:state.tracked;
    for(size_t i=0;i<frames.size();++i) if(!frames[i].archive){frames[i].memory=stored[i].features;frames[i].memory_position=stored[i].position;}
  }
}
void drop_auxiliary(MultiplexFrame& frame,bool past) {
  if(frame.archive)frame=load_multiplex_frame(frame);
  frame.masks.low_res_multimasks=at::Tensor();frame.masks.high_res_multimasks=at::Tensor();frame.masks.iou=at::Tensor();frame.masks.object_pointer=at::Tensor();
  frame.iou=at::Tensor();frame.confidence=at::Tensor();
  if(past) {frame.masks.high_res_mask=at::Tensor();frame.memory=at::Tensor();frame.memory_position=at::Tensor();frame.image=at::Tensor();frame.image_position=at::Tensor();}
}
}
Sam31TrackingFrame::Sam31TrackingFrame(const WeightStore& store,at::Device device)
    :interactive_(store,"sam3.1",device),propagation_(store,device),temporal_(store,device),memory_(store,"sam3.1",device),
     no_memory_(store.read("sam3.1/tracker.model.interactivity_no_mem_embed",device)) {}
std::vector<at::Tensor> Sam31TrackingFrame::project_interactive(const std::vector<at::Tensor>& x,const std::string& mode) const {return interactive_.project_pyramid(x,mode);}
std::vector<at::Tensor> Sam31TrackingFrame::project_propagation(const std::vector<at::Tensor>& x,const std::string& mode) const {return propagation_.project_pyramid(x,mode);}
MultiplexFrame Sam31TrackingFrame::forward(const TrackingFeatures& interactive,const TrackingFeatures& propagation,
    const MultiplexFrameRequest& request,MultiplexFrameHistory& history,const MultiplexState& buckets,const MultiplexFrameOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  const auto device=no_memory_.device();AutocastGuard autocast(device.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(request.frame_count>0 && request.index>=0 && request.index<request.frame_count,"invalid tracking frame index");
  TORCH_CHECK(buckets.valid() && buckets.width()==16 && buckets.object_count()>0,"SAM3.1 requires valid 16-slot buckets");
  TORCH_CHECK(request.points.defined()==request.labels.defined(),"points and labels must be supplied together");
  const bool direct=request.mask.defined(),points=request.points.defined(),refine=request.previous_logits.defined();
  TORCH_CHECK(!refine || (points && request.objects_to_interact.has_value()),"previous logits require points and object indices");
  const bool propagate=!direct && (!points || (!refine && !request.initial && request.objects_to_interact.has_value()));
  const bool interact=!direct && points;
  TORCH_CHECK(direct || propagate || (interact && (refine || request.initial)),"cannot select a tracking path");
  const auto check_features=[&](const TrackingFeatures& features) {
    TORCH_CHECK(features.image.defined() && features.image.sizes()==at::IntArrayRef({1,256,72,72}) && features.image.device()==device && features.high.size()==2,"one shared image's 72/144/288 tracking features are required");
  };
  if(direct || interact)check_features(interactive);
  if(propagate || request.encode_memory || options.save_image)check_features(propagation);
  const auto multimask=[&](int64_t count) {return options.multimask && (request.initial || options.multimask_tracking) && count>=options.multimask_min_points && count<=options.multimask_max_points;};
  MultiplexFrame out;out.index=request.index;
  const auto interactive_image=[&] {const auto sequence=interactive.image.flatten(2).permute({2,0,1})+no_memory_;return sequence.permute({1,2,0}).view({1,256,72,72});};
  if(direct) {
    TORCH_CHECK(request.mask.size(0)==buckets.object_count(),"initial masks must match all objects in this state");
    out.masks=interactive_.use_mask_as_output(interactive_image(),interactive.high,request.mask,mode,options.object_threshold);
    out.conditioning_objects.resize(buckets.object_count());std::iota(out.conditioning_objects.begin(),out.conditioning_objects.end(),0);
  }else {
    if(propagate) {
      TORCH_CHECK(multimask(0),"the shipped SAM3.1 propagation decoder requires multimask output");
      auto state=temporal_state(load_selected_multiplex_history(history,request.index,request.frame_count,request.reverse,options.temporal));
      const auto image=temporal_.forward(propagation.image.flatten(2).permute({2,0,1}),propagation.position.flatten(2).permute({2,0,1}),72,72,
          request.index,request.frame_count,request.initial,request.reverse,true,state,buckets,options.temporal,mode);
      restore_spatial(history,state);
      out.masks=propagation_.forward(buckets,image,propagation.high,mode,options.object_threshold,options.attenuate_iou_by_stability);
    }
    if(interact) {
      auto ids=request.objects_to_interact.value_or(std::vector<int64_t>());
      if(!request.objects_to_interact){ids.resize(buckets.object_count());std::iota(ids.begin(),ids.end(),0);}
      TORCH_CHECK(request.points.dim()==3 && request.points.size(0)==int64_t(ids.size()),"point batch must match interacted objects");
      for(auto id:ids)TORCH_CHECK(id>=0 && id<buckets.object_count(),"interacted object index out of range");
      const auto indices=at::tensor(ids,at::TensorOptions().device(device).dtype(at::kLong));
      at::Tensor previous;
      if(refine)previous=request.previous_logits.index_select(0,indices);
      else if(propagate)previous=out.masks.low_res_mask.index_select(0,indices);
      auto result=interactive_.forward(interactive_image(),interactive.high,request.points,request.labels,previous,
          multimask(request.points.size(1)),mode,options.object_threshold,options.attenuate_iou_by_stability);
      out.conditioning_objects=ids;
      if(!propagate)out.masks=std::move(result);
      else {
        const auto merge=[&](at::Tensor& dst,const at::Tensor& src){dst.index_put_({indices},src.to(dst.scalar_type()));};
        merge(out.masks.low_res_multimasks,result.low_res_multimasks);merge(out.masks.high_res_multimasks,result.high_res_multimasks);
        merge(out.masks.low_res_mask,result.low_res_mask);merge(out.masks.high_res_mask,result.high_res_mask);
        merge(out.masks.iou,result.iou);merge(out.masks.object_pointer,result.object_pointer);merge(out.masks.object_logits,result.object_logits);
      }
    }
  }
  if(options.temporal.use_pointers)out.pointer=buckets.mux(out.masks.object_pointer);
  if(options.temporal.select_by_score){out.iou=std::get<0>(out.masks.iou.max(-1));out.confidence=memory_confidence(out.masks.object_logits,out.iou);}
  if(request.encode_memory && options.temporal.memory_slots>0) {
    const auto conditions=at::tensor(out.conditioning_objects,at::TensorOptions().device(device).dtype(at::kLong));
    // Match the source sequence -> BCHW view, including singleton N strides
    // observed by convolution dispatch for a channels-last shared image.
    const auto pixels=memory_pixels(propagation);
    const auto encoded=memory_.encode_frame(pixels,out.masks.high_res_mask,out.masks.object_logits,
        {points,options.non_overlap_memory,options.object_threshold},buckets.mux_matrix(),conditions,mode);
    out.memory=encoded.features;out.memory_position=encoded.position;
  }
  if(options.save_image){out.image=propagation.image.flatten(2).permute({2,0,1});out.image_position=propagation.position.flatten(2).permute({2,0,1});}
  if(options.offload_output) {
    out.masks.low_res_mask=out.masks.low_res_mask.cpu();out.masks.high_res_mask=out.masks.high_res_mask.cpu();
    if(out.memory.defined()){out.memory=out.memory.cpu();out.memory_position=out.memory_position.cpu();}
    if(out.image.defined()){out.image=out.image.cpu();out.image_position=out.image_position.cpu();}
    drop_auxiliary(out,false);
  }
  if(options.trim_history) {
    const auto old=request.index-options.temporal.stride*options.temporal.memory_slots,far=request.index-20*options.temporal.max_pointer_frames;
    for(auto& frame:history.tracked) {
      if(frame.index==old && (!options.temporal.select_by_score || (frame.confidence.defined()?(frame.confidence<options.temporal.score_threshold).item<bool>():0.<options.temporal.score_threshold)))drop_auxiliary(frame,true);
      if(frame.index==far && options.temporal.select_by_score && !options.offload_output)drop_auxiliary(frame,true);
    }
  }
  return out;
}
std::vector<int64_t> Sam31TrackingFrame::update_masks(const TrackingFeatures& interactive,const TrackingFeatures& propagation,
    const at::Tensor& masks,const std::vector<int64_t>& indices,const std::optional<std::vector<int64_t>>& object_ids,
    MultiplexFrame& frame,MultiplexState& buckets,const MultiplexMaskUpdate& request,const MultiplexFrameOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  const auto device=no_memory_.device();AutocastGuard autocast(device.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  auto updated=load_multiplex_frame(frame);
  for(auto* value:{&updated.masks.low_res_mask,&updated.masks.high_res_mask,&updated.masks.object_logits,&updated.iou,&updated.input_masks})if(value->defined())*value=value->to(device);
  TORCH_CHECK(buckets.valid() && buckets.width()==16,"SAM3.1 requires valid 16-slot buckets");
  TORCH_CHECK(masks.dim()==4 && masks.size(1)==1 && masks.size(0)==int64_t(indices.size()) && !indices.empty(),"masks must be [objects,1,H,W] with one index per mask");
  TORCH_CHECK(!object_ids || object_ids->size()==indices.size(),"one global ID per mask is required");
  TORCH_CHECK(updated.masks.low_res_mask.defined() && updated.masks.object_logits.defined(),"current frame masks and object logits are required");
  TORCH_CHECK(interactive.image.sizes()==at::IntArrayRef({1,256,72,72}) && interactive.image.device()==device && interactive.high.size()==2,"shared interactive image features are required");
  if(request.encode_memory)TORCH_CHECK(updated.masks.high_res_mask.defined() && propagation.image.defined() && (!options.save_image || (updated.image.defined() && updated.image_position.defined())),"memory update requires high masks and saved image features");
  // Stage changes so a failed capacity/shape/encoding check leaves caller state intact.
  auto state=buckets;auto affected=indices;
  at::Tensor pointers;if(options.temporal.use_pointers)pointers=buckets.demux(updated.pointer);
  if(request.append){affected=state.next_indices(indices.size(),request.allow_new_buckets,request.prefer_new_buckets);state.add_objects(affected,object_ids,request.allow_new_buckets,request.prefer_new_buckets);}
  else for(auto index:indices)TORCH_CHECK(index>=0 && index<state.object_count(),"reconditioning object index out of range");
  const auto sequence=interactive.image.flatten(2).permute({2,0,1})+no_memory_;
  auto output=interactive_.use_mask_as_output(sequence.permute({1,2,0}).view({1,256,72,72}),interactive.high,masks,mode,options.object_threshold);
  if(request.append && updated.masks.high_res_mask.defined() && updated.masks.high_res_mask.size(-1)!=output.high_res_mask.size(-1))
    updated.masks.high_res_mask=at::upsample_bilinear2d(updated.masks.high_res_mask,{output.high_res_mask.size(-1),output.high_res_mask.size(-1)},false);
  output.low_res_mask=at::_upsample_bilinear2d_aa(output.low_res_mask,{updated.masks.low_res_mask.size(-2),updated.masks.low_res_mask.size(-1)},false);
  const auto selected=at::tensor(affected,at::TensorOptions().device(device).dtype(at::kLong));
  const auto merge=[&](at::Tensor& target,const at::Tensor& value) {
    if(request.append)target=at::cat({target,value},0);
    else {auto replacement=target.clone();replacement.index_put_({selected},value.to(target.scalar_type()));target=std::move(replacement);}
  };
  merge(updated.masks.low_res_mask,output.low_res_mask);
  if(updated.masks.high_res_mask.defined())merge(updated.masks.high_res_mask,output.high_res_mask);
  merge(updated.masks.object_logits,output.object_logits);
  if(options.temporal.select_by_score)merge(updated.iou,output.iou.squeeze(-1));
  if(updated.input_masks.defined())merge(updated.input_masks,masks);
  if(options.temporal.use_pointers){merge(pointers,output.object_pointer.to(pointers.scalar_type()));updated.pointer=state.mux(pointers);}
  for(auto index:affected)if(std::find(updated.conditioning_objects.begin(),updated.conditioning_objects.end(),index)==updated.conditioning_objects.end())updated.conditioning_objects.push_back(index);
  if(request.encode_memory) {
    TORCH_CHECK(updated.masks.high_res_mask.size(0)==state.object_count(),"updated masks must match the multiplex state");
    const auto pixels=memory_pixels(propagation);
    const auto conditions=at::tensor(updated.conditioning_objects,at::TensorOptions().device(device).dtype(at::kLong));
    const auto encoded=memory_.encode_frame(pixels,updated.masks.high_res_mask,updated.masks.object_logits,
        {request.append && request.masks_from_points,options.non_overlap_memory,options.object_threshold},state.mux_matrix(),conditions,mode);
    updated.memory=encoded.features;updated.memory_position=encoded.position;updated.memory_masks=at::Tensor();updated.memory_object_logits=at::Tensor();
  }
  frame=std::move(updated);buckets=std::move(state);return affected;
}
void Sam31TrackingFrame::update_memory(const TrackingFeatures& propagation,const at::Tensor& high,const at::Tensor& scores,
    MultiplexFrame& stored,const MultiplexState& buckets,bool reapply,const MultiplexFrameOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);const auto device=no_memory_.device();AutocastGuard autocast(device.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(high.dim()==4 && high.size(0)==buckets.object_count() && high.size(1)==1 && high.size(2)>0 && high.size(3)>0 && high.is_floating_point() && scores.sizes()==at::IntArrayRef({buckets.object_count(),1}) && scores.is_floating_point(),"invalid memory mask/proxy score shape");
  auto frame=load_multiplex_frame(stored);
  const auto masks=high.to(device),logits=scores.to(device),conditions=at::tensor(frame.conditioning_objects,at::TensorOptions().device(device).dtype(at::kLong));
  const auto encoded=memory_.encode_frame(memory_pixels(propagation),masks,logits,{false,options.non_overlap_memory,options.object_threshold},buckets.mux_matrix(),conditions,mode);
  if(reapply){
    const auto suppressed=frame.masks.object_logits.to(device).gt(options.object_threshold).logical_and(logits.lt(0));
    if(suppressed.any().item<bool>()){
      const auto pointer=buckets.demux(frame.pointer.to(device)),flag=suppressed.to(at::kFloat);
      frame.pointer=buckets.mux(flag*interactive_.project_no_object_pointer(pointer,mode)+(1-flag)*pointer);
    }
  }
  frame.memory=encoded.features;frame.memory_position=encoded.position;
  frame.image=propagation.image.flatten(2).permute({2,0,1});frame.image_position=propagation.position.flatten(2).permute({2,0,1});
  frame.memory_masks=masks.clone();frame.memory_object_logits=logits.clone();stored=std::move(frame);
}
MaskMemoryOutput Sam31TrackingFrame::encode_history(const MultiplexFrame& stored,const MultiplexState& buckets,
    const MultiplexFrameOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);const auto device=no_memory_.device();
  const auto frame=load_multiplex_frame(stored,history_masks|history_spatial);
  const auto masks=frame.memory_masks.defined()?frame.memory_masks:frame.masks.high_res_mask;
  TORCH_CHECK(frame.image.defined() && frame.image.sizes()==at::IntArrayRef({5184,1,256}) && masks.defined(),"history rebuild requires retained shared image features and full-resolution masks");
  TORCH_CHECK(masks.size(0)==buckets.object_count(),"history masks must match the destination objects");
  const auto pixels=frame.image.to(device).permute({1,2,0}).view({1,256,72,72});
  const auto conditions=at::tensor(frame.conditioning_objects,at::TensorOptions().device(device).dtype(at::kLong));
  return memory_.encode_frame(pixels,masks.to(device),(frame.memory_object_logits.defined()?frame.memory_object_logits:frame.masks.object_logits).to(device),
      {false,options.non_overlap_memory,options.object_threshold},buckets.mux_matrix(),conditions,mode);
}
}
