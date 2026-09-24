#include "sam3/video_update.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <limits>
namespace sam3 {
std::vector<int64_t> VideoMetadata::object_ids()const{std::vector<int64_t> out;for(const auto& ids:ids_per_rank)out.insert(out.end(),ids.begin(),ids.end());return out;}
VideoMetadata initialize_video_metadata(int64_t ranks,at::Device device){TORCH_CHECK(ranks>0,"video metadata requires at least one rank");VideoMetadata out;out.ids_per_rank.resize(ranks);out.buckets_per_rank.resize(ranks,0);out.device=empty_device_hotstart(device);return out;}
namespace {
at::Tensor index(const std::vector<int64_t>& ids,const at::Tensor& like){return at::tensor(ids,like.options().dtype(at::kLong));}
void validate_metadata(const VideoMetadata& previous){
  TORCH_CHECK(!previous.ids_per_rank.empty() && previous.ids_per_rank.size()==previous.buckets_per_rank.size(),"invalid rank metadata");std::set<int64_t> unique;
  for(auto id:previous.object_ids())TORCH_CHECK(id<=previous.max_id && unique.insert(id).second,"invalid or duplicate metadata object ID");
  for(auto count:previous.buckets_per_rank)TORCH_CHECK(count>=0,"negative bucket count");
}
}
VideoUpdatePlan plan_video_update(int64_t frame,bool reverse,const VideoDetections& d,const at::Tensor& masks,const at::Tensor& logits,const VideoMetadata& previous,const VideoUpdateOptions& o){
  c10::InferenceMode inference;validate_metadata(previous);TORCH_CHECK(frame>=0,"negative video frame");const bool mux=o.association.policy==AssociationPolicy::Sam31;
  TORCH_CHECK(o.recondition.policy==o.association.policy,"reconditioning and association policies must match");
  VideoUpdatePlan out;out.metadata=previous;auto& next=out.metadata;next.last_occluded.clear();out.previous_ids=previous.object_ids();
  TORCH_CHECK(masks.dim()==3 && masks.size(0)==int64_t(out.previous_ids.size()) && logits.dim()==1 && logits.size(0)==masks.size(0) && logits.device()==masks.device(),"tracking mask/score/ID count mismatch");
  TORCH_CHECK(d.boxes.sizes()==at::IntArrayRef({d.masks.size(0),4}) && d.boxes.device()==masks.device(),"invalid detection boxes");
  const auto scores=d.scores.to(at::kFloat);
  const auto tensors=associate_tracking(d.masks,scores,masks,d.keep,o.association,o.policy_mode);
  out.association=realize_association(tensors,out.previous_ids,o.association.policy);
  if(!mux && o.boundary_filter && !out.association.new_detections.empty()){
    const auto keep=detection_boundary_keep(d.boxes.index_select(0,index(out.association.new_detections,d.boxes)),o.boundary_margin).cpu();std::vector<int64_t> remaining;
    for(size_t i=0;i<out.association.new_detections.size();++i)if(keep[i].item<bool>())remaining.push_back(out.association.new_detections[i]);out.association.new_detections=std::move(remaining);
  }
  const auto count=int64_t(out.association.new_detections.size());TORCH_CHECK(!count || previous.max_id<=INT64_MAX-count,"video object ID overflow");
  for(int64_t i=0;i<count;++i)out.new_ids.push_back(previous.max_id+1+i);
  std::vector<int64_t> workload=previous.buckets_per_rank;if(!mux)for(size_t r=0;r<workload.size();++r)workload[r]=previous.ids_per_rank[r].size();
  out.new_ranks=assign_detection_devices(count,workload,mux?o.bucket_capacity:1);
  out.suppressed=at::zeros({masks.size(0)},masks.options().dtype(at::kBool));auto remove=at::zeros_like(out.suppressed);
  if(o.warmup_complete){
    if(mux){TORCH_CHECK(previous.device.size()==masks.size(0),"device hotstart metadata must align with current global IDs");const auto hot=update_device_hotstart(previous.device,tensors,frame,reverse,o.hotstart);next.device=hot.state;remove=hot.remove;out.suppressed=hot.suppress;
      const auto cpu=remove.cpu();for(size_t i=0;i<out.previous_ids.size();++i)if(cpu[i].item<bool>())out.removed.insert(out.previous_ids[i]);
    }else{const auto hot=update_host_hotstart(previous.host,out.association,out.new_ids,frame,reverse,o.hotstart);next.host=hot.state;out.removed=hot.newly_removed;}
  }
  out.tracking_masks=masks.clone();const auto candidates=recondition_candidates(out.association);
  out.correction_decision=recondition_decision(candidates,out.previous_ids,d.boxes,scores,masks,frame,o.recondition,o.policy_mode);
  out.reconditioned=out.correction_decision.geometry_ids;
  if(out.correction_decision.periodic || out.correction_decision.geometry){out.corrections=prepare_recondition_masks(candidates,out.previous_ids,d.masks,logits,masks,1152,1152,o.association.policy,o.cleanup_area,o.policy_mode);out.tracking_masks=out.corrections.low_masks;}
  else {out.corrections.low_masks=out.tracking_masks;out.corrections.binary_masks=at::empty({0,1152,1152},masks.options().dtype(at::kBool));}
  if(masks.size(0) && o.warmup_complete && o.occlusion_threshold>0){
    if(mux){const auto result=update_device_occlusion(next.device,out.tracking_masks,remove,frame,reverse,o.occlusion_threshold,o.allow_unoccluded_suppression,o.policy_mode);next.device=result.state;out.tracking_masks=result.masks;}
    else {const auto result=update_host_occlusion(previous.last_occluded,out.previous_ids,out.tracking_masks,out.removed,frame,reverse,o.occlusion_threshold,o.policy_mode);next.last_occluded=result.history;out.tracking_masks=result.masks;}
  }
  for(size_t r=0;r<next.ids_per_rank.size();++r){auto& ids=next.ids_per_rank[r];for(size_t i=0;i<out.new_ids.size();++i)if(out.new_ranks[i]==int64_t(r))ids.push_back(out.new_ids[i]);ids.erase(std::remove_if(ids.begin(),ids.end(),[&](auto id){return out.removed.count(id);}),ids.end());}
  const auto cpu_scores=scores.cpu();
  for(size_t i=0;i<out.new_ids.size();++i){const auto id=out.new_ids[i],di=out.association.new_detections[i];next.object_scores[id]=cpu_scores[di].item<double>();next.frame_scores[frame][id]=scores[di].clone();next.max_id=std::max(next.max_id,id);}
  for(auto id:out.removed){next.object_scores[id]=-1e4;next.frame_scores[frame][id]=at::scalar_tensor(-1e4,scores.options());next.last_occluded.erase(id);}
  if(o.confirmation_enabled)next.confirmation=update_confirmation(previous.confirmation,out.previous_ids,next.object_ids(),out.association.detection_to_tracks,out.new_ids,o.confirmation_threshold);
  if(mux && o.warmup_complete){
    const auto compact=compact_device_hotstart(next.device);next.device=extend_device_hotstart(compact.first,count,frame,o.hotstart.initial_keep_alive);
    // Source appends new tensor rows globally, but rank-concatenated IDs may
    // interleave them with older rows. Explicitly restore ID alignment.
    std::vector<int64_t> intermediate;for(auto id:out.previous_ids)if(!out.removed.count(id))intermediate.push_back(id);intermediate.insert(intermediate.end(),out.new_ids.begin(),out.new_ids.end());
    const auto final_ids=next.object_ids();if(intermediate!=final_ids){const auto rows=video_memory_rows(intermediate,{final_ids})[0];next.device=select_device_hotstart(next.device,index(rows,masks));}
  }
  return out;
}
void finalize_video_scores(VideoMetadata& meta,int64_t frame,const std::vector<int64_t>& ids,const at::Tensor& logits){
  c10::InferenceMode inference;TORCH_CHECK(logits.dim()==1 && logits.size(0)==int64_t(ids.size()) && logits.is_floating_point(),"invalid final tracking scores");const auto probabilities=logits.sigmoid();for(size_t i=0;i<ids.size();++i)meta.frame_scores[frame][ids[i]]=probabilities[i].clone();
}
namespace {
template<class Session> std::vector<Session*> borrowed(const std::vector<std::unique_ptr<Session>>& sessions){std::vector<Session*> out;for(const auto& s:sessions){TORCH_CHECK(s,"null local video session");out.push_back(s.get());}return out;}
template<class Sessions,class Factory> void births_removals(int64_t frame,int64_t rank,VideoUpdatePlan& p,const VideoDetections& d,Sessions& sessions,const Factory& factory){
  std::vector<int64_t> ids,rows;for(size_t i=0;i<p.new_ids.size();++i)if(p.new_ranks[i]==rank){ids.push_back(p.new_ids[i]);rows.push_back(p.association.new_detections[i]);}
  if(!ids.empty())add_video_objects(frame,ids,d.masks.index_select(0,index(rows,d.masks)),sessions,factory);
  if(!p.removed.empty())remove_video_objects(std::vector<int64_t>(p.removed.begin(),p.removed.end()),sessions);
}
void check_rank(int64_t rank,const VideoUpdatePlan& p){TORCH_CHECK(rank>=0 && rank<int64_t(p.metadata.ids_per_rank.size()),"invalid execution rank");}
}
void execute_video_update(int64_t frame,int64_t rank,VideoUpdatePlan& p,const VideoDetections& d,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,const VideoUpdateOptions& o){
  check_rank(rank,p);TORCH_CHECK(o.association.policy==AssociationPolicy::Sam3,"SAM3 execution policy required");const auto local=borrowed(sessions);
  if(!p.corrections.ids.empty())execute_reconditioning(frame,p.corrections,local);
  if(!p.previous_ids.empty())update_video_memories(frame,p.tracking_masks,p.previous_ids,local,o.warmup_complete);
  births_removals(frame,rank,p,d,sessions,factory);
}
void execute_video_update(int64_t frame,int64_t rank,VideoUpdatePlan& p,const VideoDetections& d,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,const VideoUpdateOptions& o){
  check_rank(rank,p);TORCH_CHECK(o.association.policy==AssociationPolicy::Sam31,"SAM3.1 execution policy required");const auto local=borrowed(sessions);
  if(!p.corrections.ids.empty()){const auto edited=execute_reconditioning(frame,p.corrections,local);p.reconditioned.insert(edited.affected_ids.begin(),edited.affected_ids.end());}
  if(!p.previous_ids.empty())update_video_memories(frame,p.tracking_masks,p.previous_ids,local,o.reapply_no_object_pointer);
  births_removals(frame,rank,p,d,sessions,factory);int64_t buckets=0;for(const auto& s:sessions)if(s->state().buckets)buckets+=s->state().buckets->bucket_count();p.metadata.buckets_per_rank[rank]=buckets;
}
std::map<int64_t,at::Tensor> build_video_outputs(const VideoUpdatePlan& p,const VideoDetections& d,int64_t h,int64_t w,const VideoUpdateOptions& o){
  c10::InferenceMode inference;TORCH_CHECK(h>0 && w>0,"invalid video output size");std::map<int64_t,at::Tensor> out;
  if(!p.previous_ids.empty()){const auto masks=at::upsample_bilinear2d(p.tracking_masks.unsqueeze(1),{h,w},false).gt(0);for(size_t i=0;i<p.previous_ids.size();++i)out[p.previous_ids[i]]=masks[i];}
  if(!p.new_ids.empty()){const auto selected=d.masks.index_select(0,index(p.association.new_detections,d.masks)).unsqueeze(1);const auto masks=at::upsample_bilinear2d(clean_video_mask_scores(selected,o.cleanup_area),{h,w},false).gt(0);for(size_t i=0;i<p.new_ids.size();++i)out[p.new_ids[i]]=masks[i];}
  if(o.association.policy==AssociationPolicy::Sam3)for(auto id:p.reconditioned){const auto it=p.association.track_to_recondition_detection.find(id);if(it!=p.association.track_to_recondition_detection.end())out[id]=at::upsample_bilinear2d(d.masks.slice(0,it->second,it->second+1).unsqueeze(1).to(at::kFloat),{h,w},false).gt(0)[0];}
  return out;
}
}
