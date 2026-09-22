#include "sam3/occlusion.h"
#include "sam3/autocast.h"
#include "sam3/ops.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <cmath>
namespace sam3 {
namespace {
void mode_check(const std::string& mode){TORCH_CHECK(mode=="fp32" || mode=="fp16" || mode=="bf16_reference","unknown precision");}
void mask_check(const at::Tensor& x){TORCH_CHECK(x.dim()==3 && x.is_floating_point() && x.size(1)>0 && x.size(2)>0 && (x.is_cuda() || x.device().is_cpu()),"masks must be floating [N,H,W] on CPU/CUDA");}
void policy_check(AssociationPolicy p){TORCH_CHECK(p==AssociationPolicy::Sam3 || p==AssociationPolicy::Sam31,"unknown model policy");}
int64_t find_id(const std::vector<int64_t>& ids,int64_t id){const auto it=std::find(ids.begin(),ids.end(),id);return it==ids.end()?-1:it-ids.begin();}
at::Tensor normalized_boxes(const at::Tensor& binary){const auto size=at::tensor({binary.size(2),binary.size(1),binary.size(2),binary.size(1)},at::kLong).to(binary.device());return tracking_mask_boxes(binary.unsqueeze(1)).squeeze(1)/size;}
void candidate_check(const ReconditionCandidates& c,int64_t detections){std::set<int64_t> ids;for(auto [id,d]:c){TORCH_CHECK(d>=0 && d<detections,"candidate detection index out of range");TORCH_CHECK(ids.insert(id).second,"duplicate candidate object ID");}}
}
OcclusionResult update_occlusion(const at::Tensor& masks,const at::Tensor& last,const at::Tensor& removed,int64_t frame,bool reverse,const OcclusionOptions& o,const at::Tensor& known_input,const std::string& mode){
  c10::InferenceMode inference;mask_check(masks);policy_check(o.policy);mode_check(mode);const auto n=masks.size(0);const auto device=masks.device();
  TORCH_CHECK(frame>=0 && std::isfinite(o.iou_threshold) && (o.policy==AssociationPolicy::Sam31 || !o.allow_unoccluded_to_suppress),"invalid occlusion options");
  TORCH_CHECK(last.sizes()==at::IntArrayRef{n} && last.scalar_type()==at::kLong && last.device()==device && removed.sizes()==at::IntArrayRef{n} && removed.scalar_type()==at::kBool && removed.device()==device,"occlusion history/removal shape or device mismatch");
  const auto known=known_input.defined()?known_input:at::ones({n},removed.options());TORCH_CHECK(known.sizes()==at::IntArrayRef{n} && known.scalar_type()==at::kBool && known.device()==device,"known must be bool[N]");
  AutocastGuard autocast(device.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  const auto binary=masks.gt(0);const auto prior=o.policy==AssociationPolicy::Sam31?at::where(removed,at::full_like(last,100000),last):at::where(known,last,at::where(removed,at::full_like(last,100000),at::full_like(last,-1)));
  auto suppressed=at::zeros({n},removed.options());
  if(n>1){const auto flat=binary.flatten(1).to(at::kFloat),intersection=at::mm(flat,flat.t()),area=flat.sum(1);const auto iou=intersection/(area.unsqueeze(1)+area.unsqueeze(0)-intersection).clamp_min(1);const auto pairs=iou.ge(o.iou_threshold).triu(1);
    const auto a=prior.unsqueeze(1),b=prior.unsqueeze(0);auto i=pairs.logical_and(reverse?a.lt(b):a.gt(b)),j=pairs.logical_and(reverse?b.lt(a):b.gt(a));
    if(!o.allow_unoccluded_to_suppress){i=i.logical_and(b.gt(-1));j=j.logical_and(a.gt(-1));}suppressed=i.any(1).logical_or(j.any(0));
  }
  const auto updated=at::where(binary.any(at::IntArrayRef{1,2}).logical_not().logical_or(suppressed),at::full_like(prior,frame),prior);
  return {masks.masked_fill(suppressed.view({n,1,1}),-10.),suppressed,updated};
}
DeviceOcclusionResult update_device_occlusion(const DeviceHotstartState& previous,const at::Tensor& masks,const at::Tensor& removed,int64_t frame,bool reverse,double threshold,bool allow,const std::string& mode){
  TORCH_CHECK(previous.size()==masks.size(0),"hotstart state/mask count mismatch");const auto out=update_occlusion(masks,previous.last_occluded,removed,frame,reverse,{AssociationPolicy::Sam31,threshold,allow},{},mode);auto next=previous;next.last_occluded=out.last_occluded;return {std::move(next),out.masks,out.suppressed};
}
HostOcclusionResult update_host_occlusion(const std::map<int64_t,at::Tensor>& history,const std::vector<int64_t>& ids,const at::Tensor& masks,const std::set<int64_t>& removed,int64_t frame,bool reverse,double threshold,const std::string& mode){
  c10::InferenceMode inference;mask_check(masks);TORCH_CHECK(ids.size()==uint64_t(masks.size(0)),"host occlusion ID/mask count mismatch");if(ids.empty())return {history,masks.clone(),at::empty({0},masks.options().dtype(at::kBool))};
  std::vector<at::Tensor> values;std::vector<int64_t> known,drop;for(auto id:ids){const auto it=history.find(id);values.push_back(it==history.end()?at::full({1},-1,masks.options().dtype(at::kLong)):it->second);known.push_back(it!=history.end());drop.push_back(removed.count(id));}
  const auto boolean=[&](const auto& v){return at::tensor(v,at::kLong).to(masks.device()).to(at::kBool);};const auto out=update_occlusion(masks,at::cat(values),boolean(drop),frame,reverse,{AssociationPolicy::Sam3,threshold,false},boolean(known),mode);
  HostOcclusionResult result{{},out.masks,out.suppressed};for(size_t i=0;i<ids.size();++i)result.history[ids[i]]=out.last_occluded.slice(0,i,i+1);return result;
}
at::Tensor tracking_mask_boxes(const at::Tensor& masks){
  c10::InferenceMode inference;TORCH_CHECK(masks.dim()==4 && masks.size(1)==1 && masks.scalar_type()==at::kBool && masks.size(2)>0 && masks.size(3)>0 && masks.size(2)<=INT32_MAX && masks.size(3)<=INT32_MAX,"box masks must be bool[N,1,H,W]");const auto h=masks.size(2),w=masks.size(3);const auto opts=masks.options().dtype(at::kInt);
  const auto cols=masks.squeeze(1).any(1),rows=masks.squeeze(1).any(2),xs=at::arange(w,opts),ys=at::arange(h,opts);
  const auto boxes=at::stack({at::where(cols,xs,w).amin(1),at::where(rows,ys,h).amin(1),at::where(cols,xs,-1).amax(1),at::where(rows,ys,-1).amax(1)},1).unsqueeze(1);
  // Reuse the row projection for emptiness instead of scanning all pixels again.
  return at::where(rows.any(1).view({masks.size(0),1,1}),boxes,at::zeros_like(boxes));
}
at::Tensor diagonal_box_iou(const at::Tensor& a,const at::Tensor& b){
  TORCH_CHECK(a.dim()==2 && a.size(1)==4 && a.sizes()==b.sizes() && a.is_floating_point() && b.is_floating_point() && a.device()==b.device(),"boxes must share [N,4] and device");
  const auto a0=a.slice(1,0,2),a1=a.slice(1,2,4),b0=b.slice(1,0,2),b1=b.slice(1,2,4);const auto area_a=(a1-a0).prod(-1),area_b=(b1-b0).prod(-1);const auto inter=(at::minimum(a1,b1)-at::maximum(a0,b0)).clamp_min(0).prod(-1);return inter/(area_a+area_b-inter);
}
at::Tensor clean_video_mask_scores(const at::Tensor& input,int64_t area,bool holes,bool sprinkles){
  c10::InferenceMode inference;TORCH_CHECK(input.dim()==4 && input.size(1)==1 && input.is_floating_point(),"video mask cleanup needs floating [N,1,H,W]");if(area<=0 || input.size(0)==0)return input;auto mask=input;
  if(holes){const auto bg=mask.le(0);const auto counts=std::get<1>(connected_components(bg.to(at::kByte)));mask=at::where(bg.logical_and(counts.le(area)),.1,mask);}
  if(sprinkles){const auto fg=mask.gt(0);const auto threshold=at::floor_divide(fg.sum(at::IntArrayRef{2,3},true,at::kInt),2).clamp_max(area);const auto counts=std::get<1>(connected_components(fg.to(at::kByte)));mask=at::where(fg.logical_and(counts.le(threshold)),-.1,mask);}return mask;
}
ReconditionCandidates recondition_candidates(const AssociationMetadata& a){
  ReconditionCandidates out;if(a.recondition_order.empty()){for(const auto& p:a.track_to_recondition_detection)out.push_back(p);return out;}TORCH_CHECK(a.recondition_order.size()==a.track_to_recondition_detection.size(),"candidate order/map size mismatch");std::set<int64_t> seen;for(auto id:a.recondition_order){TORCH_CHECK(seen.insert(id).second,"duplicate candidate order ID");out.emplace_back(id,a.track_to_recondition_detection.at(id));}return out;
}
ReconditionDecision recondition_decision(const ReconditionCandidates& candidates,const std::vector<int64_t>& ids,const at::Tensor& boxes,const at::Tensor& scores,const at::Tensor& masks,int64_t frame,const ReconditionOptions& o,const std::string& mode){
  c10::InferenceMode inference;mask_check(masks);policy_check(o.policy);mode_check(mode);TORCH_CHECK(frame>=0 && std::isfinite(o.box_iou_threshold) && std::isfinite(o.detection_score_threshold) && boxes.dim()==2 && boxes.size(1)==4 && boxes.is_floating_point() && scores.dim()==1 && scores.numel()==boxes.size(0) && scores.is_floating_point() && ids.size()==uint64_t(masks.size(0)) && boxes.device()==masks.device() && scores.device()==masks.device(),"invalid reconditioning inputs");candidate_check(candidates,boxes.size(0));AutocastGuard autocast(masks.device().type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  ReconditionDecision out;out.periodic=o.period>0 && frame%o.period==0 && !candidates.empty();if(o.box_iou_threshold<=0 || candidates.empty())return out;
  if(o.policy==AssociationPolicy::Sam3){for(auto [id,d]:candidates){const auto t=find_id(ids,id);if(t<0)continue;const auto binary=masks.slice(0,t,t+1).gt(0);if(!binary.any().item<bool>())continue;const auto iou=diagonal_box_iou(boxes.slice(0,d,d+1),normalized_boxes(binary));if(iou[0].lt(o.box_iou_threshold).item<bool>() && scores[d].ge(o.detection_score_threshold).item<bool>()){out.geometry=true;out.geometry_ids.insert(id);}}}
  else{std::vector<int64_t> ti,di;for(auto [id,d]:candidates){const auto t=find_id(ids,id);TORCH_CHECK(t>=0,"SAM3.1 candidate IDs must all be present (source paired-box lengths)");ti.push_back(t);di.push_back(d);}const auto index=[&](const auto& x){return at::tensor(x,at::kLong).to(masks.device());};const auto detection=index(di),track=index(ti);const auto iou=diagonal_box_iou(boxes.index_select(0,detection),normalized_boxes(masks.index_select(0,track).gt(0)));out.geometry=iou[0].lt(o.box_iou_threshold).item<bool>() && scores.index_select(0,detection).ge(o.detection_score_threshold).any().item<bool>();}
  return out;
}
ReconditionMasks prepare_recondition_masks(const ReconditionCandidates& candidates,const std::vector<int64_t>& ids,const at::Tensor& detections,const at::Tensor& scores,const at::Tensor& masks,int64_t height,int64_t width,AssociationPolicy policy,int64_t area,const std::string& mode){
  c10::InferenceMode inference;mask_check(detections);mask_check(masks);policy_check(policy);mode_check(mode);TORCH_CHECK(height>0 && width>0 && ids.size()==uint64_t(masks.size(0)) && scores.dim()==1 && scores.size(0)==masks.size(0) && scores.is_floating_point() && detections.device()==masks.device() && scores.device()==masks.device(),"invalid reconditioning mask inputs");candidate_check(candidates,detections.size(0));AutocastGuard autocast(masks.device().type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  ReconditionMasks out;out.low_masks=masks.clone();std::vector<int64_t> track;for(auto [id,d]:candidates){const auto t=find_id(ids,id);TORCH_CHECK(t>=0,"reconditioning object ID is absent");track.push_back(t);}
  const auto index=[&](const auto& x){return at::tensor(x,at::kLong).to(masks.device());};auto score=scores.index_select(0,index(track));if(policy==AssociationPolicy::Sam31)score=score.sigmoid();const auto valid=score.gt(.8).cpu();
  for(size_t i=0;i<candidates.size();++i)if(valid[i].item<bool>()){out.ids.push_back(candidates[i].first);out.detection_indices.push_back(candidates[i].second);out.track_indices.push_back(track[i]);}
  if(out.ids.empty()){out.binary_masks=at::empty({0,height,width},masks.options().dtype(at::kBool));return out;}
  if(policy==AssociationPolicy::Sam3){std::vector<at::Tensor> binaries;for(auto d:out.detection_indices)binaries.push_back(at::upsample_bilinear2d(detections.slice(0,d,d+1).unsqueeze(1),{height,width},false)[0][0].gt(0));out.binary_masks=at::stack(binaries);}
  else{const auto selected=detections.index_select(0,index(out.detection_indices));TORCH_CHECK(selected.size(1)==masks.size(1) && selected.size(2)==masks.size(2),"SAM3.1 reconditioning requires equal low-mask resolutions");out.binary_masks=at::upsample_bilinear2d(selected.unsqueeze(1),{height,width},false).squeeze(1).gt(0);const auto indices=index(out.track_indices);const auto old=masks.index_select(0,indices);const auto updated=clean_video_mask_scores(at::where(selected.gt(0).eq(old.gt(0)),old,selected).unsqueeze(1),area).squeeze(1);TORCH_CHECK(updated.scalar_type()==masks.scalar_type(),"source reconditioning scatter requires matching mask dtype");out.low_masks.index_copy_(0,indices,updated);}
  return out;
}
std::vector<ReconditionBatch> recondition_batches(const ReconditionMasks& prepared,const std::vector<std::vector<int64_t>>& states,AssociationPolicy policy,bool multiplex){
  policy_check(policy);TORCH_CHECK(prepared.binary_masks.dim()==3 && prepared.binary_masks.size(0)==int64_t(prepared.ids.size()),"reconditioning binary mask/ID mismatch");std::vector<ReconditionBatch> out;std::map<int64_t,size_t> group;std::vector<std::vector<int64_t>> rows;
  for(size_t i=0;i<prepared.ids.size();++i){for(size_t s=0;s<states.size();++s){if(find_id(states[s],prepared.ids[i])<0)continue;
    if(policy==AssociationPolicy::Sam3 || !multiplex){out.push_back({int64_t(s),{prepared.ids[i]},prepared.binary_masks.slice(0,i,i+1)});}
    else{auto it=group.find(s);if(it==group.end()){group[s]=out.size();out.push_back({int64_t(s),{}, {}});rows.emplace_back();it=group.find(s);}out[it->second].ids.push_back(prepared.ids[i]);rows[it->second].push_back(i);}
    if(policy==AssociationPolicy::Sam31)break;
  }}
  if(policy==AssociationPolicy::Sam31 && multiplex)for(size_t i=0;i<out.size();++i)out[i].masks=prepared.binary_masks.index_select(0,at::tensor(rows[i],at::kLong).to(prepared.binary_masks.device()));return out;
}
}
