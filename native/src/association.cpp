#include "sam3/association.h"
#include "sam3/autocast.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <cmath>
#include <limits>
namespace sam3 {
namespace {
void mask_input(const at::Tensor& x){TORCH_CHECK(x.dim()==3 && x.is_floating_point() && x.size(1)>0 && x.size(2)>0,"association masks must be floating [N,H,W]");}
at::Tensor resize(const at::Tensor& x,const at::Tensor& target){return at::upsample_bilinear2d(x.unsqueeze(1),{target.size(1),target.size(2)},false).squeeze(1);}
at::Tensor overlap(const at::Tensor& a,const at::Tensor& b,bool iom){
  const auto flat_a=a.flatten(1).to(at::kFloat),flat_b=b.flatten(1).to(at::kFloat);
  const auto intersection=at::mm(flat_a,flat_b.t());
  if(iom){const auto area_a=a.flatten(1).sum(1),area_b=b.flatten(1).sum(1);return intersection.to(at::kLong)/at::minimum(area_a.unsqueeze(1),area_b.unsqueeze(0)).add(1e-8);}
  const auto area_a=flat_a.sum(1),area_b=flat_b.sum(1);
  return intersection/(area_a.unsqueeze(1)+area_b.unsqueeze(0)-intersection).clamp_min(1);
}
}
AssociationTensors associate_tracking(const at::Tensor& detections,const at::Tensor& scores,const at::Tensor& tracks,const at::Tensor& keep_input,const AssociationOptions& o,const std::string& mode){
  c10::InferenceMode inference;mask_input(detections);mask_input(tracks);
  TORCH_CHECK((detections.device().is_cpu() || detections.is_cuda()) && tracks.device()==detections.device() && scores.device()==detections.device(),"association inputs must share a CPU or CUDA device");
  const auto n=detections.size(0),m=tracks.size(0);
  TORCH_CHECK(scores.dim()==1 && scores.size(0)==n && scores.is_floating_point(),"association scores must be floating [N]");
  TORCH_CHECK(o.policy==AssociationPolicy::Sam3 || o.policy==AssociationPolicy::Sam31,"unknown association policy");
  TORCH_CHECK(o.pad_tracks_to>=0 && (o.policy==AssociationPolicy::Sam31 || o.pad_tracks_to==0),"track padding is a SAM3.1 option");
  for(double value:{o.new_detection_threshold,o.track_match_threshold,o.detection_match_threshold,o.high_confidence_threshold,o.iom_recondition_threshold,o.iou_recondition_threshold})TORCH_CHECK(std::isfinite(value),"association thresholds must be finite");
  TORCH_CHECK(mode=="fp32" || mode=="fp16" || mode=="bf16_reference","unknown association precision");
  AutocastGuard autocast(detections.device().type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  const auto boolean=detections.options().dtype(at::kBool),integer=detections.options().dtype(at::kLong);
  const auto keep=keep_input.defined()?keep_input:at::ones({n},boolean);
  TORCH_CHECK(keep.dim()==1 && keep.size(0)==n && keep.scalar_type()==at::kBool && keep.device()==detections.device(),"keep must be bool [N] on the mask device");
  TORCH_CHECK(o.policy!=AssociationPolicy::Sam3 || !keep_input.defined() || keep.all().item<bool>(),"SAM3 association expects already filtered detections");
  if(m==0)return {at::zeros({0},boolean),at::zeros({0},boolean),o.policy==AssociationPolicy::Sam3?at::ones({n},boolean):scores.ge(o.new_detection_threshold),at::full({n},-1,integer),scores.ge(o.high_confidence_threshold),at::zeros({n},boolean),keep,at::zeros({n,0},boolean)};
  if(n==0){const auto nonempty=tracks.gt(0).any(at::IntArrayRef{1,2});return {o.policy==AssociationPolicy::Sam3?nonempty:at::ones({m},boolean),nonempty,at::zeros({0},boolean),at::empty({0},integer),at::zeros({0},boolean),at::zeros({0},boolean),keep,at::zeros({0,m},boolean)};}
  auto det=detections,trk=tracks;
  if(det.size(1)!=trk.size(1) || det.size(2)!=trk.size(2)){
    if(det.size(1)*det.size(2)<trk.size(1)*trk.size(2))trk=resize(trk,det);else det=resize(det,trk);
  }
  if(o.pad_tracks_to>m)trk=at::cat({trk,at::zeros({o.pad_tracks_to-m,trk.size(1),trk.size(2)},trk.options())},0);
  auto det_binary=det.gt(0);det_binary.masked_fill_(keep.logical_not().view({n,1,1}),false);
  const auto trk_binary=trk.gt(0);auto metric=overlap(det_binary,trk_binary,o.use_iom);
  const auto nonempty=trk_binary.any(at::IntArrayRef{1,2});
  const auto unmatched=nonempty.logical_and(metric.ge(o.track_match_threshold).any(0).logical_not());
  const auto is_new=scores.ge(o.new_detection_threshold).logical_and(keep).logical_and(metric.ge(o.detection_match_threshold).any(1).logical_not());
  const auto threshold=o.use_iom?o.iom_recondition_threshold:o.iou_recondition_threshold;
  const auto many_tracks=metric.ge(threshold).sum(1).gt(1),many_detections=metric.ge(threshold).sum(0).gt(1);
  metric=at::where(many_detections.unsqueeze(0),at::zeros_like(metric),metric);
  metric=at::where(many_tracks.unsqueeze(1),at::zeros_like(metric),metric);
  return {unmatched.slice(0,0,m),nonempty.slice(0,0,m),is_new,metric.argmax(1),scores.ge(o.high_confidence_threshold).logical_and(keep).logical_and(is_new.logical_not()),metric.amax(1).ge(threshold),keep,metric.ge(o.detection_match_threshold).slice(1,0,m)};
}
AssociationMetadata realize_association(const AssociationTensors& value,const std::vector<int64_t>& ids,AssociationPolicy policy){
  c10::InferenceMode inference;const auto m=int64_t(ids.size()),n=value.is_new.numel();
  TORCH_CHECK(policy==AssociationPolicy::Sam3 || policy==AssociationPolicy::Sam31,"unknown association policy");
  const auto fields=value.tensors();const std::vector<std::vector<int64_t>> shapes{{m},{m},{n},{n},{n},{n},{n},{n,m}};
  std::vector<at::Tensor> cpu;for(size_t i=0;i<fields.size();++i){TORCH_CHECK(fields[i].sizes()==at::IntArrayRef(shapes[i]) && fields[i].scalar_type()==(i==3?at::kLong:at::kBool),"invalid association result layout");cpu.push_back(fields[i].cpu().contiguous());}
  AssociationMetadata out;
  for(int64_t t=0;t<m;++t){if(cpu[0].const_data_ptr<bool>()[t])out.unmatched_tracks.push_back(ids[t]);if(!cpu[1].const_data_ptr<bool>()[t])out.empty_tracks.push_back(ids[t]);}
  for(int64_t d=0;d<n;++d){
    if(cpu[2].const_data_ptr<bool>()[d])out.new_detections.push_back(d);
    if((policy==AssociationPolicy::Sam3 && m==0) || !cpu[6].const_data_ptr<bool>()[d])continue;
    auto& matches=out.detection_to_tracks[d];for(int64_t t=0;t<m;++t)if(cpu[7].const_data_ptr<bool>()[d*m+t])matches.push_back(ids[t]);
    if(cpu[4].const_data_ptr<bool>()[d] && cpu[5].const_data_ptr<bool>()[d]){const auto t=cpu[3].const_data_ptr<int64_t>()[d];TORCH_CHECK(t>=0 && t<m,"reconditioning track index is out of range");if(!out.track_to_recondition_detection.count(ids[t]))out.recondition_order.push_back(ids[t]);out.track_to_recondition_detection[ids[t]]=d;}
  }
  return out;
}
std::vector<int64_t> assign_detection_devices(int64_t count,const std::vector<int64_t>& previous,int64_t capacity){
  TORCH_CHECK(count>=0 && capacity>0 && (count==0 || !previous.empty()),"invalid placement inputs");auto workload=previous;
  for(auto value:workload)TORCH_CHECK(value>=0,"negative device workload");
  std::vector<int64_t> devices(count);for(int64_t start=0;start<count;){auto it=std::min_element(workload.begin(),workload.end());TORCH_CHECK(*it<INT64_MAX,"device workload overflow");const auto end=start+std::min(capacity,count-start);std::fill(devices.begin()+start,devices.begin()+end,it-workload.begin());++*it;start=end;}return devices;
}
at::Tensor detection_boundary_keep(const at::Tensor& boxes,double margin){
  c10::InferenceMode inference;TORCH_CHECK(boxes.dim()==2 && boxes.size(1)==4 && boxes.is_floating_point() && std::isfinite(margin),"boxes must be floating [N,4] with finite margin");
  const auto x=(boxes.select(1,0)+boxes.select(1,2))/2,y=(boxes.select(1,1)+boxes.select(1,3))/2;
  return x.gt(margin).logical_and(x.lt(1.-margin)).logical_and(y.gt(margin)).logical_and(y.lt(1.-margin));
}
}
