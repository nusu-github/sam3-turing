#include "sam3/hotstart.h"
#include "sam3/autocast.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <limits>
namespace sam3 {
namespace {
int64_t boundary(int64_t frame,bool reverse,const HotstartOptions& o){
  TORCH_CHECK(frame>=0 && o.delay>=0 && o.unmatched_threshold>=0 && o.duplicate_threshold>=0 && o.min_keep_alive<=o.max_keep_alive,"invalid hotstart options/frame");
  TORCH_CHECK(!reverse || frame<=INT64_MAX-o.delay,"hotstart frame boundary overflow");return reverse?frame+o.delay:frame-o.delay;
}
void validate(const DeviceHotstartState& s){
  TORCH_CHECK(s.first_frame.defined() && s.first_frame.dim()==1,"hotstart first_frame must be int64[N]");const auto n=s.size();const auto device=s.first_frame.device();
  TORCH_CHECK(device.is_cpu() || device.is_cuda(),"hotstart supports CPU/CUDA");
  for(const auto& x:{s.first_frame,s.unmatched_count,s.keep_alive,s.last_occluded})TORCH_CHECK(x.defined() && x.sizes()==at::IntArrayRef{n} && x.scalar_type()==at::kLong && x.device()==device,"invalid hotstart integer vector");
  TORCH_CHECK(s.removed.defined() && s.removed.sizes()==at::IntArrayRef{n} && s.removed.scalar_type()==at::kBool && s.removed.device()==device,"invalid removed vector");
  TORCH_CHECK(s.overlap_count.defined() && s.overlap_count.sizes()==at::IntArrayRef({n,n}) && s.overlap_count.scalar_type()==at::kLong && s.overlap_count.device()==device,"invalid overlap count matrix");
}
at::Tensor overlap_increment(const at::Tensor& matches){
  // Every product is binary. Up to 2^24 rows, every partial sum is an exactly
  // representable FP32 integer, so FP32 A^T A equals the source outer-product
  // sum. Disable neural autocast: FP16 counts overflow and BF16 loses integers.
  // Beyond that exact-integer range, retain the source reduction order.
  AutocastGuard arithmetic(matches.device().type(),false,at::kFloat);
  const auto multi=matches.logical_and(matches.sum(1).gt(1).unsqueeze(1)).to(at::kFloat);
  const auto counts=matches.size(0)<=16777216?at::mm(multi.t(),multi):at::einsum("di,dj->dij",{multi,multi}).sum(0);
  return counts.triu(1).to(at::kLong);
}
}
HostHotstartResult update_host_hotstart(const HostHotstartState& previous,const AssociationMetadata& a,const std::vector<int64_t>& new_ids,int64_t frame,bool reverse,const HotstartOptions& o){
  const auto edge=boundary(frame,reverse,o);HostHotstartResult out{previous,{}};auto& s=out.state;auto& suppress=s.suppressed[frame];
  const auto within=[&](int64_t id){return reverse?s.first_frame.at(id)<edge:s.first_frame.at(id)>edge;};
  for(auto id:new_ids){s.first_frame.emplace(id,frame);TORCH_CHECK(!s.keep_alive.count(id),"new hotstart ID already has a keep-alive counter");s.keep_alive[id]=o.initial_keep_alive;}
  std::set<int64_t> matched;for(const auto& [det,ids]:a.detection_to_tracks)matched.insert(ids.begin(),ids.end());
  for(auto id:matched){auto& value=s.keep_alive.at(id);value=value>=o.max_keep_alive?o.max_keep_alive:value+1;}
  const auto decrease=[&](int64_t id){auto& value=s.keep_alive.at(id);value=value<=o.min_keep_alive?o.min_keep_alive:value-1;};
  for(auto id:a.unmatched_tracks){s.unmatched_frames[id].push_back(frame);decrease(id);}
  if(o.decrease_for_empty)for(auto id:a.empty_tracks)decrease(id);
  for(const auto& [id,frames]:s.unmatched_frames){
    if(s.removed.count(id) || out.newly_removed.count(id))continue;
    if(frames.size()>=uint64_t(o.unmatched_threshold) && within(id))out.newly_removed.insert(id);
    if(s.keep_alive.at(id)<=0 && !o.suppress_only_within_hotstart && !s.removed.count(id) && !out.newly_removed.count(id))suppress.insert(id);
  }
  for(const auto& [det,ids]:a.detection_to_tracks){
    if(ids.size()<2)continue;auto first=ids.front();
    for(auto id:ids)if(reverse?s.first_frame.at(id)>s.first_frame.at(first):s.first_frame.at(id)<s.first_frame.at(first))first=id;
    for(auto id:ids)if(id!=first)s.overlap_frames[{first,id}].push_back(frame);
  }
  for(const auto& [pair,frames]:s.overlap_frames){const auto id=pair.second;if(s.removed.count(id) || out.newly_removed.count(id))continue;if(within(id) && frames.size()>=uint64_t(o.duplicate_threshold))out.newly_removed.insert(id);}
  s.removed.insert(out.newly_removed.begin(),out.newly_removed.end());return out;
}
DeviceHotstartState empty_device_hotstart(at::Device device){
  c10::InferenceMode inference;TORCH_CHECK(device.is_cpu() || device.is_cuda(),"hotstart supports CPU/CUDA");const auto opts=at::TensorOptions().device(device).dtype(at::kLong);
  return {at::empty({0},opts),at::empty({0},opts),at::empty({0},opts),at::empty({0},opts.dtype(at::kBool)),at::empty({0,0},opts),at::empty({0},opts)};
}
DeviceHotstartResult update_device_hotstart(const DeviceHotstartState& previous,const AssociationTensors& a,int64_t frame,bool reverse,const HotstartOptions& o){
  c10::InferenceMode inference;const auto edge=boundary(frame,reverse,o);validate(previous);const auto n=previous.size();const auto device=previous.first_frame.device();
  for(const auto& x:{a.unmatched,a.nonempty})TORCH_CHECK(x.defined() && x.sizes()==at::IntArrayRef{n} && x.scalar_type()==at::kBool && x.device()==device,"association track state must match hotstart state");
  TORCH_CHECK(a.matches.defined() && a.matches.dim()==2 && a.matches.size(1)==n && a.matches.scalar_type()==at::kBool && a.matches.device()==device,"association match matrix must match hotstart state");
  auto s=previous;const auto matched=a.matches.any(0);
  s.keep_alive=at::where(matched,previous.keep_alive+1,previous.keep_alive-1).clamp(o.min_keep_alive,o.max_keep_alive);
  if(o.decrease_for_empty)s.keep_alive=at::where(a.nonempty.logical_not(),(s.keep_alive-1).clamp_min(o.min_keep_alive),s.keep_alive);
  s.unmatched_count=at::where(a.unmatched,previous.unmatched_count+1,previous.unmatched_count);
  s.overlap_count=previous.overlap_count+overlap_increment(a.matches);
  const auto within=reverse?s.first_frame.lt(edge):s.first_frame.gt(edge),alive=previous.removed.logical_not();
  auto remove=within.logical_and(s.unmatched_count.ge(o.unmatched_threshold)).logical_and(alive);
  const auto suppress=s.keep_alive.le(0).logical_and(at::scalar_tensor(!o.suppress_only_within_hotstart,previous.removed.options())).logical_and(alive).logical_and(remove.logical_not());
  if(n){const auto earlier=reverse?s.first_frame.unsqueeze(1).gt(s.first_frame.unsqueeze(0)):s.first_frame.unsqueeze(1).lt(s.first_frame.unsqueeze(0));const auto overlap=at::where(earlier,s.overlap_count,at::zeros_like(s.overlap_count)).amax(0);remove=remove.logical_or(within.logical_and(overlap.ge(o.duplicate_threshold)).logical_and(alive));}
  s.removed=previous.removed.logical_or(remove);return {std::move(s),remove,suppress};
}
DeviceHotstartState select_device_hotstart(const DeviceHotstartState& s,const at::Tensor& indices){
  c10::InferenceMode inference;validate(s);TORCH_CHECK(indices.dim()==1 && indices.scalar_type()==at::kLong && indices.device()==s.first_frame.device(),"selection must be int64[K] on the state device");
  return {s.first_frame.index_select(0,indices),s.unmatched_count.index_select(0,indices),s.keep_alive.index_select(0,indices),s.removed.index_select(0,indices),s.overlap_count.index_select(0,indices).index_select(1,indices),s.last_occluded.index_select(0,indices)};
}
std::pair<DeviceHotstartState,at::Tensor> compact_device_hotstart(const DeviceHotstartState& s){c10::InferenceMode inference;validate(s);const auto indices=at::nonzero(s.removed.logical_not()).squeeze(1);return {select_device_hotstart(s,indices),indices};}
DeviceHotstartState extend_device_hotstart(const DeviceHotstartState& s,int64_t count,int64_t frame,int64_t initial){
  c10::InferenceMode inference;validate(s);TORCH_CHECK(count>=0 && frame>=0 && count<=INT64_MAX-s.size(),"invalid hotstart extension");if(!count)return s;const auto old=s.size(),n=old+count;const auto opts=s.first_frame.options();
  auto overlap=at::zeros({n,n},opts);overlap.slice(0,0,old).slice(1,0,old).copy_(s.overlap_count);
  return {at::cat({s.first_frame,at::full({count},frame,opts)}),at::cat({s.unmatched_count,at::zeros({count},opts)}),at::cat({s.keep_alive,at::full({count},initial,opts)}),at::cat({s.removed,at::zeros({count},opts.dtype(at::kBool))}),overlap,at::cat({s.last_occluded,at::full({count},-1,opts)})};
}
ConfirmationState update_confirmation(const ConfirmationState& previous,const std::vector<int64_t>& old_ids,const std::vector<int64_t>& ids,const std::map<int64_t,std::vector<int64_t>>& matches,const std::vector<int64_t>& new_ids,int64_t threshold){
  TORCH_CHECK(previous.status.size()==old_ids.size() && previous.consecutive_detections.size()==old_ids.size(),"confirmation state/ID size mismatch");ConfirmationState out{std::vector<int64_t>(ids.size(),1),std::vector<int64_t>(ids.size(),0)};
  std::map<int64_t,size_t> index;for(size_t i=0;i<ids.size();++i)index[ids[i]]=i;
  for(size_t i=0;i<old_ids.size();++i){const auto it=index.find(old_ids[i]);if(it!=index.end()){out.status[it->second]=previous.status[i];out.consecutive_detections[it->second]=previous.consecutive_detections[i];}}
  std::set<int64_t> matched(new_ids.begin(),new_ids.end());for(const auto& [det,values]:matches)matched.insert(values.begin(),values.end());
  for(size_t i=0;i<ids.size();++i){auto& count=out.consecutive_detections[i];if(matched.count(ids[i])){TORCH_CHECK(count<INT64_MAX,"confirmation counter overflow");++count;}else count=0;if(count>=threshold)out.status[i]=2;}
  return out;
}
}
