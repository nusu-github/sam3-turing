#include "sam3/video_memory.h"
#include <c10/core/InferenceMode.h>
#include <map>
#include <set>
namespace sam3 {
VideoMemoryInputs prepare_video_memory(const at::Tensor& low,AssociationPolicy policy,bool warmup){
  c10::InferenceMode inference;
  TORCH_CHECK(low.dim()==3 && low.is_floating_point() && low.size(1)>0 && low.size(2)>0,"video memory masks require floating [N,H,W]");
  TORCH_CHECK(policy==AssociationPolicy::Sam3 || policy==AssociationPolicy::Sam31,"invalid memory policy");
  const auto n=low.size(0);
  if(!n)return {at::empty({0,1,1152,1152},low.options()),at::empty({0,1},low.options().dtype(at::kFloat))};
  auto high=at::upsample_bilinear2d(low.unsqueeze(1),{1152,1152},false);
  if((policy==AssociationPolicy::Sam31 || warmup) && !(policy==AssociationPolicy::Sam31 && n==1)){
    const auto foreground=high.gt(0);
    auto visible=foreground;
    if(n>1){const auto winner=high.argmax(0,true),ids=at::arange(n,low.options().dtype(at::kLong)).view({n,1,1,1});visible=foreground.logical_and(winner.eq(ids));}
    // Counts include only positive winners; no full float overlap image is needed.
    const auto before=foreground.sum(at::IntArrayRef{2,3}).clamp_min(1.),after=visible.sum(at::IntArrayRef{2,3});
    const auto keep=(after/before).ge(.3).unsqueeze(-1).unsqueeze(-1);
    high=at::where(keep,high,high.clamp_max(-10.));
  }
  const auto scores=at::where(high.gt(0).any(at::IntArrayRef{2,3}),10.,-10.);
  return {high,scores};
}
std::vector<std::vector<int64_t>> video_memory_rows(const std::vector<int64_t>& global,const std::vector<std::vector<int64_t>>& states){
  std::map<int64_t,int64_t> rows;for(size_t i=0;i<global.size();++i)TORCH_CHECK(rows.emplace(global[i],i).second,"global memory IDs must be unique");
  std::vector<std::vector<int64_t>> out;
  for(const auto& state:states){std::set<int64_t> seen;out.emplace_back();for(auto id:state){TORCH_CHECK(seen.insert(id).second && rows.count(id),"local memory IDs must be unique and present globally");out.back().push_back(rows.at(id));}}
  return out;
}
namespace {
template<class Session> auto plan(const at::Tensor& masks,const std::vector<int64_t>& global,const std::vector<Session*>& sessions){
  TORCH_CHECK(masks.dim()==3 && masks.size(0)==int64_t(global.size()),"global memory mask/ID count mismatch");
  std::vector<std::vector<int64_t>> ids;std::set<Session*> seen;for(auto* s:sessions){TORCH_CHECK(s && seen.insert(s).second,"local sessions must be distinct and non-null");ids.push_back(s->object_ids());}return video_memory_rows(global,ids);
}
}
void update_video_memories(int64_t frame,const at::Tensor& low,const std::vector<int64_t>& global,const std::vector<Sam3TrackingSession*>& sessions,bool warmup){
  const auto rows=plan(low,global,sessions);if(sessions.empty())return;const auto input=prepare_video_memory(low,AssociationPolicy::Sam3,warmup);
  for(size_t i=0;i<sessions.size();++i)if(!rows[i].empty()){const auto index=at::tensor(rows[i],low.options().dtype(at::kLong));sessions[i]->update_memory(frame,input.high_masks.index_select(0,index),input.object_logits.index_select(0,index));}
}
void update_video_memories(int64_t frame,const at::Tensor& low,const std::vector<int64_t>& global,const std::vector<Sam31TrackingSession*>& sessions,bool reapply){
  const auto rows=plan(low,global,sessions);if(sessions.empty())return;const auto input=prepare_video_memory(low,AssociationPolicy::Sam31);
  for(size_t i=0;i<sessions.size();++i)if(!rows[i].empty()){const auto index=at::tensor(rows[i],low.options().dtype(at::kLong));sessions[i]->update_memory(frame,input.high_masks.index_select(0,index),input.object_logits.index_select(0,index),reapply);}
}
}
