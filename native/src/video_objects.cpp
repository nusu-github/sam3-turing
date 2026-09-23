#include "sam3/video_objects.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <limits>
namespace sam3 {
int64_t video_object_destination(const std::vector<int64_t>& slots,int64_t count,VideoObjectPlacement policy){
  TORCH_CHECK(count>0,"placement requires a positive object count");
  TORCH_CHECK(policy==VideoObjectPlacement::BestFit || policy==VideoObjectPlacement::FirstState || policy==VideoObjectPlacement::NewState,"invalid placement policy");
  for(auto n:slots)TORCH_CHECK(n>=0,"available slots cannot be negative");
  if(policy==VideoObjectPlacement::NewState)return -1;
  if(policy==VideoObjectPlacement::FirstState)return slots.empty()?-1:0;
  int64_t best=-1,available=std::numeric_limits<int64_t>::max();
  for(size_t i=0;i<slots.size();++i)if(slots[i]>=count && (best<0 || slots[i]<available)){best=i;available=slots[i];}
  return best;
}
at::Tensor prepare_video_object_masks(const at::Tensor& logits){
  c10::InferenceMode inference;
  TORCH_CHECK(logits.dim()==3 && logits.is_floating_point() && logits.size(1)>0 && logits.size(2)>0,"new object logits require floating [N,H,W]");
  if(!logits.size(0))return at::empty({0,1152,1152},logits.options().dtype(at::kBool));
  return at::upsample_bilinear2d(logits.unsqueeze(1),{1152,1152},false).squeeze(1).gt(0);
}
namespace {
template<class Session> void validate(const std::vector<std::unique_ptr<Session>>& sessions){for(const auto& session:sessions)TORCH_CHECK(session,"null video session");}
template<class Session> void validate_add(const std::vector<int64_t>& ids,const at::Tensor& logits,const std::vector<std::unique_ptr<Session>>& sessions){
  validate(sessions);TORCH_CHECK(logits.dim()==3 && logits.size(0)==int64_t(ids.size()),"new object mask/ID count mismatch");
  std::set<int64_t> used;for(const auto& session:sessions)for(auto id:session->object_ids())used.insert(id);
  for(auto id:ids)TORCH_CHECK(used.insert(id).second,"new object IDs must be unique and absent from existing sessions");
}
template<class Session> void prune(std::vector<std::unique_ptr<Session>>& sessions){sessions.erase(std::remove_if(sessions.begin(),sessions.end(),[](const auto& session){return session->object_ids().empty();}),sessions.end());}
}
int64_t add_video_objects(int64_t frame,const std::vector<int64_t>& ids,const at::Tensor& logits,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory){
  validate_add(ids,logits,sessions);const auto masks=prepare_video_object_masks(logits);if(ids.empty())return -1;
  TORCH_CHECK(factory,"new video session requires a factory");auto session=factory();TORCH_CHECK(session && session->object_ids().empty(),"factory must return an empty video session");
  for(size_t i=0;i<ids.size();++i)session->add_mask(frame,ids[i],masks[i]);session->preflight(true);
  const auto index=int64_t(sessions.size());sessions.push_back(std::move(session));return index;
}
int64_t add_video_objects(int64_t frame,const std::vector<int64_t>& ids,const at::Tensor& logits,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoObjectPlacement policy){
  validate_add(ids,logits,sessions);const auto masks=prepare_video_object_masks(logits);if(ids.empty())return -1;
  std::vector<int64_t> slots;for(const auto& session:sessions)slots.push_back(session->state().buckets?session->state().buckets->available_slots():0);
  auto index=video_object_destination(slots,ids.size(),policy);
  if(index>=0){sessions[index]->add_masks(frame,ids,masks);sessions[index]->preflight(true);return index;}
  TORCH_CHECK(factory,"new video session requires a factory");auto session=factory();TORCH_CHECK(session && session->object_ids().empty(),"factory must return an empty video session");
  session->add_masks(frame,ids,masks);session->preflight(true);index=sessions.size();sessions.push_back(std::move(session));return index;
}
void remove_video_objects(const std::vector<int64_t>& ids,Sam3VideoSessions& sessions){
  validate(sessions);for(auto id:ids){for(auto& session:sessions)session->remove_object(id,false);prune(sessions);}
}
void remove_video_objects(const std::vector<int64_t>& ids,Sam31VideoSessions& sessions){
  validate(sessions);for(auto& session:sessions)session->remove_objects(ids,false);prune(sessions);
}
}
