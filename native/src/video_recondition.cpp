#include "sam3/video_recondition.h"
#include <algorithm>
namespace sam3 {
namespace {
template<class Session> std::vector<std::vector<int64_t>> state_ids(const std::vector<Session*>& sessions){
  std::set<Session*> seen;std::vector<std::vector<int64_t>> ids;
  for(auto* session:sessions){TORCH_CHECK(session && seen.insert(session).second,"sessions must be non-null and distinct");ids.push_back(session->object_ids());}
  return ids;
}
void edited(ReconditionExecution& result,int64_t state,const std::vector<int64_t>& ids){
  if(std::find(result.edited_states.begin(),result.edited_states.end(),state)==result.edited_states.end())result.edited_states.push_back(state);
  result.affected_ids.insert(ids.begin(),ids.end());
}
void validate(const ReconditionMasks& masks){
  TORCH_CHECK(masks.binary_masks.defined() && masks.binary_masks.dim()==3 && masks.binary_masks.scalar_type()==at::kBool && masks.binary_masks.size(0)==int64_t(masks.ids.size()),"prepared edits require bool [N,H,W] masks");
  TORCH_CHECK(std::set<int64_t>(masks.ids.begin(),masks.ids.end()).size()==masks.ids.size(),"prepared object IDs must be unique");
}
}
ReconditionExecution execute_reconditioning(int64_t frame,const ReconditionMasks& masks,const std::vector<Sam3TrackingSession*>& sessions){
  validate(masks);const auto ids=state_ids(sessions);ReconditionExecution result;
  for(size_t i=0;i<masks.ids.size();++i){
    std::vector<int64_t> affected;
    for(size_t s=0;s<sessions.size();++s)if(std::find(ids[s].begin(),ids[s].end(),masks.ids[i])!=ids[s].end()){
      sessions[s]->add_mask(frame,masks.ids[i],masks.binary_masks[i]);affected.push_back(s);edited(result,s,ids[s]);
    }
    for(auto s:affected){sessions[s]->preflight(true);result.preflight_states.push_back(s);}
  }
  return result;
}
ReconditionExecution execute_reconditioning(int64_t frame,const ReconditionMasks& masks,const std::vector<Sam31TrackingSession*>& sessions){
  validate(masks);const auto ids=state_ids(sessions);const auto batches=recondition_batches(masks,ids,AssociationPolicy::Sam31,true);ReconditionExecution result;
  for(const auto& batch:batches){sessions[batch.state_index]->recondition_masks(frame,batch.ids,batch.masks);edited(result,batch.state_index,ids[batch.state_index]);}
  for(size_t s=0;s<sessions.size();++s)if(std::any_of(ids[s].begin(),ids[s].end(),[&](auto id){return result.affected_ids.count(id);})){
    sessions[s]->preflight(true);result.preflight_states.push_back(s);
  }
  return result;
}
}
