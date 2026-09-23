#include "sam3/video_interaction.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
bool propagation(VideoActionType t){return t==VideoActionType::Full || t==VideoActionType::Partial || t==VideoActionType::Fetch || t==VideoActionType::Cancel;}
bool computed(VideoActionType t){return t==VideoActionType::Full || t==VideoActionType::Partial;}
void policy_check(AssociationPolicy p){TORCH_CHECK(p==AssociationPolicy::Sam3 || p==AssociationPolicy::Sam31,"invalid interaction policy");}
}
VideoActionRoute route_video_actions(const std::vector<VideoAction>& actions,int64_t n,AssociationPolicy policy){
 policy_check(policy);TORCH_CHECK(n>0,"frame count must be positive");if(actions.empty())return {VideoActionType::Full,{}};
 const auto& last=actions.back();
 if(policy==AssociationPolicy::Sam31 && last.type==VideoActionType::Cancel){
  if(actions.size()==1)return {VideoActionType::Full,{}};size_t index=actions.size()-2;
  if(actions[index].type==VideoActionType::Fetch){TORCH_CHECK(index>0,"cancel after fetch requires a preceding propagation");--index;}
  return {actions[index].type,actions[index].ids};
 }
 if(last.type==VideoActionType::Fetch)return {VideoActionType::Fetch,{}};
 if(computed(last.type)){
  if((actions.size()>1 && computed(actions[actions.size()-2].type)) || (last.frame && (*last.frame==0 || *last.frame==n-1)))return {VideoActionType::Fetch,{}};
  return {last.type,last.ids};
 }
 std::set<int64_t> ids;for(auto it=actions.rbegin();it!=actions.rend();++it){if(propagation(it->type))break;if(it->type==VideoActionType::Add || it->type==VideoActionType::Refine){TORCH_CHECK(it->ids,"add/refine action needs object IDs");ids.insert(it->ids->begin(),it->ids->end());}}
 return ids.empty()?VideoActionRoute{VideoActionType::Fetch,{}}:VideoActionRoute{VideoActionType::Partial,std::vector<int64_t>(ids.begin(),ids.end())};
}
bool video_object_was_refined(const std::vector<VideoAction>& actions,int64_t id){for(const auto& a:actions)if((a.type==VideoActionType::Add || a.type==VideoActionType::Refine) && a.ids && std::find(a.ids->begin(),a.ids->end(),id)!=a.ids->end())return true;return false;}
VideoProcessingRange video_processing_range(int64_t n,const std::set<int64_t>& initialized,std::optional<int64_t> start,std::optional<int64_t> steps,bool reverse){
 TORCH_CHECK(n>0 && (!steps || *steps>=0),"invalid propagation bounds");if(!start){TORCH_CHECK(!initialized.empty(),"add a prompt before propagation");start=*initialized.begin();}
 TORCH_CHECK(*start>=0 && *start<n,"propagation start outside video");const auto count=steps.value_or(n);
 if(reverse){const auto end=*start-std::min(*start,count);return {*start-1,end,-1,*start-1<end};}
 return {*start,*start+std::min(n-1-*start,count),1,false};
}
VideoInteractionState::VideoInteractionState(AssociationPolicy policy,int64_t frames,int64_t h,int64_t w):policy_(policy),frames_(frames),height_(h),width_(w){policy_check(policy);TORCH_CHECK(frames>0 && h>0 && w>0,"invalid interaction dimensions");}
void VideoInteractionState::check_frame(int64_t frame)const{TORCH_CHECK(frame>=0 && frame<frames_,"frame outside video");}
void VideoInteractionState::append(const VideoAction& action){
 TORCH_CHECK(int(action.type)>=int(VideoActionType::Add) && int(action.type)<=int(VideoActionType::Cancel),"invalid video action");TORCH_CHECK(policy_==AssociationPolicy::Sam31 || action.type!=VideoActionType::Cancel,"SAM3 source action history has no cancel event; cancel the active generator/session instead");
 if(action.frame)check_frame(*action.frame);if(action.type==VideoActionType::Add || action.type==VideoActionType::Refine)TORCH_CHECK(action.ids,"add/refine action needs IDs");actions_.push_back(action);
}
VideoActionRoute VideoInteractionState::route(const std::vector<int64_t>& ids,bool force)const{if(force){TORCH_CHECK(policy_==AssociationPolicy::Sam3,"forced tracker route is a SAM3 extension");return ids.empty()?VideoActionRoute{VideoActionType::Full,{}}:VideoActionRoute{VideoActionType::Partial,ids};}return route_video_actions(actions_,frames_,policy_);}
void VideoInteractionState::record(int64_t frame,const VideoOutput& output){check_frame(frame);auto& masks=cache_[frame];masks.clear();for(const auto& [id,x]:output.cached_masks)masks.emplace(id,x.clone());}
VideoRawOutput VideoInteractionState::raw(int64_t frame,const VideoMetadata& metadata,const std::set<int64_t>& suppressed)const{
 check_frame(frame);VideoRawOutput out;out.frame=frame;const auto found=cache_.find(frame);if(found!=cache_.end())out.masks=found->second;out.scores=metadata.object_scores;const auto scores=metadata.frame_scores.find(frame);if(scores!=metadata.frame_scores.end())out.tracker_scores=scores->second;out.suppressed=suppressed;return out;
}
VideoOutput VideoInteractionState::fetch(int64_t frame,const VideoMetadata& metadata,const std::set<int64_t>& suppressed)const{if(policy_==AssociationPolicy::Sam31)TORCH_CHECK(cache_.count(frame),"SAM3.1 fetch requires a cached frame");return postprocess_video_output(raw(frame,metadata,suppressed),height_,width_);}
VideoOutput VideoInteractionState::merge_refined(int64_t frame,const RefinedVideoObjects& refined,VideoMetadata& metadata,const std::set<int64_t>& suppressed){
 c10::InferenceMode inference;check_frame(frame);auto input=raw(frame,metadata,suppressed);const bool merge=policy_==AssociationPolicy::Sam3 || cache_.count(frame);
 for(const auto& [id,pair]:refined){const auto& score=pair.first;const auto& low=pair.second;TORCH_CHECK(low.dim()==2 && low.is_floating_point() && score.numel()==1 && score.is_floating_point(),"refined object needs a scalar score and floating [H,W] mask");
  input.tracker_scores[id]=(policy_==AssociationPolicy::Sam3?score.to(at::kFloat):score).reshape({}).clone();
  if(merge)input.masks[id]=at::upsample_bilinear2d(low.to(at::kFloat).unsqueeze(0).unsqueeze(0),{height_,width_},false).squeeze(0).gt(0);
 }
 auto output=postprocess_video_output(input,height_,width_);for(const auto& [id,pair]:refined)metadata.frame_scores[frame][id]=input.tracker_scores.at(id);record(frame,output);return output;
}
void VideoInteractionState::forget_object(int64_t id){for(auto& [frame,masks]:cache_)masks.erase(id);}
void VideoInteractionState::reset(){actions_.clear();cache_.clear();}
namespace {
template<class Session> RefinedVideoObjects propagate(int64_t frame,bool reverse,const std::vector<int64_t>& ids,const std::vector<Session*>& sessions,int64_t cleanup,bool preflight){
 c10::InferenceMode inference;TORCH_CHECK(frame>=0 && cleanup>=0,"invalid partial propagation options");const std::set<int64_t> requested(ids.begin(),ids.end());std::set<Session*> seen;std::set<int64_t> owners;RefinedVideoObjects result;
 for(auto* session:sessions){TORCH_CHECK(session && seen.insert(session).second,"sessions must be non-null and distinct");const auto local=session->object_ids();bool selected=false;for(auto id:local)if(requested.count(id)){TORCH_CHECK(owners.insert(id).second,"requested object has multiple local owners");selected=true;}if(!selected)continue;
  TrackingPropagation request;request.start=frame;request.max_steps=0;request.reverse=reverse;request.encode_memory=true;request.preflight=preflight;
  session->propagate(request,[&](const TrackingSessionOutput& value){const auto low=clean_video_mask_scores(value.low_masks,cleanup).squeeze(1);for(size_t i=0;i<value.object_ids.size();++i)if(requested.count(value.object_ids[i]))result.emplace(value.object_ids[i],std::make_pair(value.object_logits[i].reshape({}),low[i]));return true;});
 }return result;
}
}
RefinedVideoObjects propagate_video_refinements(int64_t frame,bool reverse,const std::vector<int64_t>& ids,const std::vector<Sam3TrackingSession*>& sessions,int64_t cleanup,bool preflight){return propagate(frame,reverse,ids,sessions,cleanup,preflight);}
RefinedVideoObjects propagate_video_refinements(int64_t frame,bool reverse,const std::vector<int64_t>& ids,const std::vector<Sam31TrackingSession*>& sessions,int64_t cleanup,bool preflight){return propagate(frame,reverse,ids,sessions,cleanup,preflight);}
}
