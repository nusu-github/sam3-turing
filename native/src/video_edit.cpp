#include "sam3/video_edit.h"
#include "sam3/video_collective.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
std::optional<int64_t> owner(const VideoMetadata& metadata,int64_t id){std::optional<int64_t> out;for(size_t r=0;r<metadata.ids_per_rank.size();++r)for(auto value:metadata.ids_per_rank[r])if(value==id){TORCH_CHECK(!out,"duplicate metadata ID");out=r;}return out;}
template<class Session> Session* local_session(const std::vector<std::unique_ptr<Session>>& sessions,int64_t id){Session* out=nullptr;for(const auto& s:sessions){TORCH_CHECK(s,"null tracking session");const auto ids=s->object_ids();if(std::find(ids.begin(),ids.end(),id)!=ids.end()){TORCH_CHECK(!out,"multiple local owners");out=s.get();}}return out;}
void validate_points(const TrackingPoints& points){
 TORCH_CHECK(points.points.defined()==points.labels.defined() && (points.points.defined() || points.box.defined()),"points require labels or a box");
 if(points.points.defined()){const auto& p=points.points;const auto& l=points.labels;TORCH_CHECK((p.dim()==2 || (p.dim()==3 && p.size(0)==1)) && p.size(-1)==2 && ((l.dim()==1 && l.size(0)==p.size(-2)) || (l.dim()==2 && l.size(0)==1 && l.size(1)==p.size(-2))),"invalid point/label shape");}
 TORCH_CHECK(!points.box.defined() || points.box.numel()==4,"box must contain four coordinates");
}
void realign_confirmation(VideoMetadata& metadata,const std::vector<int64_t>& old){
 if(metadata.confirmation.status.empty())return;
 TORCH_CHECK(metadata.confirmation.status.size()==old.size() && metadata.confirmation.consecutive_detections.size()==old.size(),"misaligned confirmation metadata");ConfirmationState next;
 for(auto id:metadata.object_ids()){const auto it=std::find(old.begin(),old.end(),id);if(it==old.end()){next.status.push_back(1);next.consecutive_detections.push_back(0);}else{const auto i=it-old.begin();next.status.push_back(metadata.confirmation.status[i]);next.consecutive_detections.push_back(metadata.confirmation.consecutive_detections[i]);}}
 metadata.confirmation=std::move(next);
}
void assert_object(VideoMetadata& metadata,VideoSuppressionHistory& suppressions,int64_t id,int64_t frame,bool mask,int64_t threshold){
 metadata.object_scores[id]=1.;metadata.frame_scores[frame][id]=at::scalar_tensor(1.,at::kFloat);metadata.host.removed.erase(id);for(auto& [f,ids]:metadata.host.suppressed)ids.erase(id);for(auto& [f,ids]:suppressions)ids.erase(id);
 if(!metadata.confirmation.status.empty()){const auto ids=metadata.object_ids();const auto index=std::find(ids.begin(),ids.end(),id)-ids.begin();TORCH_CHECK(index<int64_t(metadata.confirmation.status.size()),"missing edited object confirmation");metadata.confirmation.status[index]=mask?2:1;metadata.confirmation.consecutive_detections[index]=threshold;}
}
void clear_detector_conditions(Sam3TrackingSession& session,int64_t id,int64_t frame,int64_t window){
 std::vector<int64_t> frames;for(const auto& object:session.state().objects)if(object.id==id)for(const auto& [f,mask]:object.masks)if(!object.points.count(f) && std::max(f,frame)-std::min(f,frame)<=window)frames.push_back(f);
 const auto ids=session.object_ids();for(auto f:frames)for(auto other:ids)session.clear_input(f,other);
}
template<class Edit> VideoOutput edit(int64_t frame,int64_t id,bool mask,Edit operation,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options,std::optional<at::Device> target={}){
 c10::InferenceMode inference;TORCH_CHECK(interaction.policy()==AssociationPolicy::Sam3 && frame>=0 && frame<interaction.frame_count() && options.rank>=0 && options.rank<int64_t(metadata.ids_per_rank.size()) && options.cleanup_area>=0 && options.conditioning_window>=0 && options.confirmation_threshold>0 && factory,"invalid video edit options");
 auto rank=owner(metadata,id);TORCH_CHECK(!rank || *rank==options.rank,"edit must execute on the object's owning rank");
 if(!mask && rank && options.stateless_refinement && !video_object_was_refined(interaction.actions(),id)){remove_video_user_object(id,sessions,metadata,interaction,false);rank.reset();}
 auto* session=local_session(sessions,id);TORCH_CHECK(bool(session)==bool(rank),"session and metadata ownership disagree");std::unique_ptr<Sam3TrackingSession> created;
 if(!session){created=factory();TORCH_CHECK(created && created->object_ids().empty(),"factory must return an empty session");session=created.get();}
 const auto preview=operation(*session);session->preflight(true);if(!mask)clear_detector_conditions(*session,id,frame,options.conditioning_window);
 const auto position=std::find(preview.object_ids.begin(),preview.object_ids.end(),id);TORCH_CHECK(position!=preview.object_ids.end(),"edit preview omitted requested ID");auto video=preview.masks;
 if(!mask)video=clean_video_mask_scores(video,options.cleanup_area);
 auto selected=video[position-preview.object_ids.begin()].squeeze(0).gt(0).to(at::kFloat);
 if(target && selected.device()!=*target)selected=transfer_video_tensor(selected,*target);
 if(created){const auto old=metadata.object_ids();metadata.ids_per_rank[options.rank].push_back(id);metadata.max_id=std::max(metadata.max_id,id);realign_confirmation(metadata,old);sessions.push_back(std::move(created));}
 assert_object(metadata,suppressions,id,frame,mask,options.confirmation_threshold);interaction.append({rank?VideoActionType::Refine:VideoActionType::Add,frame,std::vector<int64_t>{id}});
 return interaction.merge_refined(frame,{{id,{at::scalar_tensor(1.,selected.options()),selected}}},metadata,suppressions[frame]);
}
}
void remove_video_user_object(int64_t id,Sam3VideoSessions& sessions,VideoMetadata& metadata,VideoInteractionState& interaction,bool record){
 TORCH_CHECK(owner(metadata,id),"unknown video object");const auto old=metadata.object_ids();remove_video_objects({id},sessions);for(auto& ids:metadata.ids_per_rank)ids.erase(std::remove(ids.begin(),ids.end(),id),ids.end());metadata.object_scores.erase(id);realign_confirmation(metadata,old);interaction.forget_object(id);if(record)interaction.append({VideoActionType::Remove,{},std::vector<int64_t>{id}});
}
VideoOutput edit_video_points(int64_t frame,int64_t id,const TrackingPoints& points,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options){
 validate_points(points);
 return edit(frame,id,false,[&](auto& s){return s.add_points(frame,id,points,true,options.use_previous_memory);},sessions,factory,metadata,interaction,suppressions,options);}
VideoOutput edit_video_mask(int64_t frame,int64_t id,const at::Tensor& mask,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options){TORCH_CHECK(mask.defined() && mask.dim()==2 && mask.numel()>0,"mask must be nonempty [H,W]");return edit(frame,id,true,[&](auto& s){return s.add_mask(frame,id,mask);},sessions,factory,metadata,interaction,suppressions,options);}
void remove_video_user_object(int64_t id,Sam31VideoSessions& sessions,VideoMetadata& metadata,VideoInteractionState& interaction,bool record){
 const auto rank=owner(metadata,id);
 if(rank){const auto old=metadata.object_ids();remove_video_objects({id},sessions);for(auto& ids:metadata.ids_per_rank)ids.erase(std::remove(ids.begin(),ids.end(),id),ids.end());metadata.object_scores.erase(id);realign_confirmation(metadata,old);
   int64_t buckets=0;for(const auto& s:sessions)if(s->state().buckets)buckets+=s->state().buckets->bucket_count();metadata.buckets_per_rank[*rank]=buckets;}
 // The source accepts objects already removed by an earlier user action or
 // hotstart: still forget cached masks and record the requested remove action.
 interaction.forget_object(id);
 if(record)interaction.append({VideoActionType::Remove,{},std::vector<int64_t>{id}});
}
namespace {
template<class Edit> VideoOutput edit31(int64_t frame,int64_t id,bool mask,Edit operation,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options,std::optional<at::Device> target={}){
 c10::InferenceMode inference;
 TORCH_CHECK(interaction.policy()==AssociationPolicy::Sam31 && frame>=0 && frame<interaction.frame_count() && options.rank>=0 && options.rank<int64_t(metadata.ids_per_rank.size()) && options.cleanup_area>=0 && options.confirmation_threshold>0 && factory,"invalid SAM3.1 video edit options");
 auto rank=owner(metadata,id);TORCH_CHECK(!rank || *rank==options.rank,"edit must execute on the object's owning rank");
 if(!mask && rank && options.stateless_refinement && !video_object_was_refined(interaction.actions(),id)){remove_video_user_object(id,sessions,metadata,interaction,false);rank.reset();}
 auto* session=local_session(sessions,id);TORCH_CHECK(bool(session)==bool(rank),"session and metadata ownership disagree");
 if(!mask && session && !video_object_was_refined(interaction.actions(),id) && session->object_ids().size()>1){auto extracted=session->extract_object(id);session=extracted.get();sessions.push_back(std::move(extracted));}
 std::unique_ptr<Sam31TrackingSession> created;if(!session){created=factory();TORCH_CHECK(created && created->object_ids().empty(),"factory must return an empty session");session=created.get();}
 const auto preview=operation(*session);if(!mask && !options.empty_points_mask.defined())session->discard_mask_only_inputs();session->preflight(true);
 const auto position=std::find(preview.object_ids.begin(),preview.object_ids.end(),id);TORCH_CHECK(position!=preview.object_ids.end(),"edit preview omitted requested ID");
 const auto scores=mask?preview.masks:clean_video_mask_scores(preview.masks,options.cleanup_area);
 auto selected=scores[position-preview.object_ids.begin()].squeeze(0).gt(0).to(at::kFloat);
 if(target && selected.device()!=*target)selected=transfer_video_tensor(selected,*target);
 if(created){const auto old=metadata.object_ids();metadata.ids_per_rank[options.rank].push_back(id);metadata.max_id=std::max(metadata.max_id,id);realign_confirmation(metadata,old);sessions.push_back(std::move(created));}
 int64_t buckets=0;for(const auto& s:sessions)if(s->state().buckets)buckets+=s->state().buckets->bucket_count();metadata.buckets_per_rank[options.rank]=buckets;
 assert_object(metadata,suppressions,id,frame,mask,options.confirmation_threshold);interaction.append({rank?VideoActionType::Refine:VideoActionType::Add,frame,std::vector<int64_t>{id}});
 return interaction.merge_refined(frame,{{id,{at::scalar_tensor(1.,selected.options()),selected}}},metadata,suppressions[frame]);
}
} // namespace
VideoOutput edit_video_points(int64_t frame,int64_t id,const TrackingPoints& points,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options){
 validate_points(points);
 if(options.empty_points_mask.defined())TORCH_CHECK(points.points.defined() && points.points.numel()==0 && !points.box.defined() && options.empty_points_mask.dim()==2 && options.empty_points_mask.numel()>0,"image restoration requires empty points and a nonempty [H,W] mask");
 return edit31(frame,id,false,[&](auto& session){if(options.empty_points_mask.defined()){const auto ids=session.object_ids();if(std::find(ids.begin(),ids.end(),id)!=ids.end())session.clear_input(frame,id);return session.add_mask(frame,id,options.empty_points_mask);}return session.add_points(frame,id,points,options.clear_old_points,options.use_previous_memory);},sessions,factory,metadata,interaction,suppressions,options);
}
VideoOutput edit_video_mask(int64_t frame,int64_t id,const at::Tensor& mask,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options){
 TORCH_CHECK(mask.defined() && mask.dim()==2 && mask.numel()>0,"mask must be nonempty [H,W]");
 return edit31(frame,id,true,[&](auto& session){return session.add_mask(frame,id,mask);},sessions,factory,metadata,interaction,suppressions,options);
}
VideoOutput edit_video_points(int64_t frame,int64_t id,const TrackingPoints& points,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options,at::Device target){
 validate_points(points);
 return edit(frame,id,false,[&](auto& s){return s.add_points(frame,id,points,true,options.use_previous_memory);},sessions,factory,metadata,interaction,suppressions,options,target);}
VideoOutput edit_video_mask(int64_t frame,int64_t id,const at::Tensor& mask,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options,at::Device target){TORCH_CHECK(mask.defined() && mask.dim()==2 && mask.numel()>0,"mask must be nonempty [H,W]");return edit(frame,id,true,[&](auto& s){return s.add_mask(frame,id,mask);},sessions,factory,metadata,interaction,suppressions,options,target);}
VideoOutput edit_video_points(int64_t frame,int64_t id,const TrackingPoints& points,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options,at::Device target){
 validate_points(points);
 if(options.empty_points_mask.defined())TORCH_CHECK(points.points.defined() && points.points.numel()==0 && !points.box.defined() && options.empty_points_mask.dim()==2 && options.empty_points_mask.numel()>0,"image restoration requires empty points and a nonempty [H,W] mask");
 return edit31(frame,id,false,[&](auto& session){if(options.empty_points_mask.defined()){const auto ids=session.object_ids();if(std::find(ids.begin(),ids.end(),id)!=ids.end())session.clear_input(frame,id);return session.add_mask(frame,id,options.empty_points_mask);}return session.add_points(frame,id,points,options.clear_old_points,options.use_previous_memory);},sessions,factory,metadata,interaction,suppressions,options,target);
}
VideoOutput edit_video_mask(int64_t frame,int64_t id,const at::Tensor& mask,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options,at::Device target){
 TORCH_CHECK(mask.defined() && mask.dim()==2 && mask.numel()>0,"mask must be nonempty [H,W]");
 return edit31(frame,id,true,[&](auto& session){return session.add_mask(frame,id,mask);},sessions,factory,metadata,interaction,suppressions,options,target);
}
}
