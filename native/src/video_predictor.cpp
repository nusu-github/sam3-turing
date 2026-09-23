#include "sam3/video_predictor.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
VideoOutput centers(VideoOutput out,int64_t h,int64_t w,bool enabled){
 if(enabled && !out.centers.defined()){const auto o=out.masks.options().dtype(at::kFloat);const auto y=at::arange(h,o).view({1,h,1}),x=at::arange(w,o).view({1,1,w});const auto mass=out.masks.sum(at::IntArrayRef{1,2}).clamp_min(1e-6);out.centers=at::stack({(out.masks*x).sum(at::IntArrayRef{1,2})/mass/w,(out.masks*y).sum(at::IntArrayRef{1,2})/mass/h},1);}return out;
}
}
VideoPredictorOptions video_predictor_defaults(AssociationPolicy policy){
 TORCH_CHECK(policy==AssociationPolicy::Sam3 || policy==AssociationPolicy::Sam31,"invalid video model");const bool mux=policy==AssociationPolicy::Sam31;VideoPredictorOptions o;o.model=policy;
 o.detection.model=policy;o.detection.nms=mux?VideoNmsMode::Sam31Batched:VideoNmsMode::Greedy;o.detection.score_threshold=mux?.4:.5;o.detection.use_iom=mux;o.detection.boundary_filter=mux;
 auto& u=o.update;u.association.policy=u.recondition.policy=policy;u.association.new_detection_threshold=mux?.65:.7;u.association.use_iom=mux;u.association.iom_recondition_threshold=mux?.5:.8;
 u.hotstart.delay=15;u.hotstart.unmatched_threshold=8;u.hotstart.duplicate_threshold=8;u.hotstart.suppress_only_within_hotstart=!mux;
 if(!mux){u.hotstart.min_keep_alive=-1;u.hotstart.max_keep_alive=30;u.hotstart.initial_keep_alive=30;}
 u.recondition.period=16;u.occlusion_threshold=.7;u.cleanup_area=mux?0:16;u.confirmation_enabled=mux;
 o.sam3_session.offload_state=true;o.sam3_session.frame.temporal.select_by_score=true;o.sam31_session.offload_state=true;o.sam31_session.all_edits_conditioning=false;o.output_batch_size=mux?16:1;return o;
}
struct VideoPredictor::Impl {
 WeightStore store;Tokenizer tokenizer;FrameProvider provider;int64_t frames,h,w,cached_index=-1,encodes=0;at::Device device;VideoPredictorOptions options;bool mux;
 std::shared_ptr<VisionEncoder> vision;std::shared_ptr<GroundingDetector> detector;std::shared_ptr<Sam3TrackingFrame> core3;std::shared_ptr<Sam31TrackingFrame> core31;std::unique_ptr<VideoFrameEncoder> encoder;
 Sam3VideoSessions sessions3;Sam31VideoSessions sessions31;Sam3SessionFactory factory3;Sam31SessionFactory factory31;VideoFrameFeatures cached;
 VideoMetadata metadata;VideoInteractionState interaction;VideoSuppressionHistory suppressions;std::set<int64_t> initialized;std::optional<int64_t> prompt_frame;
 VideoSemanticPrompt semantic;GroundingPrompt encoded;bool has_text=false;std::atomic<bool> cancelled{false};
 Impl(const WeightStore& s,const std::filesystem::path& vocabulary,FrameProvider f,int64_t n,int64_t height,int64_t width,at::Device d,const VideoPredictorOptions& o)
  :store(s),tokenizer(vocabulary),provider(std::move(f)),frames(n),h(height),w(width),device(d),options(o),mux(o.model==AssociationPolicy::Sam31),metadata(initialize_video_metadata(1,d)),interaction(o.model,n,height,width){
  TORCH_CHECK(provider && n>0 && h>0 && w>0 && o.output_batch_size>0,"invalid video source/options");TORCH_CHECK(o.mode=="fp32" || o.mode=="fp16" || o.mode=="bf16_reference","invalid neural mode");
  TORCH_CHECK(o.detection.model==o.model && o.update.association.policy==o.model && o.update.recondition.policy==o.model,"video policies must match model");
  const auto model=mux?"sam3.1":"sam3";vision=std::make_shared<VisionEncoder>(store,model,device);detector=std::make_shared<GroundingDetector>(store,model,device);
  if(mux){core31=std::make_shared<Sam31TrackingFrame>(store,device);encoder=std::make_unique<VideoFrameEncoder>(vision,detector,core31,device);factory31=[this]{return std::make_unique<Sam31TrackingSession>(core31,[this](int64_t i){return features(i).tracking;},frames,h,w,device,options.mode,options.sam31_session);};}
  else{core3=std::make_shared<Sam3TrackingFrame>(store,device);encoder=std::make_unique<VideoFrameEncoder>(vision,detector,core3,device);factory3=[this]{return std::make_unique<Sam3TrackingSession>(core3,[this](int64_t i){return features(i).tracking.propagation;},frames,h,w,device,options.mode,options.sam3_session);};}
 }
 void check(int64_t frame)const{TORCH_CHECK(frame>=0 && frame<frames,"frame outside video");}
 const VideoFrameFeatures& features(int64_t frame){check(frame);if(cached_index!=frame){auto rgb=provider(frame);TORCH_CHECK(rgb.sizes()==at::IntArrayRef({3,h,w}) && rgb.scalar_type()==at::kByte,"frame provider must return U8 RGB [3,H,W]");auto next=encoder->encode_rgb(rgb,options.mode);cached=std::move(next);cached_index=frame;++encodes;}return cached;}
 GeometryPrompt empty_geometry()const{const auto o=at::TensorOptions().device(device);const auto labels=at::empty({0,1},o.dtype(at::kLong)),padding=at::empty({1,0},o.dtype(at::kBool));return {at::empty({0,1,2},o),labels,padding,at::empty({0,1,4},o),labels,padding};}
 void encode_text(){
  const auto model=mux?"sam3.1":"sam3";const auto text=has_text?*semantic.text:"<text placeholder>";const auto texts=mux?std::vector<std::string>{text,"visual","geometric"}:std::vector<std::string>{text,"visual"};
  std::vector<at::Tensor> rows;const auto o=at::TensorOptions().device(device);for(const auto& ids:tokenizer.tokenize(texts))rows.push_back(at::tensor(ids,o.dtype(at::kLong)));
  // Text weights are temporary; retained token features are small. Trunk and
  // tracker weights remain shared across semantic replacements and sessions.
  TextEncoder text_encoder(store,model,device);const auto result=text_encoder.forward(at::stack(rows),options.mode);encoded.text_padding=std::get<0>(result);encoded.text_features=std::get<1>(result);encoded.image_ids=at::zeros({1},o.dtype(at::kLong));encoded.text_ids=at::full({1},!mux && !has_text?1:0,o.dtype(at::kLong));
 }
 GroundingPrompt prompt(int64_t frame){if(!encoded.text_features.defined())encode_text();auto result=encoded;result.geometry=empty_geometry();
  if(prompt_frame==frame && semantic.boxes_xywh.defined()){auto boxes=semantic.boxes_xywh.to(device,at::kFloat).clone();boxes.slice(1,0,2).add_(boxes.slice(1,2,4)*.5);result.geometry.boxes=boxes.unsqueeze(1);result.geometry.box_labels=semantic.box_labels.to(device,at::kLong).unsqueeze(1);result.geometry.box_padding=at::zeros({1,boxes.size(0)},at::TensorOptions().device(device).dtype(at::kBool));}
  result.visual_features=semantic.visual_features;result.visual_padding=semantic.visual_padding;return result;
 }
 void reset(){sessions3.clear();sessions31.clear();metadata=initialize_video_metadata(1,device);interaction.reset();suppressions.clear();initialized.clear();prompt_frame.reset();semantic={};encoded={};has_text=false;cached={};cached_index=-1;cancelled.store(false);}
 void cache_raw(const VideoRawOutput& raw){VideoOutput cache;cache.cached_masks=raw.masks;interaction.record(raw.frame,cache);}
 template<class Sessions,class Factory> VideoRawOutput full(int64_t frame,bool reverse,bool direct, Sessions& sessions,const Factory& factory){
  const auto& visual=features(frame);auto raw=encoder->detect(visual,prompt(frame),options.mode);auto detection=options.detection;
  if(mux && direct)detection.nms=VideoNmsMode::Sam31Perflib;
  detection.allow_new_detections=mux || has_text || (prompt_frame==frame && semantic.boxes_xywh.defined()) || semantic.visual_features.defined();
  auto detections=postprocess_video_detections(raw.detection,detection);TORCH_CHECK(detections.size()==1,"one semantic prompt belongs to this video session");auto& d=detections.front();
  std::vector<at::Tensor> masks,scores;std::vector<int64_t> ids;TrackingPropagation request;request.start=frame;request.max_steps=0;request.reverse=reverse;request.encode_memory=false;
  for(auto& session:sessions)session->propagate(request,[&](const auto& value){ids.insert(ids.end(),value.object_ids.begin(),value.object_ids.end());masks.push_back(value.low_masks.squeeze(1));scores.push_back(value.object_logits.flatten());return true;});
  const auto o=at::TensorOptions().device(device).dtype(at::kFloat);auto low=at::empty({0,288,288},o),logits=at::empty({0},o);
  if(!ids.empty()){const auto order=at::tensor(video_memory_rows(ids,{metadata.object_ids()})[0],o.dtype(at::kLong));low=clean_video_mask_scores(at::cat(masks).index_select(0,order).unsqueeze(1),options.update.cleanup_area).squeeze(1);logits=at::cat(scores).index_select(0,order);}
  auto plan=plan_video_update(frame,reverse,d,low,logits,metadata,options.update);execute_video_update(frame,0,plan,d,sessions,factory,options.update);auto masks_out=build_video_outputs(plan,d,h,w,options.update);finalize_video_scores(plan.metadata,frame,plan.previous_ids,logits);metadata=std::move(plan.metadata);metadata.host.removed.insert(plan.removed.begin(),plan.removed.end());
  VideoRawOutput out;out.frame=frame;out.masks=std::move(masks_out);out.scores=metadata.object_scores;out.tracker_scores=metadata.frame_scores[frame];out.removed=metadata.host.removed;out.frame_stats={{"num_obj_tracked",int64_t(metadata.object_ids().size())},{"num_obj_dropped",0}};
  if(mux){const auto suppressed=plan.suppressed.cpu();for(size_t i=0;i<plan.previous_ids.size();++i)if(suppressed[i].template item<bool>())out.suppressed.insert(plan.previous_ids[i]);}else if(metadata.host.suppressed.count(frame))out.suppressed=metadata.host.suppressed.at(frame);
  out.unconfirmed=std::set<int64_t>{};if(options.update.confirmation_enabled){const auto all=metadata.object_ids();for(size_t i=0;i<all.size();++i)if(metadata.confirmation.status[i]==1)out.unconfirmed->insert(all[i]);}
  suppressions[frame]=out.suppressed;initialized.insert(frame);cache_raw(out);return out;
 }
 VideoRawOutput full(int64_t frame,bool reverse,bool direct){return mux?full(frame,reverse,direct,sessions31,factory31):full(frame,reverse,direct,sessions3,factory3);}
 void ensure_cache(int64_t frame){if(!interaction.cached_frames().count(frame))interaction.record(frame,VideoOutput{});}
};
VideoPredictor::VideoPredictor(const WeightStore& s,const std::filesystem::path& vocabulary,FrameProvider provider,int64_t n,int64_t h,int64_t w,at::Device d,const VideoPredictorOptions& o):impl_(std::make_unique<Impl>(s,vocabulary,std::move(provider),n,h,w,d,o)){}
VideoPredictor::~VideoPredictor()=default;VideoPredictor::VideoPredictor(VideoPredictor&&) noexcept=default;VideoPredictor& VideoPredictor::operator=(VideoPredictor&&) noexcept=default;
VideoOutput VideoPredictor::add_prompt(int64_t frame,const VideoSemanticPrompt& p){
 c10::InferenceMode inference;auto& s=*impl_;s.check(frame);TORCH_CHECK(p.text || p.boxes_xywh.defined() || p.visual_features.defined(),"semantic input requires text, boxes or visual tokens");TORCH_CHECK(p.boxes_xywh.defined()==p.box_labels.defined(),"boxes require labels");
 if(p.boxes_xywh.defined()){const auto b=p.boxes_xywh.to(at::kFloat);TORCH_CHECK(b.dim()==2 && b.size(0)>0 && b.size(1)==4 && p.box_labels.dim()==1 && p.box_labels.size(0)==b.size(0),"boxes must be [N,4] with [N] labels");TORCH_CHECK(at::isfinite(b).all().item<bool>() && b.ge(0).all().item<bool>() && b.le(1).all().item<bool>() && (b.slice(1,0,2)+b.slice(1,2,4)*.5).le(1).all().item<bool>(),"boxes must be normalized xywh with normalized centers");TORCH_CHECK(((p.box_labels==0)|(p.box_labels==1)).all().item<bool>(),"box labels must be 0 or 1");}
 TORCH_CHECK(p.visual_features.defined()==p.visual_padding.defined(),"visual tokens require padding");if(p.visual_features.defined())TORCH_CHECK(p.visual_features.dim()==3 && p.visual_features.size(1)==1 && p.visual_features.size(2)==256 && p.visual_padding.scalar_type()==at::kBool && p.visual_padding.sizes()==at::IntArrayRef({1,p.visual_features.size(0)}),"visual tokens must be [N,1,256] with bool [1,N] padding");
 s.reset();s.semantic=p;for(auto* x:{&s.semantic.boxes_xywh,&s.semantic.box_labels,&s.semantic.visual_features,&s.semantic.visual_padding})if(x->defined())*x=x->clone();s.has_text=p.text && (s.mux || *p.text!="visual");s.prompt_frame=frame;s.encode_text();auto raw=s.full(frame,false,true);auto out=postprocess_video_output(raw,s.h,s.w,{}, {},s.options.centers);s.interaction.record(frame,out);return out;
}
VideoOutput VideoPredictor::add_points(int64_t frame,int64_t id,const TrackingPoints& p,bool clear,bool previous,bool stateless){
 c10::InferenceMode inference;auto& s=*impl_;s.check(frame);s.ensure_cache(frame);VideoOutput out;
 if(s.mux){Sam31VideoEditOptions o;o.cleanup_area=s.options.update.cleanup_area;o.confirmation_threshold=s.options.update.confirmation_threshold;o.clear_old_points=clear;o.use_previous_memory=previous;o.stateless_refinement=stateless;out=edit_video_points(frame,id,p,s.sessions31,s.factory31,s.metadata,s.interaction,s.suppressions,o);}
 else{TORCH_CHECK(clear,"SAM3 high-level point API replaces previous points");VideoEditOptions o;o.cleanup_area=s.options.update.cleanup_area;o.confirmation_threshold=s.options.update.confirmation_threshold;o.use_previous_memory=previous;o.stateless_refinement=stateless;out=edit_video_points(frame,id,p,s.sessions3,s.factory3,s.metadata,s.interaction,s.suppressions,o);}
 s.initialized.insert(frame);return centers(std::move(out),s.h,s.w,s.options.centers);
}
VideoOutput VideoPredictor::add_mask(int64_t frame,int64_t id,const at::Tensor& mask){c10::InferenceMode inference;auto& s=*impl_;s.check(frame);TORCH_CHECK(!s.mux,"high-level SAM3.1 exact-mask orchestration is not integrated; use its low-level session mask API");s.ensure_cache(frame);VideoEditOptions o;o.cleanup_area=s.options.update.cleanup_area;o.confirmation_threshold=s.options.update.confirmation_threshold;auto out=edit_video_mask(frame,id,mask,s.sessions3,s.factory3,s.metadata,s.interaction,s.suppressions,o);s.initialized.insert(frame);return centers(std::move(out),s.h,s.w,s.options.centers);}
void VideoPredictor::remove_object(int64_t id){auto& s=*impl_;if(s.mux)remove_video_user_object(id,s.sessions31,s.metadata,s.interaction);else remove_video_user_object(id,s.sessions3,s.metadata,s.interaction);}
VideoOutput VideoPredictor::fetch(int64_t frame)const{const auto& s=*impl_;const auto found=s.suppressions.find(frame);return centers(s.interaction.fetch(frame,s.metadata,found==s.suppressions.end()?std::set<int64_t>{}:found->second),s.h,s.w,s.options.centers);}
void VideoPredictor::propagate(const VideoPredictorPropagation& request,const OutputCallback& callback){
 c10::InferenceMode inference;auto& s=*impl_;TORCH_CHECK(callback,"output callback is required");const auto range=video_processing_range(s.frames,s.initialized,request.start,request.max_steps,request.reverse);const auto route=s.interaction.route(s.metadata.object_ids(),request.force_tracker);s.interaction.append({route.type,request.start,route.ids});s.cancelled.store(false);if(range.empty)return;
 VideoOutputBufferOptions o;o.frame_count=s.frames;o.end_frame=range.end;o.height=s.h;o.width=s.w;o.reverse=request.reverse;o.hotstart_delay=s.options.update.hotstart.delay;o.confirmation_threshold=s.options.update.confirmation_threshold;o.batch_size=s.options.output_batch_size;o.centers=s.options.centers;VideoOutputBuffer buffer(o);
 for(auto frame=range.first;request.reverse?frame>=range.end:frame<=range.end;frame+=range.step){
  if(s.cancelled.load())break;
  if(route.type==VideoActionType::Full){for(const auto& emitted:buffer.push(s.full(frame,request.reverse,false))){s.interaction.record(emitted.frame,emitted.output);if(!callback(emitted.frame,emitted.output)){s.cancelled.store(true);break;}}}
  else if(route.type==VideoActionType::Fetch){if(!callback(frame,fetch(frame)))s.cancelled.store(true);}
  else{TORCH_CHECK(route.type==VideoActionType::Partial && route.ids,"invalid propagation action");s.ensure_cache(frame);RefinedVideoObjects refined;if(s.mux){std::vector<Sam31TrackingSession*> owners;for(auto& x:s.sessions31)owners.push_back(x.get());refined=propagate_video_refinements(frame,request.reverse,*route.ids,owners,s.options.update.cleanup_area);}else{std::vector<Sam3TrackingSession*> owners;for(auto& x:s.sessions3)owners.push_back(x.get());refined=propagate_video_refinements(frame,request.reverse,*route.ids,owners,s.options.update.cleanup_area);}auto out=centers(s.interaction.merge_refined(frame,refined,s.metadata,s.suppressions[frame]),s.h,s.w,s.options.centers);s.initialized.insert(frame);if(!callback(frame,out))s.cancelled.store(true);}
 }
 if(s.cancelled.load()){buffer.cancel();if(s.mux)s.interaction.append({VideoActionType::Cancel,{},{}});}
}
void VideoPredictor::cancel() noexcept{if(impl_)impl_->cancelled.store(true);}
void VideoPredictor::reset(){impl_->reset();}
const VideoMetadata& VideoPredictor::metadata()const{return impl_->metadata;}
const VideoInteractionState& VideoPredictor::interaction()const{return impl_->interaction;}
int64_t VideoPredictor::visual_encodes()const{return impl_->encodes;}
}
