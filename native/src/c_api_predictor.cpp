#include "c_api_internal.h"
#include <cmath>
using namespace sam3::api;
namespace {
sam3::AssociationPolicy policy(int32_t model){require(model==SAM3_MODEL_3 || model==SAM3_MODEL_31,"unknown predictor model");return model==SAM3_MODEL_3?sam3::AssociationPolicy::Sam3:sam3::AssociationPolicy::Sam31;}
sam3::VideoPredictorOptions config(const sam3_predictor_options& o,const Context& context){
  auto c=sam3::video_predictor_defaults(policy(o.model));c.mode=context.mode;
  require((o.model==SAM3_MODEL_3)==(context.model=="sam3"),"predictor/context model mismatch");
  validate_video_options(o.tracking);
  require(!o.image_only || o.tracking.frames==1,"image sources require exactly one frame");
  require(o.nms>=0 && o.nms<=2 && o.policy_precision>=SAM3_FP32 && o.policy_precision<=SAM3_BF16_REFERENCE,"invalid predictor arithmetic/NMS mode");
  require(o.output_batch_size>0 && o.confirmation_threshold>0 && o.cleanup_area>=0 && o.bucket_capacity>0 && o.pad_tracks_to>=0 && o.hotstart_delay>=0 && o.unmatched_threshold>=0 && o.duplicate_threshold>=0 && o.min_keep_alive<=o.max_keep_alive,"invalid predictor count options");
  require(o.image_detection_threshold>=0 && o.image_detection_threshold<=1,"invalid image detection threshold");
  for(auto value:{o.image_detection_threshold,o.detection_score_threshold,o.nms_threshold,o.detection_boundary_margin,o.new_detection_threshold,o.track_match_threshold,o.detection_match_threshold,o.high_confidence_threshold,o.iom_recondition_threshold,o.iou_recondition_threshold,o.recondition_box_iou_threshold,o.recondition_detection_score_threshold,o.boundary_margin,o.occlusion_threshold})require(std::isfinite(value),"predictor thresholds must be finite");
  c.centers=o.centers;
  c.image_only=o.image_only;
  c.output_batch_size=o.output_batch_size;
  c.image_detection_threshold=o.image_detection_threshold;
  c.detection.score_threshold=o.detection_score_threshold;
  c.detection.nms_threshold=o.nms_threshold;
  c.detection.boundary_margin=o.detection_boundary_margin;
  c.detection.use_iom=o.detection_use_iom;
  c.detection.boundary_filter=o.detection_boundary_filter;
  c.update.association.new_detection_threshold=o.new_detection_threshold;
  c.update.association.track_match_threshold=o.track_match_threshold;
  c.update.association.detection_match_threshold=o.detection_match_threshold;
  c.update.association.high_confidence_threshold=o.high_confidence_threshold;
  c.update.association.iom_recondition_threshold=o.iom_recondition_threshold;
  c.update.association.iou_recondition_threshold=o.iou_recondition_threshold;
  c.update.association.pad_tracks_to=o.pad_tracks_to;
  c.update.association.use_iom=o.association_use_iom;
  c.update.hotstart.unmatched_threshold=o.unmatched_threshold;
  c.update.hotstart.duplicate_threshold=o.duplicate_threshold;
  c.update.hotstart.initial_keep_alive=o.initial_keep_alive;
  c.update.hotstart.min_keep_alive=o.min_keep_alive;
  c.update.hotstart.max_keep_alive=o.max_keep_alive;
  c.update.hotstart.suppress_only_within_hotstart=o.suppress_only_within_hotstart;
  c.update.hotstart.decrease_for_empty=o.decrease_for_empty;
  c.update.hotstart.delay=o.hotstart_delay;
  c.update.recondition.period=o.recondition_period;
  c.update.recondition.box_iou_threshold=o.recondition_box_iou_threshold;
  c.update.recondition.detection_score_threshold=o.recondition_detection_score_threshold;
  c.update.boundary_filter=o.boundary_filter;
  c.update.confirmation_enabled=o.confirmation_enabled;
  c.update.warmup_complete=o.warmup_complete;
  c.update.allow_unoccluded_suppression=o.allow_unoccluded_suppression;
  c.update.reapply_no_object_pointer=o.reapply_no_object_pointer;
  c.update.boundary_margin=o.boundary_margin;
  c.update.occlusion_threshold=o.occlusion_threshold;
  c.update.confirmation_threshold=o.confirmation_threshold;
  c.update.cleanup_area=o.cleanup_area;
  c.update.bucket_capacity=o.bucket_capacity;
  c.detection.nms=static_cast<sam3::VideoNmsMode>(o.nms);
  c.detection.policy_mode=c.update.policy_mode=o.policy_precision==SAM3_FP32?"fp32":o.policy_precision==SAM3_FP16?"fp16":"bf16_reference";
  if(o.model==SAM3_MODEL_3)c.sam3_session=tracking_options(o.tracking);else c.sam31_session=multiplex_options(o.tracking);
  return c;
}
template<class F> auto predictor(sam3_predictor* p,F&& operation){require(p,"predictor is required");auto guard=lock(p->mutex);return operation(*p->value);}
void pack(int64_t frame,const sam3::VideoOutput& value,sam3_result** out){
  std::map<std::string,at::Tensor> fields{{"frame",at::scalar_tensor(frame,at::kLong)},{"ids",value.ids},{"probabilities",value.probabilities},{"boxes_xywh",value.boxes_xywh},{"masks",value.masks},{"centers",value.centers}};
  std::vector<int64_t> ids;for(const auto& [id,mask]:value.cached_masks){ids.push_back(id);fields.emplace("cached_masks/"+std::to_string(id),mask);}
  fields.emplace("cached_ids",at::tensor(ids,at::kLong));for(const auto& [name,count]:value.frame_stats)fields.emplace("stats/"+name,at::scalar_tensor(count,at::kLong));result(std::move(fields),out);
}
}
extern "C" {
sam3_status sam3_predictor_options_init(sam3_predictor_options* o,int32_t model) noexcept{return protect([&]{
  require(o,"predictor options are required");const auto c=sam3::video_predictor_defaults(policy(model));*o={};o->struct_size=sizeof(*o);o->model=model;
  sam3_video_options_init(&o->tracking,model);o->tracking.select_by_score=model==SAM3_MODEL_3;o->tracking.all_edits_conditioning=model==SAM3_MODEL_3;
  o->centers=c.centers;
  o->image_only=c.image_only;
  o->output_batch_size=c.output_batch_size;
  o->image_detection_threshold=c.image_detection_threshold;
  o->detection_score_threshold=c.detection.score_threshold;
  o->nms_threshold=c.detection.nms_threshold;
  o->detection_boundary_margin=c.detection.boundary_margin;
  o->detection_use_iom=c.detection.use_iom;
  o->detection_boundary_filter=c.detection.boundary_filter;
  o->new_detection_threshold=c.update.association.new_detection_threshold;
  o->track_match_threshold=c.update.association.track_match_threshold;
  o->detection_match_threshold=c.update.association.detection_match_threshold;
  o->high_confidence_threshold=c.update.association.high_confidence_threshold;
  o->iom_recondition_threshold=c.update.association.iom_recondition_threshold;
  o->iou_recondition_threshold=c.update.association.iou_recondition_threshold;
  o->pad_tracks_to=c.update.association.pad_tracks_to;
  o->association_use_iom=c.update.association.use_iom;
  o->unmatched_threshold=c.update.hotstart.unmatched_threshold;
  o->duplicate_threshold=c.update.hotstart.duplicate_threshold;
  o->initial_keep_alive=c.update.hotstart.initial_keep_alive;
  o->min_keep_alive=c.update.hotstart.min_keep_alive;
  o->max_keep_alive=c.update.hotstart.max_keep_alive;
  o->suppress_only_within_hotstart=c.update.hotstart.suppress_only_within_hotstart;
  o->decrease_for_empty=c.update.hotstart.decrease_for_empty;
  o->hotstart_delay=c.update.hotstart.delay;
  o->recondition_period=c.update.recondition.period;
  o->recondition_box_iou_threshold=c.update.recondition.box_iou_threshold;
  o->recondition_detection_score_threshold=c.update.recondition.detection_score_threshold;
  o->boundary_filter=c.update.boundary_filter;
  o->confirmation_enabled=c.update.confirmation_enabled;
  o->warmup_complete=c.update.warmup_complete;
  o->allow_unoccluded_suppression=c.update.allow_unoccluded_suppression;
  o->reapply_no_object_pointer=c.update.reapply_no_object_pointer;
  o->boundary_margin=c.update.boundary_margin;
  o->occlusion_threshold=c.update.occlusion_threshold;
  o->confirmation_threshold=c.update.confirmation_threshold;
  o->cleanup_area=c.update.cleanup_area;
  o->bucket_capacity=c.update.bucket_capacity;
  o->nms=int32_t(c.detection.nms);o->policy_precision=SAM3_FP32;
});}
sam3_status sam3_semantic_prompt_init(sam3_semantic_prompt* p) noexcept{return protect([&]{require(p,"semantic prompt is required");*p={};p->struct_size=sizeof(*p);});}
sam3_status sam3_predictor_create(sam3_context* context,const sam3_predictor_options* o,sam3_predictor** out) noexcept{return protect([&]{
  output(out);require(context,"context is required");options(o);auto& c=*context->value;const auto settings=config(*o,c);require(!c.vocabulary.empty(),"predictor vocabulary_path is required");
  auto handle=std::make_unique<sam3_predictor>();handle->context=context->value;sam3::VideoPredictorModules modules;modules.vision=c.vision();modules.detector=c.detector();if(c.model=="sam3")modules.sam3=c.tracking();else modules.sam31=c.multiplex();
  const auto& source=o->tracking;handle->value=std::make_unique<sam3::VideoPredictor>(c.store,c.vocabulary,frame_provider(source),source.frames,source.height,source.width,c.device,settings,modules);*out=handle.release();
});}
void sam3_predictor_release(sam3_predictor* p) noexcept{delete p;}
sam3_status sam3_predictor_add_prompt(sam3_predictor* handle,int64_t frame,const sam3_semantic_prompt* input,sam3_result** out) noexcept{return protect([&]{output(out);options(input);predictor(handle,[&](auto& p){
  sam3::VideoSemanticPrompt prompt;if(input->text){const auto& text=*input->text;require((text.data || !text.bytes) && text.bytes<=SIZE_MAX,"invalid UTF-8 view");prompt.text=std::string(text.data?text.data:"",size_t(text.bytes));}
  prompt.boxes_xywh=tensor(input->boxes_xywh);prompt.box_labels=tensor(input->box_labels);prompt.visual_features=tensor(input->visual_features);prompt.visual_padding=tensor(input->visual_padding);pack(frame,p.add_prompt(frame,prompt),out);
});});}
sam3_status sam3_predictor_add_points(sam3_predictor* handle,int64_t frame,int64_t id,const sam3_tensor_view* points,const sam3_tensor_view* labels,const sam3_tensor_view* box,int32_t normalized,int32_t clear,int32_t previous,int32_t stateless,sam3_result** out) noexcept{return protect([&]{output(out);predictor(handle,[&](auto& p){const sam3::TrackingPoints input{tensor(points),tensor(labels),tensor(box),bool(normalized)};pack(frame,p.add_points(frame,id,input,clear,previous,stateless),out);});});}
sam3_status sam3_predictor_add_mask(sam3_predictor* handle,int64_t frame,int64_t id,const sam3_tensor_view* mask,sam3_result** out) noexcept{return protect([&]{output(out);require(mask,"mask is required");predictor(handle,[&](auto& p){pack(frame,p.add_mask(frame,id,tensor(mask)),out);});});}
sam3_status sam3_predictor_fetch(sam3_predictor* handle,int64_t frame,sam3_result** out) noexcept{return protect([&]{output(out);predictor(handle,[&](auto& p){pack(frame,p.fetch(frame),out);});});}
sam3_status sam3_predictor_propagate(sam3_predictor* handle,int64_t start,int64_t steps,int32_t reverse,int32_t force,sam3_output_callback callback,void* user) noexcept{return protect([&]{
  require(callback && start>=-1 && steps>=-1,"callback and valid propagation bounds are required");predictor(handle,[&](auto& p){sam3::VideoPredictorPropagation request;if(start>=0)request.start=start;if(steps>=0)request.max_steps=steps;request.reverse=reverse;request.force_tracker=force;
  p.propagate(request,[&](int64_t frame,const sam3::VideoOutput& value){sam3_result* raw=nullptr;pack(frame,value,&raw);const auto held=std::unique_ptr<sam3_result,decltype(&sam3_result_release)>(raw,sam3_result_release);const auto status=callback(user,raw);if(status!=0 && status!=1)throw Failure(SAM3_CALLBACK_ERROR,"output callback reported failure");return status==0;});});
});}
sam3_status sam3_predictor_cancel(sam3_predictor* handle) noexcept{return protect([&]{require(handle,"predictor is required");handle->value->cancel();});}
sam3_status sam3_predictor_remove_object(sam3_predictor* handle,int64_t id) noexcept{return protect([&]{predictor(handle,[&](auto& p){p.remove_object(id);});});}
sam3_status sam3_predictor_reset(sam3_predictor* handle) noexcept{return protect([&]{predictor(handle,[&](auto& p){p.reset();});});}
sam3_status sam3_predictor_info(sam3_predictor* handle,sam3_result** out) noexcept{return protect([&]{output(out);predictor(handle,[&](auto& p){std::vector<int64_t> frames;for(const auto& [frame,masks]:p.interaction().cached_frames())frames.push_back(frame);result({{"ids",at::tensor(p.metadata().object_ids(),at::kLong)},{"visual_encodes",at::scalar_tensor(p.visual_encodes(),at::kLong)},{"cached_frames",at::tensor(frames,at::kLong)},{"action_count",at::scalar_tensor(int64_t(p.interaction().actions().size()),at::kLong)}},out);});});}
}
