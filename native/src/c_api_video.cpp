#include "c_api_internal.h"
#include <cmath>
using namespace sam3::api;
namespace {
sam3::TemporalOptions temporal(const sam3_video_options& o){
  require(o.memory_slots>=0 && o.memory_slots<=7 && o.max_pointer_frames>0 && o.stride>0 && (o.max_conditioning_frames==-1 || o.max_conditioning_frames>=2) && std::isfinite(o.score_threshold),"invalid temporal options");
  return {o.memory_slots,o.max_conditioning_frames,o.max_pointer_frames,o.stride,bool(o.keep_first),bool(o.select_by_score),o.score_threshold};
}
}
namespace sam3::api {
void validate_video_options(const sam3_video_options& o,bool require_provider){
  options(&o);require(o.frames>0 && o.height>0 && o.width>0 && (!require_provider || o.provider),"video dimensions and frame provider are required");
  require(o.fill_hole_area>=0 && o.multimask_min_points>=0 && o.multimask_max_points>=o.multimask_min_points && std::isfinite(o.object_threshold),"invalid mask options");temporal(o);
}
TrackingSessionOptions tracking_options(const sam3_video_options& input){const auto* o=&input;const auto timing=temporal(input);
  if(o->history_directory && *o->history_directory)throw Failure(SAM3_UNSUPPORTED,"SAM3 history disk paging is not implemented yet");
    sam3::TrackingSessionOptions config;config.offload_state=o->offload_state;config.non_overlap_output=o->non_overlap_output;config.clear_near_input=o->clear_near_input;config.clear_near_multi_object=o->clear_near_multi_object;
    config.all_edits_conditioning=o->all_edits_conditioning;config.always_start_at_first_annotation=o->always_start_at_first_annotation;config.fill_hole_area=o->fill_hole_area;
    config.frame.temporal=timing;config.frame.non_overlap_memory=o->non_overlap_memory;config.frame.multimask=o->multimask;config.frame.multimask_tracking=o->multimask_tracking;config.frame.multimask_min_points=o->multimask_min_points;config.frame.multimask_max_points=o->multimask_max_points;
  return config;
}
MultiplexSessionOptions multiplex_options(const sam3_video_options& input){const auto* o=&input;const auto timing=temporal(input);
    sam3::MultiplexSessionOptions config;config.offload_state=o->offload_state;config.non_overlap_output=o->non_overlap_output;config.all_edits_conditioning=o->all_edits_conditioning;config.always_start_at_first_annotation=o->always_start_at_first_annotation;config.fill_hole_area=o->fill_hole_area;
    if(o->history_directory)config.history_directory=std::filesystem::u8path(o->history_directory);
    static_cast<sam3::TemporalOptions&>(config.frame.temporal)=timing;config.frame.temporal.only_past_pointers=o->only_past_pointers;config.frame.temporal.signed_pointer_time=o->signed_pointer_time;config.frame.temporal.temporal_v2=o->temporal_v2;config.frame.temporal.encode_pointer_time=o->encode_pointer_time;config.frame.temporal.use_pointers=o->use_pointers;
    config.frame.non_overlap_memory=o->non_overlap_memory;config.frame.multimask=o->multimask;config.frame.multimask_tracking=o->multimask_tracking;config.frame.multimask_min_points=o->multimask_min_points;config.frame.multimask_max_points=o->multimask_max_points;config.frame.attenuate_iou_by_stability=o->attenuate_iou_by_stability;config.frame.object_threshold=o->object_threshold;
  return config;
}
VideoPredictor::FrameProvider frame_provider(const sam3_video_options& o){
  const auto provider=o.provider;const auto user=o.provider_user;const auto h=o.height,w=o.width;
  return [provider,user,h,w](int64_t index){sam3_rgb_view view{};if(provider(user,index,&view)!=0)throw Failure(SAM3_CALLBACK_ERROR,"frame provider reported failure");require(view.height==h && view.width==w,"frame provider changed video dimensions");return rgb(view);};
}
}
extern "C" {
sam3_status sam3_video_options_init(sam3_video_options* o,int32_t model) noexcept{return protect([&]{require(o && (model==SAM3_MODEL_3 || model==SAM3_MODEL_31),"options and model are required");*o={};o->struct_size=sizeof(*o);o->offload_state=1;o->non_overlap_output=model==SAM3_MODEL_31;o->clear_near_input=model==SAM3_MODEL_3;o->all_edits_conditioning=1;o->memory_slots=7;o->max_conditioning_frames=4;o->max_pointer_frames=16;o->stride=1;o->score_threshold=.01;o->multimask=1;o->multimask_tracking=1;o->multimask_max_points=1;o->temporal_v2=1;o->encode_pointer_time=1;o->use_pointers=1;});}
sam3_status sam3_video_create(sam3_context* context,const sam3_video_options* o,sam3_video** out) noexcept{return protect([&]{
  output(out);require(context,"context is required");options(o);validate_video_options(*o);
  auto handle=std::make_unique<sam3_video>();handle->context=context->value;auto& c=*handle->context;
  const auto h=o->height,w=o->width;const auto pixels=frame_provider(*o);
  if(c.model=="sam3"){
    const auto config=tracking_options(*o);
    const auto core=c.tracking();handle->vision3=std::make_shared<sam3::Sam3TrackingVision>(c.vision(),core,c.device);const auto vision=handle->vision3;const auto mode=c.mode;
    handle->tracker=std::make_unique<sam3::Sam3TrackingSession>(core,[vision,pixels,mode](int64_t index){return vision->encode_rgb(pixels(index),mode);},o->frames,h,w,c.device,c.mode,config);
  }else{
    const auto config=multiplex_options(*o);
    const auto core=c.multiplex();handle->vision31=std::make_shared<sam3::Sam31TrackingVision>(c.vision(),core,c.device);const auto vision=handle->vision31;const auto mode=c.mode;
    handle->multiplex=std::make_unique<sam3::Sam31TrackingSession>(core,[vision,pixels,mode](int64_t index){return vision->encode_rgb(pixels(index),mode);},o->frames,h,w,c.device,c.mode,config);
  }
  *out=handle.release();
});}
void sam3_video_release(sam3_video* handle) noexcept{delete handle;}
sam3_status sam3_video_add_points(sam3_video* handle,int64_t frame,int64_t object,const sam3_tensor_view* points,const sam3_tensor_view* labels,const sam3_tensor_view* box,int32_t normalized,int32_t clear,int32_t previous,sam3_result** out) noexcept{return protect([&]{output(out);video(handle,[&](auto& session){const sam3::TrackingPoints input{tensor(points),tensor(labels),tensor(box),bool(normalized)};video_result(session.add_points(frame,object,input,clear,previous),out);});});}
sam3_status sam3_video_add_mask(sam3_video* handle,int64_t frame,int64_t object,const sam3_tensor_view* mask,sam3_result** out) noexcept{return protect([&]{output(out);require(mask,"mask is required");video(handle,[&](auto& session){video_result(session.add_mask(frame,object,tensor(mask)),out);});});}
sam3_status sam3_video_add_masks(sam3_video* handle,int64_t frame,const int64_t* ids,int64_t count,const sam3_tensor_view* masks,sam3_result** out) noexcept{return protect([&]{output(out);require(handle && ids && count>0 && masks,"video, IDs and masks are required");auto guard=lock(handle->mutex);if(!handle->multiplex)throw Failure(SAM3_UNSUPPORTED,"simultaneous brush input is a SAM3.1 operation");video_result(handle->multiplex->add_masks(frame,std::vector<int64_t>(ids,ids+count),tensor(masks)),out);});}
sam3_status sam3_video_preflight(sam3_video* handle,int32_t encode) noexcept{return protect([&]{video(handle,[&](auto& session){session.preflight(encode);});});}
sam3_status sam3_video_propagate(sam3_video* handle,int64_t start,int64_t steps,int32_t reverse,int32_t encode,int32_t preflight,sam3_output_callback callback,void* user) noexcept{return protect([&]{require(callback && start>=-1 && steps>=-1,"callback and valid propagation bounds are required");video(handle,[&](auto& session){sam3::TrackingPropagation request;if(start>=0)request.start=start;if(steps>=0)request.max_steps=steps;request.reverse=reverse;request.encode_memory=encode;request.preflight=preflight;session.propagate(request,[&](const auto& output){return emit(output,callback,user);});});});}
sam3_status sam3_video_cancel(sam3_video* handle) noexcept{return protect([&]{require(handle,"video is required");if(handle->multiplex)handle->multiplex->cancel();else handle->tracker->cancel();});}
sam3_status sam3_video_clear_input(sam3_video* handle,int64_t frame,int64_t object,sam3_result** out) noexcept{return protect([&]{output(out);video(handle,[&](auto& session){video_result(session.clear_input(frame,object),out);});});}
sam3_status sam3_video_remove_object(sam3_video* handle,int64_t object,int32_t strict,sam3_output_callback callback,void* user) noexcept{return protect([&]{require(handle,"video is required");auto guard=lock(handle->mutex);if(handle->multiplex)handle->multiplex->remove_object(object,strict);else for(const auto& value:handle->tracker->remove_object(object,strict))if(!emit(value,callback,user))break;});}
sam3_status sam3_video_object_ids(sam3_video* handle,sam3_result** out) noexcept{return protect([&]{output(out);video(handle,[&](auto& session){result({{"ids",at::tensor(session.object_ids(),at::kLong)}},out);});});}
sam3_status sam3_video_reset(sam3_video* handle) noexcept{return protect([&]{video(handle,[&](auto& session){session.reset();});});}
}
