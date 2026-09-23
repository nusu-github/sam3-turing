#include "sam3/multiplex_session.h"
#include "sam3/multiplex_storage.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
at::Tensor resize(const at::Tensor& value,int64_t h,int64_t w,bool aa=false) {if(value.size(-2)==h && value.size(-1)==w)return value;return aa?at::_upsample_bilinear2d_aa(value,{h,w},false):at::upsample_bilinear2d(value,{h,w},false);}
void erase(std::vector<MultiplexFrame>& values,int64_t index){values.erase(std::remove_if(values.begin(),values.end(),[&](const auto& value){return value.index==index;}),values.end());}
at::Tensor non_overlap(const at::Tensor& masks){if(masks.size(0)<2)return masks;const auto ids=at::arange(masks.size(0),masks.options().dtype(at::kLong)).view({-1,1,1,1});return at::where(masks.argmax(0,true)==ids,masks,masks.clamp_max(-10.));}
}
Sam31TrackingSession::Sam31TrackingSession(std::shared_ptr<const Sam31TrackingFrame> core,FeatureProvider provider,int64_t frames,int64_t h,int64_t w,at::Device device,const std::string& mode,const MultiplexSessionOptions& options)
    :core_(std::move(core)),provider_(std::move(provider)),frames_(frames),height_(h),width_(w),device_(device),storage_(device),mode_(mode),options_(options),frame_options_(options.frame) {
  detail::check_mode(mode);TORCH_CHECK(core_ && provider_ && frames>0 && h>0 && w>0,"invalid multiplex session");
  device_=at::empty({0},at::TensorOptions().device(device)).device();storage_=(options.offload_state || !options.history_directory.empty())?at::Device(at::kCPU):device_;
  // Session retention is responsible for storage: core trimming would discard
  // the full masks/images required to reconstruct joint memory after edits.
  frame_options_.offload_output=false;frame_options_.trim_history=false;frame_options_.save_image=true;
}
void Sam31TrackingSession::check_frame(int64_t index)const{TORCH_CHECK(index>=0 && index<frames_,"frame index out of range");}
std::vector<int64_t> Sam31TrackingSession::object_ids()const{std::vector<int64_t> ids;for(const auto& object:state_.objects)ids.push_back(object.id);return ids;}
MultiplexTrackingFeatures Sam31TrackingSession::features(int64_t index){
  check_frame(index);if(cached_index_!=index){auto value=provider_(index);for(auto* item:{&value.interactive,&value.propagation}){TORCH_CHECK(item->image.sizes()==at::IntArrayRef({1,256,72,72}) && item->position.sizes()==item->image.sizes() && item->high.size()==2,"provider requires both shared tracking necks");item->image=item->image.to(device_);item->position=item->position.to(device_);for(auto& high:item->high)high=high.to(device_);}cached_=std::move(value);cached_index_=index;}return cached_;
}
MultiplexFrame* Sam31TrackingSession::find(int64_t index){for(auto* group:{&state_.history.conditioning,&state_.history.tracked})for(auto& frame:*group)if(frame.index==index)return &frame;return nullptr;}
void Sam31TrackingSession::put(MultiplexFrame frame,bool conditioning){auto& group=conditioning?state_.history.conditioning:state_.history.tracked;auto& other=conditioning?state_.history.tracked:state_.history.conditioning;erase(other,frame.index);for(auto& old:group)if(old.index==frame.index){old=std::move(frame);return;}group.push_back(std::move(frame));}
void Sam31TrackingSession::store(MultiplexFrame& frame,bool compress){
  if(frame.memory.defined())frame.memory=(compress?frame.memory.to(at::kBFloat16):frame.memory).to(storage_);
  for(auto* value:{&frame.memory_position,&frame.image,&frame.image_position,&frame.masks.low_res_mask,&frame.masks.high_res_mask,&frame.memory_masks,&frame.memory_object_logits})if(value->defined())*value=value->to(storage_);
  frame.masks.low_res_multimasks=at::Tensor();frame.masks.high_res_multimasks=at::Tensor();frame.masks.iou=at::Tensor();frame.masks.object_pointer=at::Tensor();
  if(!options_.history_directory.empty()){if(frame.confidence.defined())frame.confidence=frame.confidence.cpu();archive_multiplex_frame(frame,options_.history_directory);}
}
void Sam31TrackingSession::remap(const MultiplexState& next){
  remap_multiplex_history(state_.history,*state_.buckets,next,[&](const auto& frame,const auto& layout){const auto encoded=core_->encode_history(frame,layout,frame_options_,mode_);return std::make_pair(encoded.features,encoded.position);},mode_,[&](auto& frame){store(frame,false);});
  state_.buckets=next;
}
size_t Sam31TrackingSession::ensure_object(int64_t id,bool prefer_new){
  for(size_t i=0;i<state_.objects.size();++i)if(state_.objects[i].id==id)return i;
  const auto index=state_.objects.size();
  if(!state_.buckets)state_.buckets=MultiplexController().get_state(1,device_,at::kFloat,false,std::vector<int64_t>{id});
  else {auto next=*state_.buckets;next.add_objects({int64_t(index)},std::vector<int64_t>{id},true,prefer_new);remap(next);}
  state_.objects.push_back(MultiplexSessionObject{id});return index;
}
MultiplexFrame Sam31TrackingSession::blank(int64_t index){
  MultiplexFrame frame;frame.index=index;const auto count=int64_t(state_.objects.size()),batch=state_.buckets->bucket_count();const auto opts=at::TensorOptions().device(device_).dtype(at::kFloat);
  frame.masks.low_res_mask=at::full({count,1,288,288},-1024.,opts);frame.masks.high_res_mask=at::full({count,1,1008,1008},-1024.,opts);frame.masks.object_logits=at::full({count,1},-1024.,opts);frame.pointer=at::zeros({batch,16,256},opts);
  if(frame_options_.temporal.select_by_score)frame.iou=at::zeros({count},opts);
  const auto value=features(index);frame.image=value.propagation.image.flatten(2).permute({2,0,1});frame.image_position=value.propagation.position.flatten(2).permute({2,0,1});return frame;
}
void Sam31TrackingSession::merge_edit(int64_t index,size_t object,const MultiplexFrame& edit,const at::Tensor& video){
  const auto existing=find(index);auto frame=existing?load_multiplex_frame(*existing):blank(index);
  frame.masks.low_res_mask=frame.masks.low_res_mask.to(device_).clone();frame.masks.low_res_mask.slice(0,object,object+1).copy_(resize(edit.masks.low_res_mask,288,288,true));
  frame.masks.high_res_mask=frame.masks.high_res_mask.to(device_).clone();frame.masks.high_res_mask.slice(0,object,object+1).copy_(resize(edit.masks.high_res_mask,1008,1008));
  const auto score_type=existing?c10::promoteTypes(frame.masks.object_logits.scalar_type(),edit.masks.object_logits.scalar_type()):edit.masks.object_logits.scalar_type();
  frame.masks.object_logits=frame.masks.object_logits.to(score_type).clone();frame.masks.object_logits.slice(0,object,object+1).copy_(edit.masks.object_logits);
  frame.pointer=frame.pointer.to(edit.pointer.scalar_type()).clone();
  for(size_t bucket=0;bucket<state_.buckets->assignments().size();++bucket)for(int64_t slot=0;slot<16;++slot)if(state_.buckets->assignments()[bucket][slot]==int64_t(object))frame.pointer[bucket][slot].copy_(edit.pointer[0][0]);
  if(frame_options_.temporal.select_by_score){frame.iou=frame.iou.to(existing?c10::promoteTypes(frame.iou.scalar_type(),edit.masks.iou.scalar_type()):edit.masks.iou.scalar_type()).clone();frame.iou.slice(0,object,object+1).copy_(std::get<0>(edit.masks.iou.max(-1)));frame.confidence=memory_confidence(frame.masks.object_logits,frame.iou);}
  if(std::find(frame.conditioning_objects.begin(),frame.conditioning_objects.end(),object)==frame.conditioning_objects.end())frame.conditioning_objects.push_back(object);
  frame.memory=at::Tensor();frame.memory_position=at::Tensor();frame.memory_masks=at::Tensor();frame.memory_object_logits=at::Tensor();store(frame);put(std::move(frame),!state_.tracked_direction.count(index) || options_.all_edits_conditioning);
  state_.objects[object].video_edits[index]=video.to(storage_);state_.dirty.insert(index);state_.annotated.insert(index);
}
TrackingSessionOutput Sam31TrackingSession::output(const MultiplexFrame& stored,bool preview)const{
  const auto frame=load_multiplex_frame(stored,history_output);
  auto masks=resize(frame.masks.low_res_mask.to(device_),height_,width_);
  if(preview){masks=masks.clone();for(size_t i=0;i<state_.objects.size();++i)if(state_.objects[i].video_edits.count(frame.index))masks.slice(0,i,i+1).copy_(state_.objects[i].video_edits.at(frame.index).to(device_));}
  return {frame.index,object_ids(),frame.masks.low_res_mask.to(device_),postprocess_tracking_masks(masks,height_,width_,options_.non_overlap_output,options_.fill_hole_area),frame.masks.object_logits};
}
TrackingSessionOutput Sam31TrackingSession::add_points(int64_t index,int64_t id,const TrackingPoints& prompt,bool clear,bool use_previous){
  c10::InferenceMode inference;check_frame(index);AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(prompt.points.defined()==prompt.labels.defined() && (prompt.points.defined() || prompt.box.defined()),"points require labels or a box");TORCH_CHECK(!prompt.box.defined() || clear,"box input requires clearing old points");
  const auto opts=at::TensorOptions().device(device_).dtype(at::kFloat);auto coords=prompt.points.defined()?prompt.points.to(opts):at::empty({0,2},opts);auto labels=prompt.labels.defined()?prompt.labels.to(device_,at::kInt):at::empty({0},opts.dtype(at::kInt));
  if(coords.dim()==2)coords=coords.unsqueeze(0);if(labels.dim()==1)labels=labels.unsqueeze(0);TORCH_CHECK(coords.dim()==3 && coords.size(0)==1 && coords.size(2)==2 && labels.sizes()==coords.sizes().slice(0,2),"one object's points and labels are required");
  if(prompt.normalized)coords=coords*1008;
  if(prompt.box.defined()){TORCH_CHECK(prompt.box.numel()==4,"box requires xyxy coordinates");auto box=prompt.box.to(opts).reshape({1,2,2});if(prompt.normalized)box=box*1008;coords=at::cat({box,coords},1);labels=at::cat({at::tensor({2,3},labels.options()).view({1,2}),labels},1);}
  auto previous=state_;try {
    const auto object=ensure_object(id,true);auto& item=state_.objects[object];if(!clear && item.points.count(index)){coords=at::cat({item.points.at(index).points,coords},1);labels=at::cat({item.points.at(index).labels,labels},1);}
    MultiplexFrameRequest request;request.index=index;request.frame_count=frames_;request.initial=true;request.encode_memory=false;request.points=coords;request.labels=labels;
    if((item.refined.count(index) || use_previous) && find(index)){request.previous_logits=load_multiplex_frame(*find(index),history_output).masks.low_res_mask.slice(0,object,object+1).to(device_).clamp(-32.,32.);request.objects_to_interact=std::vector<int64_t>{0};}
    const auto value=features(index);const auto singleton=MultiplexController().get_state(1,device_,at::kFloat,false,std::vector<int64_t>{id});MultiplexFrameHistory empty;
    const auto edit=core_->forward(value.interactive,value.propagation,request,empty,singleton,frame_options_,mode_);
    item.points[index]={coords,labels,{},false};item.masks.erase(index);if(state_.tracked_direction.count(index))item.refined.insert(index);
    merge_edit(index,object,edit,postprocess_tracking_masks(edit.masks.low_res_mask,height_,width_,false,options_.fill_hole_area));return output(*find(index),true);
  }catch(...){state_=std::move(previous);throw;}
}
TrackingSessionOutput Sam31TrackingSession::add_mask(int64_t index,int64_t id,const at::Tensor& mask){
  c10::InferenceMode inference;check_frame(index);TORCH_CHECK(mask.dim()==2 && mask.numel()>0,"mask must be [H,W]");AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  auto previous=state_;try {
    const auto object=ensure_object(id,false);const auto raw=mask.to(device_,at::kFloat).unsqueeze(0).unsqueeze(0),video=resize(raw,height_,width_,true)>.5;
    MultiplexFrameRequest request;request.index=index;request.frame_count=frames_;request.initial=true;request.encode_memory=false;request.mask=resize(raw,1152,1152,true);
    const auto value=features(index);const auto singleton=MultiplexController().get_state(1,device_,at::kFloat,false,std::vector<int64_t>{id});MultiplexFrameHistory empty;
    const auto edit=core_->forward(value.interactive,value.propagation,request,empty,singleton,frame_options_,mode_);
    auto& item=state_.objects[object];item.masks[index]=video.to(storage_);item.points.erase(index);item.refined.erase(index);
    merge_edit(index,object,edit,at::where(video,1024.,-1024.));
    for(auto& other:state_.objects)if(other.id!=id && other.video_edits.count(index))other.video_edits[index]=at::where(video.to(storage_),-1024.,other.video_edits.at(index));
    return output(*find(index),true);
  }catch(...){state_=std::move(previous);throw;}
}
TrackingSessionOutput Sam31TrackingSession::add_masks(int64_t index,const std::vector<int64_t>& ids,const at::Tensor& masks){
  c10::InferenceMode inference;check_frame(index);TORCH_CHECK(!ids.empty() && masks.dim()==3 && masks.size(0)==int64_t(ids.size()),"one [H,W] mask per object ID is required");
  TORCH_CHECK(std::set<int64_t>(ids.begin(),ids.end()).size()==ids.size(),"mask object IDs must be unique");
  auto previous=state_;try {
    AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
    std::vector<size_t> objects;for(auto id:ids)objects.push_back(ensure_object(id,false));
    const auto value=features(index);const auto group=MultiplexController().get_state(ids.size(),device_,at::kFloat,false,ids);MultiplexFrameHistory empty;
    MultiplexFrameRequest request;request.index=index;request.frame_count=frames_;request.initial=true;request.encode_memory=false;request.mask=resize(masks.to(device_,at::kFloat).unsqueeze(1),1152,1152,true);
    const auto batch=core_->forward(value.interactive,value.propagation,request,empty,group,frame_options_,mode_);
    const auto binary=resize(masks.to(device_,at::kFloat).unsqueeze(1),height_,width_,true)>.5,counts=binary.to(at::kLong).sum(0,true);
    for(size_t i=0;i<ids.size();++i){
      MultiplexFrame edit;edit.masks.low_res_mask=batch.masks.low_res_mask.slice(0,i,i+1);edit.masks.high_res_mask=batch.masks.high_res_mask.slice(0,i,i+1);edit.masks.object_logits=batch.masks.object_logits.slice(0,i,i+1);edit.masks.iou=batch.masks.iou.slice(0,i,i+1);edit.pointer=at::zeros({1,16,256},batch.pointer.options());edit.pointer[0][0].copy_(batch.pointer[i/16][i%16]);
      const auto own=binary.slice(0,i,i+1),video=at::where((counts-own.to(at::kLong))>0,-1024.,at::where(own,1024.,-1024.));
      auto& object=state_.objects[objects[i]];object.masks[index]=own.to(storage_);object.points.erase(index);object.refined.erase(index);merge_edit(index,objects[i],edit,video);
    }
    for(auto& object:state_.objects)if(std::find(ids.begin(),ids.end(),object.id)==ids.end() && object.video_edits.count(index))object.video_edits[index]=at::where((counts>0).to(storage_),-1024.,object.video_edits.at(index));
    return output(*find(index),true);
  }catch(...){state_=std::move(previous);throw;}
}
TrackingSessionOutput Sam31TrackingSession::recondition_masks(int64_t index,const std::vector<int64_t>& ids,const at::Tensor& masks){
  c10::InferenceMode inference;check_frame(index);
  TORCH_CHECK(!ids.empty() && masks.dim()==3 && masks.size(0)==int64_t(ids.size()) && masks.size(1)>0 && masks.size(2)>0,"one nonempty [H,W] mask per object ID is required");
  TORCH_CHECK(std::set<int64_t>(ids.begin(),ids.end()).size()==ids.size(),"mask object IDs must be unique");
  TORCH_CHECK(state_.buckets && find(index),"reconditioning requires an existing frame");
  std::vector<int64_t> objects;
  for(auto id:ids){const auto it=std::find_if(state_.objects.begin(),state_.objects.end(),[&](const auto& obj){return obj.id==id;});TORCH_CHECK(it!=state_.objects.end(),"reconditioning cannot introduce an object ID");objects.push_back(it-state_.objects.begin());}
  auto previous=state_;try {
    AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
    auto frame=load_multiplex_frame(*find(index));auto layout=*state_.buckets;const auto value=features(index);
    const auto raw=masks.to(device_,at::kFloat).unsqueeze(1);
    // The demo's deferred current output omits high masks. Native history
    // retains a 1008 grid for paging/remapping, while brush logits use 1152.
    // Reconstruct this auxiliary grid after the deferred update; preflight
    // will impose overlap constraints before encoding it.
    frame.masks.high_res_mask=at::Tensor();
    MultiplexMaskUpdate request;request.encode_memory=false;
    core_->update_masks(value.interactive,value.propagation,resize(raw,1152,1152,true),objects,ids,frame,layout,request,frame_options_,mode_);
    frame.masks.high_res_mask=resize(frame.masks.low_res_mask.to(device_),1008,1008);
    if(frame_options_.temporal.select_by_score)frame.confidence=memory_confidence(frame.masks.object_logits,frame.iou);
    const auto binary=resize(raw,height_,width_,true)>.5,counts=binary.to(at::kLong).sum(0,true);
    for(size_t i=0;i<ids.size();++i){
      const auto own=binary.slice(0,i,i+1);auto& object=state_.objects[objects[i]];
      object.masks[index]=own.to(storage_);object.points.erase(index);object.refined.erase(index);
      object.video_edits[index]=at::where((counts-own.to(at::kLong))>0,-1024.,at::where(own,1024.,-1024.)).to(storage_);
    }
    // Only already pending brush outputs are suppressed, as in the source.
    for(auto& object:state_.objects)if(std::find(ids.begin(),ids.end(),object.id)==ids.end() && object.video_edits.count(index))object.video_edits[index]=at::where((counts>0).to(storage_),-1024.,object.video_edits.at(index));
    store(frame);put(std::move(frame),!state_.tracked_direction.count(index) || options_.all_edits_conditioning);
    state_.dirty.insert(index);state_.annotated.insert(index);return output(*find(index),true);
  }catch(...){state_=std::move(previous);throw;}
}
void Sam31TrackingSession::update_memory(int64_t index,const at::Tensor& masks,const at::Tensor& logits,bool reapply){
  c10::InferenceMode inference;check_frame(index);TORCH_CHECK(state_.buckets && find(index),"memory update requires an existing frame");
  auto previous=state_;try {
    const auto value=features(index);
    for(auto* group:{&state_.history.conditioning,&state_.history.tracked})for(auto& stored:*group)if(stored.index==index){
      auto frame=load_multiplex_frame(stored);core_->update_memory(value.propagation,masks,logits,frame,*state_.buckets,reapply,frame_options_,mode_);store(frame);stored=std::move(frame);
    }
  }catch(...){state_=std::move(previous);throw;}
}
void Sam31TrackingSession::preflight(bool encode){
  c10::InferenceMode inference;AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);auto previous=state_;try {
    for(auto index:state_.dirty){auto* original=find(index);if(!original)continue;auto frame=load_multiplex_frame(*original);frame.masks.low_res_mask=frame.masks.low_res_mask.to(device_).clone();
      for(size_t i=0;i<state_.objects.size();++i)if(state_.objects[i].video_edits.count(index))frame.masks.low_res_mask.slice(0,i,i+1).copy_(resize(state_.objects[i].video_edits.at(index).to(device_),288,288,true));
      frame.masks.high_res_mask=non_overlap(resize(frame.masks.low_res_mask,1008,1008));frame.memory_masks=at::Tensor();frame.memory_object_logits=at::Tensor();
      if(encode && frame_options_.temporal.memory_slots>0){const auto memory=core_->encode_history(frame,*state_.buckets,frame_options_,mode_);frame.memory=memory.features;frame.memory_position=memory.position;}
      store(frame);*original=std::move(frame);
      // Brush/point previews are temporary. Once consolidated, later edits
      // must start from the stored low logits, not reapply old +/-1024 masks.
      for(auto& object:state_.objects)object.video_edits.erase(index);
    }
    state_.dirty.clear();if(!state_.first_annotation && !state_.annotated.empty())state_.first_annotation=*state_.annotated.begin();state_.started=true;
  }catch(...){state_=std::move(previous);throw;}
}
void Sam31TrackingSession::propagate(const TrackingPropagation& request,const OutputCallback& callback){
  c10::InferenceMode inference;TORCH_CHECK(callback,"output callback is required");if(request.preflight)preflight(request.encode_memory);TORCH_CHECK(state_.buckets && !state_.history.conditioning.empty(),"add an annotation before propagation");
  int64_t first=state_.history.conditioning.front().index;for(const auto& frame:state_.history.conditioning)first=std::min(first,frame.index);
  auto start=request.start.value_or(first);if(options_.always_start_at_first_annotation && state_.first_annotation)start=*state_.first_annotation;check_frame(start);const auto steps=request.max_steps.value_or(frames_);TORCH_CHECK(steps>=0,"max_steps must be nonnegative");
  const auto end=request.reverse?start-std::min(start,steps):start+std::min(frames_-1-start,steps);cancelled_.store(false);
  for(auto index=start;;index+=request.reverse?-1:1){
    if(!state_.annotated.count(index)){
      const auto value=features(index);MultiplexFrameRequest step;step.index=index;step.frame_count=frames_;step.reverse=request.reverse;step.encode_memory=request.encode_memory;
      auto working=state_.history;
      auto frame=core_->forward(value.interactive,value.propagation,step,working,*state_.buckets,frame_options_,mode_);store(frame);put(std::move(frame),false);
    }
    state_.tracked_direction[index]=request.reverse;const auto result=output(*find(index));if(!callback(result) || cancelled_.load() || index==end)break;
  }
}
void Sam31TrackingSession::classify_inputs(){
  state_.annotated.clear();for(auto& object:state_.objects){for(const auto& [index,value]:object.points)state_.annotated.insert(index);for(const auto& [index,value]:object.masks)state_.annotated.insert(index);}
  auto frames=state_.history.conditioning;for(auto& frame:frames)if(!state_.annotated.count(frame.index)){frame.conditioning_objects.clear();put(std::move(frame),false);}
  if(state_.annotated.empty()){state_.history={};state_.dirty.clear();state_.tracked_direction.clear();state_.first_annotation.reset();state_.started=false;}
  else {
    if(!state_.first_annotation || !state_.annotated.count(*state_.first_annotation))state_.first_annotation=*state_.annotated.begin();
    // Keep remaining annotations usable after the last conditioning input is
    // cleared, including sessions whose later edits were non-conditioning.
    if(state_.history.conditioning.empty())if(auto* frame=find(*state_.annotated.begin()))put(*frame,true);
  }
}
TrackingSessionOutput Sam31TrackingSession::clear_input(int64_t index,int64_t id){
  c10::InferenceMode inference;check_frame(index);auto found=std::find_if(state_.objects.begin(),state_.objects.end(),[&](const auto& object){return object.id==id;});TORCH_CHECK(found!=state_.objects.end(),"unknown object ID");const auto object=found-state_.objects.begin();
  auto previous=state_;try {
  found->points.erase(index);found->masks.erase(index);found->video_edits.erase(index);found->refined.erase(index);
  if(auto* frame=find(index)){auto& conditions=frame->conditioning_objects;conditions.erase(std::remove(conditions.begin(),conditions.end(),object),conditions.end());state_.dirty.insert(index);}
  classify_inputs();const auto frame=find(index)?*find(index):blank(index);return output(frame,true);
  }catch(...){state_=std::move(previous);throw;}
}
void Sam31TrackingSession::remove_object(int64_t id,bool strict){
  c10::InferenceMode inference;auto found=std::find_if(state_.objects.begin(),state_.objects.end(),[&](const auto& object){return object.id==id;});if(found==state_.objects.end()){TORCH_CHECK(!strict,"unknown object ID");return;}if(state_.objects.size()==1){reset();return;}
  auto previous=state_;try {
  const auto index=found-state_.objects.begin();auto next=*state_.buckets;next.remove_objects({index});remap(next);state_.objects.erase(state_.objects.begin()+index);classify_inputs();
  }catch(...){state_=std::move(previous);throw;}
}
void Sam31TrackingSession::reset(){state_=MultiplexSessionState{};cancelled_.store(false);}
}
