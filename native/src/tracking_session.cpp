#include "sam3/tracking_session.h"
#include "sam3/autocast.h"
#include "sam3/ops.h"
#include "detector_layers.h"
#include "frame_order.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
TrackingFrame* find(std::vector<TrackingFrame>& frames,int64_t index) {
  for (auto& frame:frames) if (frame.index==index) return &frame;return nullptr;
}
void put(std::vector<TrackingFrame>& frames,TrackingFrame value) {
  if (auto current=find(frames,value.index)) *current=std::move(value);else frames.push_back(std::move(value));
}
void erase(std::vector<TrackingFrame>& frames,int64_t index) {
  frames.erase(std::remove_if(frames.begin(),frames.end(),[&](const auto& f){return f.index==index;}),frames.end());
}
std::vector<TrackingFrame>& group(TrackingHistory& history,bool conditioning) {return conditioning?history.conditioning:history.tracked;}
at::Tensor non_overlap(const at::Tensor& masks) {
  if (masks.size(0)<=1) return masks;
  const auto ids=at::arange(masks.size(0),masks.options().dtype(at::kLong)).view({-1,1,1,1});
  return at::where(masks.argmax(0,true)==ids,masks,masks.clamp_max(-10.));
}
at::Tensor resize(const at::Tensor& masks,int64_t h,int64_t w,bool antialias=false) {
  if (masks.size(2)==h && masks.size(3)==w) return masks;
  return antialias?at::_upsample_bilinear2d_aa(masks,{h,w},false):at::upsample_bilinear2d(masks,{h,w},false);
}
}
at::Tensor postprocess_tracking_masks(const at::Tensor& input,int64_t h,int64_t w,bool overlap,int64_t area) {
  c10::InferenceMode inference;
  TORCH_CHECK(input.dim()==4 && input.size(1)==1 && h>0 && w>0,"invalid tracking output masks/size");
  auto masks=resize(input,h,w);
  if (overlap) masks=non_overlap(masks);
  if (area>0 && masks.size(0)>0) {
    const auto background=masks<=0,areas=std::get<1>(connected_components(background.to(at::kByte)));
    masks=at::where(background&(areas<=area),.1,masks);
    const auto foreground=masks>0,foreground_areas=std::get<1>(connected_components(foreground.to(at::kByte)));
    const auto threshold=foreground.sum({2,3},true,at::kInt).floor_divide(2).clamp_max(area);
    masks=at::where(foreground&(foreground_areas<=threshold),-.1,masks);
  }
  return masks;
}
Sam3TrackingSession::Sam3TrackingSession(std::shared_ptr<const Sam3TrackingFrame> core,FeatureProvider provider,
    int64_t frames,int64_t height,int64_t width,at::Device device,const std::string& mode,const TrackingSessionOptions& options)
    :core_(std::move(core)),provider_(std::move(provider)),frames_(frames),height_(height),width_(width),device_(device),storage_(device),mode_(mode),options_(options) {
  detail::check_mode(mode);TORCH_CHECK(core_ && provider_ && frames>0 && height>0 && width>0,"invalid tracking session");
  device_=at::empty({0},at::TensorOptions().device(device)).device();storage_=options.offload_state?at::Device(at::kCPU):device_;
}
void Sam3TrackingSession::check_frame(int64_t index) const {TORCH_CHECK(index>=0 && index<frames_,"frame index out of range");}
std::vector<int64_t> Sam3TrackingSession::object_ids() const {std::vector<int64_t> result;for (const auto& object:state_.objects) result.push_back(object.id);return result;}
size_t Sam3TrackingSession::object_index(int64_t id,bool create) {
  for (size_t i=0;i<state_.objects.size();++i) if (state_.objects[i].id==id) return i;
  TORCH_CHECK(create,"unknown object ID");
  TORCH_CHECK(!state_.started,"this low-level tracker requires new IDs before propagation starts; reset or use a higher-level dynamic tracker");
  state_.objects.push_back(TrackingObjectState{id});return state_.objects.size()-1;
}
TrackingFeatures Sam3TrackingSession::features(int64_t frame,int64_t batch) {
  check_frame(frame);TORCH_CHECK(batch>0,"tracking requires objects");
  if (cached_index_!=frame) {
    auto current=provider_(frame);
    TORCH_CHECK(current.image.sizes()==at::IntArrayRef({1,256,72,72}) && current.position.sizes()==current.image.sizes() && current.high.size()==2,"provider must return one image's tracking features");
    current.image=current.image.to(device_);current.position=current.position.to(device_);
    for (auto& high:current.high) high=high.to(device_);
    cached_=std::move(current);cached_index_=frame;
  }
  return {cached_.image.expand({batch,-1,-1,-1}),cached_.position.expand({batch,-1,-1,-1}),
      {cached_.high[0].expand({batch,-1,-1,-1}),cached_.high[1].expand({batch,-1,-1,-1})}};
}
at::Tensor Sam3TrackingSession::cache_position(const at::Tensor& position) {
  if (!position.defined()) return {};
  if (!position_cache_.defined()) position_cache_=position.slice(0,0,1).clone();
  return position_cache_.expand({position.size(0),-1,-1,-1});
}
TrackingFrame Sam3TrackingSession::run(int64_t frame,int64_t batch,TrackingFrameRequest request,TrackingHistory& history) {
  request.index=frame;request.frame_count=frames_;
  auto out=core_->forward(features(frame,batch),request,history,options_.frame,mode_);
  // CPU consolidation reads these tensors immediately. An asynchronous D2H
  // copy followed by CPU resize/copy is a race; Python dispatch overhead can
  // accidentally hide it in the source. Stored CPU tensors must be ready.
  if (out.memory.defined()) out.memory=out.memory.to(at::kBFloat16).to(storage_);
  out.low_mask=out.low_mask.to(storage_);out.high_mask=at::Tensor();
  out.memory_position=cache_position(out.memory_position);return out;
}
TrackingSessionOutput Sam3TrackingSession::output(const TrackingEdit& edit,bool propagation) const {
  const auto masks=(edit.video_mask.defined()?edit.video_mask:edit.frame.low_mask).to(device_,true);
  return {edit.frame.index,object_ids(),propagation?masks:at::Tensor(),
      postprocess_tracking_masks(masks,height_,width_,options_.non_overlap_output,options_.fill_hole_area),edit.frame.object_logits};
}
TrackingSessionOutput Sam3TrackingSession::add_points(int64_t frame,int64_t id,const TrackingPoints& prompt,bool clear,bool use_previous) {
  c10::InferenceMode inference;check_frame(frame);
  AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(prompt.points.defined()==prompt.labels.defined() && (prompt.points.defined() || prompt.box.defined()),"points require labels or a box");
  TORCH_CHECK(!prompt.box.defined() || clear,"a box requires clearing old points");
  auto coords=prompt.points.defined()?prompt.points.to(device_,at::kFloat):at::empty({0,2},at::TensorOptions().device(device_).dtype(at::kFloat));
  auto labels=prompt.labels.defined()?prompt.labels.to(device_,at::kInt):at::empty({0},at::TensorOptions().device(device_).dtype(at::kInt));
  if (coords.dim()==2) coords=coords.unsqueeze(0);if (labels.dim()==1) labels=labels.unsqueeze(0);
  TORCH_CHECK(coords.dim()==3 && coords.size(0)==1 && coords.size(2)==2 && labels.sizes()==coords.sizes().slice(0,2),"one object's [N,2] points and [N] labels are required");
  if (prompt.normalized) coords=coords*1008;
  if (prompt.box.defined()) {
    TORCH_CHECK(prompt.box.numel()==4,"box requires four xyxy coordinates");
    auto box=prompt.box.to(device_,at::kFloat).reshape({1,2,2});if (prompt.normalized) box=box*1008;
    coords=at::cat({box,coords},1);labels=at::cat({at::tensor({2,3},labels.options()).view({1,2}),labels},1);
  }
  auto& object=state_.objects[object_index(id)];
  if (!clear && object.points.count(frame)) {coords=at::cat({object.points.at(frame).points,coords},1);labels=at::cat({object.points.at(frame).labels,labels},1);}
  if (options_.max_points>0 && coords.size(1)>options_.max_points) {
    const auto first=options_.max_points/2,last=options_.max_points-first;
    coords=at::cat({coords.slice(1,0,first),coords.slice(1,coords.size(1)-last)},1);
    labels=at::cat({labels.slice(1,0,first),labels.slice(1,labels.size(1)-last)},1);
  }
  object.points[frame]={coords,labels,{},false};object.masks.erase(frame);
  TrackingFrameRequest request;request.initial=!state_.tracked_direction.count(frame);request.reverse=request.initial?false:state_.tracked_direction.at(frame);
  request.use_previous=use_previous;request.encode_memory=false;request.points=coords;request.labels=labels;
  const auto conditioning=request.initial || options_.all_edits_conditioning;
  auto& pending=conditioning?object.pending_conditioning:object.pending_tracked;
  const TrackingFrame* previous=pending.count(frame)?&pending.at(frame).frame:find(object.history.conditioning,frame);
  if (!previous) previous=find(object.history.tracked,frame);
  if (previous && previous->low_mask.defined()) request.previous_logits=previous->low_mask.to(device_,true).clamp(-32.,32.);
  pending[frame]={run(frame,1,request,object.history),{}};
  return output(consolidate(frame,conditioning,false,true));
}
TrackingSessionOutput Sam3TrackingSession::add_mask(int64_t frame,int64_t id,const at::Tensor& mask) {
  c10::InferenceMode inference;check_frame(frame);
  AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(mask.dim()==2 && mask.numel()>0,"mask must be [H,W]");
  const auto original=mask.to(device_,at::kFloat).unsqueeze(0).unsqueeze(0);
  const auto input=resize(original,1152,1152,true),video=resize(original,height_,width_,true)>.5;
  auto& object=state_.objects[object_index(id)];object.masks[frame]=video;object.points.erase(frame);
  TrackingFrameRequest request;request.initial=!state_.tracked_direction.count(frame);request.reverse=request.initial?false:state_.tracked_direction.at(frame);request.mask=input;request.encode_memory=false;
  const auto conditioning=request.initial || options_.all_edits_conditioning;
  auto& pending=conditioning?object.pending_conditioning:object.pending_tracked;
  auto result=run(frame,1,request,object.history);result.low_mask=at::Tensor();
  pending[frame]={std::move(result),at::where(video,1024.,-1024.)};
  for (auto& other:state_.objects) if (other.id!=id) {
    auto& edits=conditioning?other.pending_conditioning:other.pending_tracked;
    if (edits.count(frame) && edits.at(frame).video_mask.defined()) edits.at(frame).video_mask=at::where(video,-1024.,edits.at(frame).video_mask);
  }
  return output(consolidate(frame,conditioning,false,true));
}
TrackingEdit Sam3TrackingSession::consolidate(int64_t frame,bool conditioning,bool encode,bool video_resolution) {
  TORCH_CHECK(!encode || !video_resolution,"cannot encode memory at video resolution");
  const auto batch=static_cast<int64_t>(state_.objects.size()),h=video_resolution?height_:288,w=video_resolution?width_:288;
  const auto opts=at::TensorOptions().device(device_).dtype(at::kFloat);
  TrackingEdit merged;auto& result=merged.frame;result.index=frame;
  auto masks=at::full({batch,1,h,w},-1024.,opts.device(storage_));
  result.pointer=at::full({batch,256},-1024.,opts);result.object_logits=at::full({batch,1},10.,opts);
  if (options_.frame.temporal.select_by_score) result.iou=at::zeros({batch,1},opts);
  at::Tensor empty_pointer;
  for (int64_t i=0;i<batch;++i) {
    auto& object=state_.objects[i];auto& pending=conditioning?object.pending_conditioning:object.pending_tracked;
    const auto edit=pending.find(frame);const TrackingFrame* source=nullptr;at::Tensor video;
    if (edit!=pending.end()) {source=&edit->second.frame;video=edit->second.video_mask;}
    if (!source) source=find(object.history.conditioning,frame);
    if (!source) source=find(object.history.tracked,frame);
    if (!source) {
      if (encode) {
        if (!empty_pointer.defined()) {
          TrackingFrameRequest request;request.index=frame;request.frame_count=frames_;request.initial=true;request.encode_memory=false;request.mask=at::zeros({1,1,1008,1008},opts);
          TrackingHistory blank;empty_pointer=core_->forward(features(frame,1),request,blank,options_.frame,mode_).pointer;
        }
        result.pointer.slice(0,i,i+1).copy_(empty_pointer);
      }
      continue;
    }
    const auto value=video.defined()?video:source->low_mask;
    masks.slice(0,i,i+1).copy_(resize(value,h,w,video.defined()));
    result.pointer.slice(0,i,i+1).copy_(source->pointer);result.object_logits.slice(0,i,i+1).copy_(source->object_logits);
    if (result.iou.defined()) result.iou.slice(0,i,i+1).copy_(source->iou);
  }
  if (video_resolution) merged.video_mask=masks;else result.low_mask=masks;
  if (encode) {
    const auto high=non_overlap(resize(masks.to(device_,true),1008,1008));
    const auto memory=core_->encode_memory(features(frame,batch),high,result.object_logits,true,options_.frame.non_overlap_memory,mode_);
    result.memory=memory.features.to(at::kBFloat16).to(storage_);result.memory_position=cache_position(memory.position);
  }
  return merged;
}
void Sam3TrackingSession::split(const TrackingFrame& packed,bool conditioning) {
  for (size_t i=0;i<state_.objects.size();++i) {
    const auto slice=[&](const at::Tensor& value) {return value.defined()?value.slice(0,i,i+1):at::Tensor();};
    TrackingFrame out;out.index=packed.index;out.low_mask=slice(packed.low_mask);out.pointer=slice(packed.pointer);out.object_logits=slice(packed.object_logits);
    out.iou=slice(packed.iou);out.memory=slice(packed.memory);out.memory_position=slice(packed.memory_position);
    put(group(state_.objects[i].history,conditioning),std::move(out));
  }
}
void Sam3TrackingSession::clear_near(int64_t frame) {
  if (!options_.clear_near_input || (state_.objects.size()>1 && !options_.clear_near_multi_object)) return;
  const auto distance=options_.frame.temporal.stride*options_.frame.temporal.memory_slots;
  // Upstream clears per-object slices only; the packed history is retained.
  for (auto& object:state_.objects) {
    auto& frames=object.history.tracked;
    frames.erase(std::remove_if(frames.begin(),frames.end(),[&](const auto& value){return value.index>=frame-distance && value.index<=frame+distance;}),frames.end());
  }
}
void Sam3TrackingSession::update_memory(int64_t frame,const at::Tensor& masks,const at::Tensor& logits){
  c10::InferenceMode inference;check_frame(frame);const auto n=int64_t(state_.objects.size());
  TORCH_CHECK(masks.dim()==4 && masks.size(0)==n && masks.size(1)==1 && masks.size(2)>0 && masks.size(3)>0 && masks.is_floating_point() && logits.sizes()==at::IntArrayRef({n,1}) && logits.is_floating_point(),"invalid memory mask/proxy score shape");
  if(!n || (!find(state_.history.conditioning,frame) && !find(state_.history.tracked,frame)))return;
  auto previous=state_;try {
    const auto memory=core_->encode_memory(features(frame,n),masks.to(device_),logits.to(device_),false,options_.frame.non_overlap_memory,mode_);
    const auto encoded=memory.features.to(at::kBFloat16).to(storage_),position=cache_position(memory.position);
    for(bool cond:{true,false})if(auto* current=find(group(state_.history,cond),frame)){
      current->memory=encoded;current->memory_position=position;split(*current,cond);
    }
  }catch(...){state_=std::move(previous);throw;}
}
void Sam3TrackingSession::preflight(bool encode) {
  c10::InferenceMode inference;AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  state_.started=true;
  for (const bool conditioning:{false,true}) {
    detail::FrameSet frame_set;
    for (const auto& object:state_.objects) {
      std::vector<int64_t> keys;for (const auto& [frame,_]:conditioning?object.pending_conditioning:object.pending_tracked) keys.push_back(frame);
      frame_set.update(keys);
    }
    const auto frames=frame_set.order();
    auto& consolidated=conditioning?state_.consolidated_conditioning:state_.consolidated_tracked;
    consolidated.insert(frames.begin(),frames.end());
    for (const auto frame:frames) {auto merged=consolidate(frame,conditioning,encode,false);put(group(state_.history,conditioning),merged.frame);split(merged.frame,conditioning);clear_near(frame);}
    for (auto& object:state_.objects) (conditioning?object.pending_conditioning:object.pending_tracked).clear();
  }
  for (const auto& frame:state_.history.conditioning) erase(state_.history.tracked,frame.index);
  for (auto& object:state_.objects) for (const auto& frame:object.history.conditioning) erase(object.history.tracked,frame.index);
  for (const auto frame:state_.consolidated_conditioning) state_.consolidated_tracked.erase(frame);
  std::set<int64_t> inputs,consolidated=state_.consolidated_conditioning;consolidated.insert(state_.consolidated_tracked.begin(),state_.consolidated_tracked.end());
  for (const auto& object:state_.objects) {for (const auto& [frame,_]:object.points) inputs.insert(frame);for (const auto& [frame,_]:object.masks) inputs.insert(frame);}
  TORCH_CHECK(inputs==consolidated,"session inputs and consolidated frame indices disagree");
  if (!state_.first_annotation && !inputs.empty()) state_.first_annotation=*inputs.begin();
  if (!state_.first_annotation || !find(state_.history.conditioning,*state_.first_annotation)) {
    state_.first_annotation.reset();for (const auto& frame:state_.history.conditioning) if (!state_.first_annotation || frame.index<*state_.first_annotation) state_.first_annotation=frame.index;
  }
}
void Sam3TrackingSession::propagate(const TrackingPropagation& request,const OutputCallback& callback) {
  c10::InferenceMode inference;TORCH_CHECK(callback,"output callback is required");
  AutocastGuard autocast(device_.type(),mode_!="fp32",mode_=="fp16"?at::kHalf:at::kBFloat16);
  cancelled_.store(false);if (request.preflight) preflight();
  TORCH_CHECK(!state_.history.conditioning.empty(),"add prompts before propagation");
  auto start=options_.always_start_at_first_annotation?state_.first_annotation:request.start;
  if (!start) {start=state_.history.conditioning.front().index;for (const auto& frame:state_.history.conditioning) start=std::min(*start,frame.index);}
  check_frame(*start);const auto steps=request.max_steps.value_or(frames_);TORCH_CHECK(steps>=0,"negative propagation range");
  const auto end=request.reverse?*start-std::min(steps,*start):*start+std::min(steps,frames_-1-*start);
  for (auto frame=*start;request.reverse?frame>=end:frame<=end;frame+=request.reverse?-1:1) {
    if (cancelled_.load()) break;
    const auto conditioning=state_.consolidated_conditioning.count(frame)>0;
    TrackingFrame current;
    if (conditioning || state_.consolidated_tracked.count(frame)) {
      auto entry=find(group(state_.history,conditioning),frame);TORCH_CHECK(entry,"missing consolidated frame");current=*entry;
      if (conditioning) clear_near(frame);
    } else {
      TrackingFrameRequest input;input.reverse=request.reverse;input.encode_memory=request.encode_memory;
      current=run(frame,state_.objects.size(),input,state_.history);put(state_.history.tracked,current);
    }
    split(current,conditioning);state_.tracked_direction[frame]=request.reverse;
    if (!callback(output({current,{}},true))) break;
  }
}
void Sam3TrackingSession::reset_results() {
  for (auto& object:state_.objects) {object.points.clear();object.masks.clear();object.history={};object.pending_conditioning.clear();object.pending_tracked.clear();}
  state_.history={};state_.consolidated_conditioning.clear();state_.consolidated_tracked.clear();state_.tracked_direction.clear();state_.first_annotation.reset();state_.started=false;
}
void Sam3TrackingSession::reset() {reset_results();state_.objects.clear();}
void Sam3TrackingSession::clear_input_impl(int64_t frame,size_t index) {
  auto& object=state_.objects[index];object.points.erase(frame);object.masks.erase(frame);object.pending_conditioning.erase(frame);object.pending_tracked.erase(frame);
  for (const auto& entry:state_.objects) if (entry.points.count(frame) || entry.masks.count(frame)) return;
  state_.consolidated_conditioning.erase(frame);state_.consolidated_tracked.erase(frame);
  if (auto out=find(state_.history.conditioning,frame)) {put(state_.history.tracked,*out);erase(state_.history.conditioning,frame);state_.tracked_direction.erase(frame);}
  for (auto& entry:state_.objects) if (auto out=find(entry.history.conditioning,frame)) {put(entry.history.tracked,*out);erase(entry.history.conditioning,frame);}
  if (state_.history.conditioning.empty()) reset_results();
}
TrackingSessionOutput Sam3TrackingSession::clear_input(int64_t frame,int64_t id) {
  c10::InferenceMode inference;check_frame(frame);const auto index=object_index(id);
  clear_input_impl(frame,index);bool conditioning=false;
  for (const auto& object:state_.objects) conditioning|=object.pending_conditioning.count(frame)>0;
  return output(consolidate(frame,conditioning,false,true));
}
std::vector<TrackingSessionOutput> Sam3TrackingSession::remove_object(int64_t id,bool strict) {
  c10::InferenceMode inference;std::vector<TrackingSessionOutput> results;
  auto index=state_.objects.size();for (size_t i=0;i<state_.objects.size();++i) if (state_.objects[i].id==id) index=i;
  if (index==state_.objects.size()) {TORCH_CHECK(!strict,"unknown object ID");return results;}
  if (state_.objects.size()==1) {reset();return results;}
  std::vector<int64_t> points,masks;for (const auto& [frame,_]:state_.objects[index].points) points.push_back(frame);for (const auto& [frame,_]:state_.objects[index].masks) masks.push_back(frame);
  const auto inputs=detail::frame_set_order({points,masks},true);
  for (const auto frame:inputs) clear_input_impl(frame,index);
  std::vector<int64_t> keep;for (size_t i=0;i<state_.objects.size();++i) if (i!=index) keep.push_back(i);
  state_.objects.erase(state_.objects.begin()+index);
  for (const bool conditioning:{true,false}) for (auto& frame:group(state_.history,conditioning)) {
    const auto select=[&](const at::Tensor& value) {return value.defined()?value.index_select(0,at::tensor(keep,value.options().dtype(at::kLong))):at::Tensor();};
    frame.memory=select(frame.memory);frame.memory_position=cache_position(select(frame.memory_position));frame.low_mask=select(frame.low_mask);
    frame.pointer=select(frame.pointer);frame.object_logits=select(frame.object_logits);frame.iou=select(frame.iou);
    if (options_.frame.temporal.select_by_score) frame.confidence=memory_confidence(frame.object_logits,frame.iou);
    split(frame,conditioning);
  }
  for (const auto frame:inputs) {
    bool conditioning=false;for (const auto& object:state_.objects) conditioning|=object.pending_conditioning.count(frame)>0;
    results.push_back(output(consolidate(frame,conditioning,false,true)));
  }
  return results;
}
}
