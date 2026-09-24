#include "sam3/temporal_memory.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <set>
namespace sam3 {
namespace {
const TemporalFrame* find(const std::vector<TemporalFrame>& frames,int64_t index) {
  for (const auto& frame:frames) if (frame.index==index) return &frame;
  return nullptr;
}
bool contains(const std::vector<int64_t>& values,int64_t value) {return std::find(values.begin(),values.end(),value)!=values.end();}
int64_t floor_div(int64_t a,int64_t positive_b) {return a/positive_b-(a%positive_b<0);}
void validate_frames(const std::vector<TemporalFrame>& frames) {
  std::set<int64_t> ids;
  for (const auto& frame:frames) TORCH_CHECK(frame.index>=0 && ids.insert(frame.index).second,"duplicate or negative temporal frame index");
}
const TemporalFrame* lookup(const TemporalState& state,const std::vector<int64_t>& unselected,int64_t index) {
  if (const auto frame=find(state.tracked,index)) return frame;
  return contains(unselected,index)?find(state.conditioning,index):nullptr;
}
}
std::pair<std::vector<int64_t>,std::vector<int64_t>> select_conditioning_frames(int64_t current,const std::vector<int64_t>& order,int64_t limit,bool keep_first) {
  TORCH_CHECK(limit>=-1,"invalid conditioning frame limit");
  TORCH_CHECK(std::set<int64_t>(order.begin(),order.end()).size()==order.size(),"duplicate conditioning frames");
  if (limit==-1 || static_cast<int64_t>(order.size())<=limit) return {order,{}};
  TORCH_CHECK(limit>=2,"source selection requires at least two conditioning frames");
  std::vector<int64_t> selected,unselected;
  const auto insert=[&](int64_t value) {if (!contains(selected,value)) selected.push_back(value);};
  std::optional<int64_t> before,after,first_before,last_after;
  for (const auto t:order) {
    if (t<current) { if (!before || t>*before) before=t;if (!first_before || t<*first_before) first_before=t; }
    if (t>=current && (!after || t<*after)) after=t;
    if (t>current && (!last_after || t>*last_after)) last_after=t;
  }
  if (keep_first) { if (first_before) insert(*first_before);else if (last_after) insert(*last_after); }
  if (before) insert(*before);if (after) insert(*after);
  std::vector<int64_t> remaining;
  for (const auto t:order) if (!contains(selected,t)) remaining.push_back(t);
  std::stable_sort(remaining.begin(),remaining.end(),[&](int64_t a,int64_t b) {return std::abs(a-current)<std::abs(b-current);});
  const auto count=limit-static_cast<int64_t>(selected.size());
  // Preserve Python's [:negative] behavior in keep-first/limit-two edge cases.
  const auto take=count>=0?std::min<int64_t>(count,remaining.size()):std::max<int64_t>(0,static_cast<int64_t>(remaining.size())+count);
  selected.insert(selected.end(),remaining.begin(),remaining.begin()+take);
  for (const auto t:order) if (!contains(selected,t)) unselected.push_back(t);
  return {selected,unselected};
}
TemporalPlan plan_sam3_memory(int64_t frame,int64_t frame_count,bool reverse,const TemporalState& state,const TemporalOptions& options) {
  TORCH_CHECK(frame_count>0 && frame>=0 && frame<frame_count && options.stride>0 && options.memory_slots>=0 && options.max_pointer_frames>=0,"invalid temporal planning options");
  validate_frames(state.conditioning);validate_frames(state.tracked);
  TORCH_CHECK(!state.conditioning.empty(),"tracking memory requires a conditioning frame");
  std::vector<int64_t> order;for (const auto& entry:state.conditioning) order.push_back(entry.index);
  TemporalPlan plan;std::tie(plan.selected_conditioning,plan.unselected_conditioning)=select_conditioning_frames(frame,order,options.max_conditioning_frames,options.keep_first);
  const int64_t sign=reverse?-1:1,max_pointers=std::min(frame_count,options.max_pointer_frames);
  if (options.select_by_score && !((frame==0 && !reverse) || (frame==frame_count-1 && reverse))) {
    const auto start=frame+(reverse?1:-1),end=reverse?frame_count:0,step=reverse?options.stride:-options.stride;
    for (auto i=start;reverse?i<end:i>end;i+=step) {
      const auto entry=find(state.tracked,i);
      if (!entry || !entry->effective_iou.defined()) continue;
      TORCH_CHECK(entry->effective_iou.numel()==1,"memory confidence must be scalar");
      if ((entry->effective_iou>options.score_threshold).item<bool>()) plan.filtered_frames.insert(plan.filtered_frames.begin(),i);
      if (static_cast<int64_t>(plan.filtered_frames.size())>=max_pointers-1) break;
    }
    if (!contains(plan.filtered_frames,start)) plan.filtered_frames.push_back(start);
  }
  for (const auto t:plan.selected_conditioning) {
    plan.spatial.push_back({t,(frame-t)*sign,true});
    if (reverse?t>=frame:t<=frame) plan.pointers.push_back({t,(frame-t)*sign,true});
  }
  for (int64_t pos=1;pos<options.memory_slots;++pos) {
    const auto relative=options.memory_slots-pos;
    int64_t previous;
    if (options.select_by_score) {
      if (relative>static_cast<int64_t>(plan.filtered_frames.size())) continue;
      previous=plan.filtered_frames[plan.filtered_frames.size()-relative];
    } else if (relative==1) previous=frame+(reverse?1:-1);
    else if (!reverse) previous=floor_div(frame-2,options.stride)*options.stride-(relative-2)*options.stride;
    else previous=-floor_div(-(frame+2),options.stride)*options.stride+(relative-2)*options.stride;
    if (lookup(state,plan.unselected_conditioning,previous)) plan.spatial.push_back({previous,pos,false});
  }
  for (int64_t diff=1;diff<max_pointers;++diff) {
    int64_t t;
    if (!options.select_by_score) {
      t=frame+(reverse?diff:-diff);if (t<0 || t>=frame_count) break;
    } else {
      // The source deliberately excludes the oldest filtered entry here.
      if (diff>=static_cast<int64_t>(plan.filtered_frames.size())) break;
      t=plan.filtered_frames[plan.filtered_frames.size()-diff];
    }
    if (lookup(state,plan.unselected_conditioning,t)) plan.pointers.push_back({t,diff,false});
  }
  return plan;
}
at::Tensor memory_confidence(const at::Tensor& logits,const at::Tensor& iou) {
  c10::InferenceMode inference;
  const auto confidence=at::where(logits>0.,logits.sigmoid()*2.-1.,at::zeros_like(logits));
  return (confidence*iou).mean();
}
Sam3MemoryConditioner::Sam3MemoryConditioner(const WeightStore& store,at::Device device)
    :attention_(store,"sam3",device),device_(device) {
  const std::string root="sam3/tracker.";
  for (const std::string name:{"maskmem_tpos_enc","no_mem_embed"}) weights_.emplace(name,store.read(root+name,device));
  for (auto& [name,value]:store.read_prefix(root+"obj_ptr_tpos_proj.",device)) weights_.emplace(name.substr(root.size()),std::move(value));
  for (const std::string name:{"cond_frame_spatial_embedding","cond_frame_obj_ptr_embedding"})
    if (store.records().count(root+name)) weights_.emplace(name,store.read(root+name,device));
  device_=detail::weight(weights_,"no_mem_embed").device();
}
TemporalAssembly Sam3MemoryConditioner::assemble(int64_t frame,int64_t frame_count,bool reverse,const TemporalState& state,
    const TemporalOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  const auto& temporal=detail::weight(weights_,"maskmem_tpos_enc");
  TORCH_CHECK(options.memory_slots>0 && options.memory_slots<=temporal.size(0),"invalid trained memory slot count");
  TemporalAssembly out;out.plan=plan_sam3_memory(frame,frame_count,reverse,state,options);
  std::vector<at::Tensor> features,positions,pointers;std::vector<int64_t> times,conditioned;
  const auto entry_for=[&](const TemporalReference& reference) {
    const auto entry=reference.conditioning?find(state.conditioning,reference.frame):lookup(state,out.plan.unselected_conditioning,reference.frame);
    TORCH_CHECK(entry,"missing selected temporal frame");return entry;
  };
  int64_t batch=0;
  for (const auto& reference:out.plan.spatial) {
    const auto entry=entry_for(reference);
    TORCH_CHECK(entry->features.dim()==4 && entry->features.size(1)==64 && entry->position.dim()==4,"invalid stored SAM3 spatial memory");
    if (!batch) batch=entry->features.size(0);
    TORCH_CHECK(entry->features.size(0)==batch,"inconsistent spatial memory batch");
    features.push_back(entry->features.to(device_,entry->features.scalar_type(),true).flatten(2).permute({2,0,1}));
    auto position=entry->position.to(device_).flatten(2).permute({2,0,1});
    if (reference.conditioning && weights_.count("cond_frame_spatial_embedding")) position=position+detail::weight(weights_,"cond_frame_spatial_embedding");
    position=position+temporal[options.memory_slots-(reference.conditioning?0:reference.position)-1];
    positions.push_back(position);
  }
  for (const auto& reference:out.plan.pointers) {
    const auto entry=entry_for(reference);
    TORCH_CHECK(entry->pointer.dim()==2 && entry->pointer.size(1)==256 && entry->pointer.device()==device_,"stored object pointers must be [B,256] on the execution device");
    if (!batch) batch=entry->pointer.size(0);
    TORCH_CHECK(entry->pointer.size(0)==batch,"inconsistent pointer batch");
    pointers.push_back(entry->pointer);times.push_back(reference.position);conditioned.push_back(reference.conditioning);
  }
  if (!pointers.empty()) {
    auto values=at::stack(pointers,0);
    if (weights_.count("cond_frame_obj_ptr_embedding")) values=values+detail::weight(weights_,"cond_frame_obj_ptr_embedding")*
      at::tensor(conditioned,values.options().dtype(at::kLong)).unsqueeze(-1).unsqueeze(-1).to(at::kFloat);
    const auto offsets=at::tensor(times,values.options().dtype(at::kLong))/ (std::min(frame_count,options.max_pointer_frames)-1);
    auto frequencies=at::arange(128,values.options().dtype(at::kFloat));
    frequencies=at::pow(10000.,2*frequencies.floor_divide(2)/128);
    const auto angle=offsets.unsqueeze(-1)/frequencies;
    auto position=detail::linear(weights_,at::cat({angle.sin(),angle.cos()},-1),"obj_ptr_tpos_proj").unsqueeze(1).expand({-1,batch,-1});
    values=values.reshape({-1,batch,4,64}).permute({0,2,1,3}).flatten(0,1);
    position=at::repeat_interleave(position,4,0);
    features.push_back(values);positions.push_back(position);out.pointer_tokens=values.size(0);
  }
  TORCH_CHECK(!features.empty(),"no spatial or pointer memory selected");
  if(device_.is_cpu()) {
    // Stored memory remains BF16 even in an FP16 session. CPU autocast's cat
    // policy rejects that mixed lower-precision input. Concatenation performs
    // no neural arithmetic: use normal type promotion, as on the CUDA path.
    AutocastGuard concatenate(at::kCPU,false,at::kBFloat16);
    out.memory=at::cat(features,0);out.position=at::cat(positions,0);
  } else {out.memory=at::cat(features,0);out.position=at::cat(positions,0);}
  return out;
}
at::Tensor Sam3MemoryConditioner::forward(const at::Tensor& source,const at::Tensor& source_position,int64_t height,int64_t width,
    int64_t frame,int64_t frame_count,bool initial,bool reverse,bool use_previous,const TemporalState& state,
    const TemporalOptions& options,const std::string& mode,TemporalAssembly* trace) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(height>0 && width>0 && source.dim()==3 && source.size(0)==height*width && source.size(1)>0 && source.size(2)==256 && source.device()==device_,"invalid current-frame features");
  if (trace) *trace=TemporalAssembly();
  auto output=source;
  if (options.memory_slots==0) {}
  else if (initial || !use_previous) output=source+detail::weight(weights_,"no_mem_embed");
  else {
    auto assembled=assemble(frame,frame_count,reverse,state,options,mode);
    output=attention_.forward(source,source_position,assembled.memory,assembled.position,assembled.pointer_tokens,mode);
    if (trace) *trace=std::move(assembled);
  }
  return output.permute({1,2,0}).view({source.size(1),256,height,width});
}
}
