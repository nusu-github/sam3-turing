#include "sam3/multiplex_temporal.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
MultiplexTemporalFrame* find_frame(std::vector<MultiplexTemporalFrame>& frames,int64_t index) {
  for (auto& frame:frames) if (frame.index==index) return &frame;
  return nullptr;
}
TemporalState selection_state(const MultiplexTemporalState& state) {
  TemporalState result;
  for (const auto& frame:state.conditioning) result.conditioning.push_back({frame.index,{},{},{},frame.effective_iou});
  for (const auto& frame:state.tracked) result.tracked.push_back({frame.index,{},{},{},frame.effective_iou});
  return result;
}
}
MultiplexMemoryConditioner::MultiplexMemoryConditioner(const WeightStore& store,at::Device device)
    :attention_(store,"sam3.1",device),device_(device) {
  const std::string root="sam3.1/tracker.model.";
  weights_.emplace("maskmem_tpos_enc",store.read(root+"maskmem_tpos_enc",device));
  for (auto& [name,value]:detail::load(store,root+"obj_ptr_tpos_proj.",device)) weights_.emplace("obj_ptr_tpos_proj."+name,std::move(value));
  device_=detail::weight(weights_,"maskmem_tpos_enc").device();
}
MultiplexTemporalAssembly MultiplexMemoryConditioner::assemble(int64_t frame,int64_t frame_count,bool reverse,
    MultiplexTemporalState& state,const MultiplexState& buckets,const MultiplexTemporalOptions& options,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  const auto& temporal=detail::weight(weights_,"maskmem_tpos_enc");
  TORCH_CHECK(buckets.valid() && buckets.width()==16,"SAM3.1 temporal memory requires valid 16-slot buckets");
  TORCH_CHECK(options.memory_slots>0 && options.memory_slots<=temporal.size(0),"invalid trained memory slot count");
  MultiplexTemporalAssembly out;out.plan=plan_sam3_memory(frame,frame_count,reverse,selection_state(state),options);
  // Spatial/stride/filter selection is shared with SAM3, but the shipped 3.1
  // model includes future conditioning pointers and uses unsigned distances.
  std::vector<TemporalReference> pointers;
  if (options.use_pointers) {
    for (const auto index:out.plan.selected_conditioning) {
      if (options.only_past_pointers && (reverse?index<frame:index>frame)) continue;
      pointers.push_back({index,options.signed_pointer_time?(frame-index)*(reverse?-1:1):std::abs(frame-index),true});
    }
    for (const auto& reference:out.plan.pointers) if (!reference.conditioning) pointers.push_back(reference);
  }
  out.plan.pointers=std::move(pointers);
  const auto entry_for=[&](const TemporalReference& reference) {
    auto entry=reference.conditioning?find_frame(state.conditioning,reference.frame):find_frame(state.tracked,reference.frame);
    if (!entry && !reference.conditioning) entry=find_frame(state.conditioning,reference.frame);
    TORCH_CHECK(entry,"missing selected multiplex temporal frame");return entry;
  };
  std::vector<at::Tensor> memory,positions,images,image_positions,pointer_values;
  std::vector<int64_t> times;
  for (const auto& reference:out.plan.spatial) {
    auto entry=entry_for(reference);
    if (!entry->features.defined()) continue;
    auto features=entry->features.to(device_,entry->features.scalar_type(),true);
    if (features.dim()==5) {features=buckets.demux(features).contiguous();entry->features=features;}
    TORCH_CHECK(features.dim()==4,"stored multiplex spatial memory must be BCHW or per-slot 5D");
    if (features.size(0)==0) continue;
    memory.push_back(features.flatten(2).permute({2,0,1}));
    if (!entry->position.defined()) continue;
    auto position=entry->position.to(device_,entry->position.scalar_type(),true);
    if (position.dim()==5) {position=buckets.demux(position).contiguous();entry->position=position;}
    TORCH_CHECK(position.dim()==4,"invalid stored multiplex spatial positions");
    position=position.flatten(2).permute({2,0,1});
    int64_t slot;
    if (options.temporal_v2) slot=(reference.position<=0 || reference.position>=options.memory_slots)?options.memory_slots-1:options.memory_slots-reference.position-1;
    else slot=options.memory_slots-(reference.conditioning?0:reference.position)-1;
    const auto time=temporal[slot];
    TORCH_CHECK(entry->image.defined() && entry->image_position.defined(),"selected spatial memory requires its image stream");
    images.push_back(entry->image.to(device_));image_positions.push_back(entry->image_position.to(device_)+time);
    positions.push_back(position+time);
  }
  for (const auto& reference:out.plan.pointers) {
    const auto entry=entry_for(reference);
    if (!entry->pointer.defined()) continue;
    TORCH_CHECK(entry->pointer.dim()==3 && entry->pointer.size(0)==buckets.bucket_count() && entry->pointer.size(1)==16 && entry->pointer.size(2)==256 && entry->pointer.device()==device_,"object pointers must be [buckets,16,256] on the execution device");
    pointer_values.push_back(entry->pointer);times.push_back(reference.position);
  }
  if (!pointer_values.empty()) {
    const auto values=at::cat(pointer_values,1).transpose(0,1);
    at::Tensor position;
    const auto opts=at::TensorOptions().device(device_).dtype(at::kFloat);
    if (options.encode_pointer_time) {
      const auto offsets=at::tensor(times,opts.dtype(at::kLong))/(std::min(frame_count,options.max_pointer_frames)-1);
      auto frequencies=at::arange(128,opts);frequencies=at::pow(10000.,2*frequencies.floor_divide(2)/128);
      const auto angle=offsets.unsqueeze(-1)/frequencies;
      position=detail::linear(weights_,at::cat({angle.sin(),angle.cos()},-1),"obj_ptr_tpos_proj");
    } else position=at::zeros({static_cast<int64_t>(times.size()),256},opts);
    position=at::repeat_interleave(position.unsqueeze(1).expand({-1,buckets.bucket_count(),-1}),16,0);
    memory.push_back(values);positions.push_back(position);out.pointer_tokens=values.size(0);
  }
  if (memory.empty()) return out;
  TORCH_CHECK(!positions.empty(),"selected multiplex memory has no positions");
  const auto concatenate=[&]{
    out.memory=at::cat(memory,0);out.position=at::cat(positions,0);
    if (images.empty() || image_positions.empty()) return;
    out.image=at::cat(images,0);out.image_position=at::cat(image_positions,0);out.fuse=true;
  };
  if(device_.is_cpu()) {
    // BF16 stored spatial memory and FP16 pointers use normal type promotion.
    // CPU autocast's cat policy rejects this mixed lower-precision input.
    AutocastGuard join(at::kCPU,false,at::kBFloat16);concatenate();
  } else concatenate();
  return out;
}
at::Tensor MultiplexMemoryConditioner::forward(const at::Tensor& source,const at::Tensor& source_position,int64_t height,int64_t width,
    int64_t frame,int64_t frame_count,bool initial,bool reverse,bool use_previous,
    MultiplexTemporalState& state,const MultiplexState& buckets,const MultiplexTemporalOptions& options,const std::string& mode,MultiplexTemporalAssembly* trace) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(buckets.valid() && buckets.width()==16,"invalid temporal bucket allocation");
  const auto batch=buckets.bucket_count();
  TORCH_CHECK(height>0 && width>0 && source.dim()==3 && source.size(0)==height*width && (source.size(1)==1 || source.size(1)==batch) && source.size(2)==256 && source.device()==device_,"invalid shared/batched current-frame features");
  TORCH_CHECK(source_position.dim()==3 && source_position.size(0)==height*width && (source_position.size(1)==1 || source_position.size(1)==batch) && source_position.size(2)==256 && source_position.device()==device_,"invalid current-frame positions");
  if (trace) *trace=MultiplexTemporalAssembly();
  auto output=source.expand({-1,batch,-1});
  if (options.memory_slots!=0) {
    TORCH_CHECK(!initial && use_previous,"SAM3.1 initial/bypass frames require interactive or direct-mask heads");
    auto assembled=assemble(frame,frame_count,reverse,state,buckets,options,mode);
    if (assembled.fuse)
      output=attention_.forward(output,source_position.expand({-1,batch,-1}),assembled.memory,assembled.position,
          assembled.pointer_tokens,mode,source,assembled.image,assembled.image_position);
    if (trace) *trace=std::move(assembled);
  }
  return output.permute({1,2,0}).view({batch,256,height,width});
}
}
