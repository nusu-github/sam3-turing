#include "sam3/multiplex_history.h"
#include "sam3/multiplex_storage.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <map>
#include <set>
namespace sam3 {
void remap_multiplex_history(MultiplexFrameHistory& history,const MultiplexState& source,const MultiplexState& destination,
    const MultiplexHistoryRebuilder& rebuild,const std::string& mode,const std::function<void(MultiplexFrame&)>& retain) {
  c10::InferenceMode inference;detail::check_mode(mode);
  TORCH_CHECK(source.valid() && destination.valid() && source.width()==16 && destination.width()==16,"valid 16-slot states are required");
  TORCH_CHECK(source.object_ids() && destination.object_ids(),"history remapping requires global object IDs");
  const auto& old_ids=*source.object_ids();const auto& new_ids=*destination.object_ids();
  TORCH_CHECK(std::set<int64_t>(old_ids.begin(),old_ids.end()).size()==old_ids.size() && std::set<int64_t>(new_ids.begin(),new_ids.end()).size()==new_ids.size(),"global object IDs must be unique");
  TORCH_CHECK(source.mux_matrix().device()==destination.mux_matrix().device(),"remap states must share a compute device");
  const auto device=source.mux_matrix().device();AutocastGuard autocast(device.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  std::map<int64_t,int64_t> old_index;for(size_t i=0;i<old_ids.size();++i)old_index.emplace(old_ids[i],i);
  std::vector<int64_t> old_rows,new_rows,matches(destination.bucket_count(),-1);
  for(size_t i=0;i<new_ids.size();++i)if(old_index.count(new_ids[i])){new_rows.push_back(i);old_rows.push_back(old_index.at(new_ids[i]));}
  for(int64_t dst=0;dst<destination.bucket_count();++dst)for(int64_t src=0;src<source.bucket_count();++src) {
    bool same=true;
    for(int64_t slot=0;slot<16 && same;++slot) {
      const auto a=source.assignments()[src][slot],b=destination.assignments()[dst][slot];
      if(b>=0)same=a>=0 && old_ids[a]==new_ids[b];
      else if(b==MultiplexState::padding)same=a==MultiplexState::padding;
      // Removed slots keep their historical contribution, matching source removal.
    }
    if(same){matches[dst]=src;break;}
  }
  const bool changed=std::find(matches.begin(),matches.end(),-1)!=matches.end();
  auto next=history;
  const auto rows=[&](const at::Tensor& value,double fill,bool required=false) {
    if(!value.defined())return at::Tensor();
    if(value.dim()==0 || value.size(0)!=source.object_count()) {TORCH_CHECK(!required,"historical object rows do not match the source layout");return at::Tensor();}
    auto shape=value.sizes().vec();shape[0]=destination.object_count();auto result=at::full(shape,fill,value.options());
    const auto old=at::tensor(old_rows,value.options().dtype(at::kLong)),fresh=at::tensor(new_rows,value.options().dtype(at::kLong));
    result.index_copy_(0,fresh,value.index_select(0,old));return result;
  };
  for(auto* group:{&next.conditioning,&next.tracked})for(auto& frame:*group) {
    frame=load_multiplex_frame(frame);frame.archive.reset();
    const auto old_memory=frame.memory,old_position=frame.memory_position;
    frame.masks.low_res_mask=rows(frame.masks.low_res_mask,-1024,true);frame.masks.high_res_mask=rows(frame.masks.high_res_mask,-1024);
    frame.memory_masks=rows(frame.memory_masks,-1024,true);frame.memory_object_logits=rows(frame.memory_object_logits,-1024,true);frame.masks.object_logits=rows(frame.masks.object_logits,-1024,true);frame.input_masks=rows(frame.input_masks,0);
    frame.masks.low_res_multimasks=rows(frame.masks.low_res_multimasks,-1024);frame.masks.high_res_multimasks=rows(frame.masks.high_res_multimasks,-1024);
    frame.masks.iou=rows(frame.masks.iou,0);frame.masks.object_pointer=rows(frame.masks.object_pointer,0);frame.iou=rows(frame.iou,0);
    if(frame.confidence.defined())frame.confidence=frame.iou.defined()?memory_confidence(frame.masks.object_logits,frame.iou):at::Tensor();
    if(frame.pointer.defined()) {
      const auto old=frame.pointer;
      TORCH_CHECK(old.dim()==3 && old.size(0)==source.bucket_count() && old.size(1)==16,"pointers must be [buckets,16,channels]");
      auto result=at::zeros({destination.bucket_count(),16,old.size(2)},old.options());
      for(int64_t dst=0;dst<destination.bucket_count();++dst) {
        if(matches[dst]>=0){result.select(0,dst).copy_(old.select(0,matches[dst]));continue;}
        for(int64_t slot=0;slot<16;++slot) {
          const auto index=destination.assignments()[dst][slot];if(index<0 || !old_index.count(new_ids[index]))continue;
          const auto wanted=old_index.at(new_ids[index]);
          for(int64_t src=0;src<source.bucket_count();++src)for(int64_t old_slot=0;old_slot<16;++old_slot)
            if(source.assignments()[src][old_slot]==wanted)result[dst][slot].copy_(old[src][old_slot]);
        }
      }
      frame.pointer=std::move(result);
    }
    std::vector<int64_t> conditions;
    for(auto index:frame.conditioning_objects){TORCH_CHECK(index>=0 && index<source.object_count(),"conditioning index out of range");const auto found=std::find(new_ids.begin(),new_ids.end(),old_ids[index]);if(found!=new_ids.end())conditions.push_back(found-new_ids.begin());}
    frame.conditioning_objects=std::move(conditions);
    std::pair<at::Tensor,at::Tensor> encoded;
    if(changed && (old_memory.defined() || old_position.defined())) {
      TORCH_CHECK(bool(rebuild),"changed dense-memory buckets require a rebuilder with retained image features and masks");
      encoded=rebuild(frame,destination);
    }
    const auto spatial=[&](const at::Tensor& old,const at::Tensor& rebuilt) {
      if(!old.defined())return at::Tensor();
      TORCH_CHECK(old.dim()==4 && old.size(0)==source.bucket_count(),"dense spatial history must be [buckets,channels,H,W]");
      auto shape=old.sizes().vec();shape[0]=destination.bucket_count();at::Tensor result;
      if(changed){TORCH_CHECK(rebuilt.defined() && rebuilt.sizes()==at::IntArrayRef(shape),"rebuilder returned incompatible spatial memory");result=rebuilt.to(old.options()).clone();}
      else result=at::empty(shape,old.options());
      for(size_t i=0;i<matches.size();++i)if(matches[i]>=0)result.select(0,i).copy_(old.select(0,matches[i]));
      return result;
    };
    frame.memory=spatial(old_memory,encoded.first);frame.memory_position=spatial(old_position,encoded.second);
    if(retain)retain(frame);
  }
  history=std::move(next);
}
}
