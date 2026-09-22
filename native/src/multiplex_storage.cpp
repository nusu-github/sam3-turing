#include "sam3/multiplex_storage.h"
#include <algorithm>
namespace sam3 {
namespace {
std::map<std::string,at::Tensor*> fields(MultiplexFrame& f,uint32_t mask){
  std::map<std::string,at::Tensor*> out;
  if(mask&(history_masks|history_output))out.insert({{"low",&f.masks.low_res_mask},{"scores",&f.masks.object_logits}});
  if(mask&history_masks)out.insert({{"high",&f.masks.high_res_mask},{"input",&f.input_masks},{"ious",&f.iou},
      {"low_multi",&f.masks.low_res_multimasks},{"high_multi",&f.masks.high_res_multimasks},{"mask_ious",&f.masks.iou},{"mask_pointer",&f.masks.object_pointer}});
  if(mask&history_spatial)out.insert({{"memory",&f.memory},{"position",&f.memory_position},{"image",&f.image},{"image_position",&f.image_position}});
  if(mask&history_pointers)out.emplace("pointer",&f.pointer);
  return out;
}
}
MultiplexFrame load_multiplex_frame(const MultiplexFrame& frame,uint32_t mask){
  auto result=frame;if(result.archive)for(auto& [name,value]:fields(result,mask))if(!value->defined())*value=result.archive->read(name);if(mask==history_all)result.archive.reset();return result;
}
MultiplexFrameHistory load_selected_multiplex_history(const MultiplexFrameHistory& history,int64_t index,int64_t frame_count,bool reverse,const MultiplexTemporalOptions& options){
  auto working=history;bool archived=false;
  for(const auto* group:{&history.conditioning,&history.tracked})for(const auto& frame:*group)archived=archived || bool(frame.archive);
  if(!archived || options.memory_slots==0)return working;
  TemporalState metadata;
  for(const bool cond:{true,false})for(const auto& past:cond?working.conditioning:working.tracked)
    (cond?metadata.conditioning:metadata.tracked).push_back({past.index,{},{},{},past.confidence});
  const auto plan=plan_sam3_memory(index,frame_count,reverse,metadata,options);
  const auto load=[&](int64_t wanted,bool cond,uint32_t fields){
    auto* group=cond?&working.conditioning:&working.tracked;
    auto found=std::find_if(group->begin(),group->end(),[&](const auto& past){return past.index==wanted;});
    if(found==group->end() && !cond){group=&working.conditioning;found=std::find_if(group->begin(),group->end(),[&](const auto& past){return past.index==wanted;});}
    TORCH_CHECK(found!=group->end(),"selected history frame is missing");*found=load_multiplex_frame(*found,fields);
  };
  for(const auto& ref:plan.spatial)load(ref.frame,ref.conditioning,history_spatial);
  if(options.use_pointers){
    for(auto past:plan.selected_conditioning)if(!options.only_past_pointers || (reverse?past>=index:past<=index))load(past,true,history_pointers);
    for(const auto& ref:plan.pointers)if(!ref.conditioning)load(ref.frame,false,history_pointers);
  }
  return working;
}
void archive_multiplex_frame(MultiplexFrame& frame,const std::filesystem::path& directory){
  // All modified fields must be resident when replacing an old archive.
  auto loaded=load_multiplex_frame(frame);std::map<std::string,at::Tensor> values;auto tensors=fields(loaded,history_all);
  for(auto& [name,value]:tensors)values.emplace(name,*value);
  auto archive=TensorArchive::write(directory,values);
  for(auto& [name,value]:tensors)*value=at::Tensor();loaded.archive=std::move(archive);frame=std::move(loaded);
}
}
