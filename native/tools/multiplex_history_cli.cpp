#include "sam3/multiplex_history.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <numeric>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==5,"usage: sam3_multiplex_history STORE cpu|cuda fp32|fp16|bf16_reference ADDED_OBJECTS");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[2]);const std::string mode=argv[3];const auto added=std::stoll(argv[4]);TORCH_CHECK(added>0,"positive added count required");
    const sam3::Sam31TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(argv[1])),device);
    const auto old_state=sam3::MultiplexController().get_state(3,device,at::kFloat,false,std::vector<int64_t>{10,20,30});auto next_state=old_state;
    const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
    const auto features=[&]{return sam3::TrackingFeatures{at::randn({1,256,72,72},opts),at::randn({1,256,72,72},opts),{at::randn({1,32,288,288},opts),at::randn({1,64,144,144},opts)}};};
    sam3::MultiplexFrameHistory history;sam3::MultiplexFrameOptions policy;sam3::MultiplexFrameRequest request;request.frame_count=3;
    for(int64_t index=0;index<2;++index) {
      request.index=index;request.initial=index==0;if(index==0){request.points=at::rand({3,2,2},opts)*1008;request.labels=at::ones({3,2},opts.dtype(at::kInt));}else{request.points=at::Tensor();request.labels=at::Tensor();}
      auto frame=core.forward(features(),features(),request,history,old_state,policy,mode);
      frame.memory=frame.memory.to(at::kBFloat16).cpu();frame.memory_position=frame.memory_position.cpu();frame.image=frame.image.cpu();frame.masks.high_res_mask=frame.masks.high_res_mask.cpu();
      (index==0?history.conditioning:history.tracked).push_back(std::move(frame));
    }
    std::vector<int64_t> indices(added),ids(added);std::iota(indices.begin(),indices.end(),3);std::iota(ids.begin(),ids.end(),100);
    next_state.add_objects(indices,ids,true,true);const auto original_memory=history.conditioning[0].memory.clone(),original_pointer=history.conditioning[0].pointer.clone();
    int64_t attempts=0;bool rejected=false;
    try {sam3::remap_multiplex_history(history,old_state,next_state,[&](const auto& frame,const auto& state){TORCH_CHECK(++attempts<2,"injected second-frame rebuild failure");auto shape=frame.memory.sizes().vec();shape[0]=state.bucket_count();return std::make_pair(at::zeros(shape,frame.memory.options()),at::zeros(shape,frame.memory_position.options()));},mode);}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected && attempts==2 && history.conditioning[0].masks.low_res_mask.size(0)==3 && at::equal(history.conditioning[0].memory,original_memory),"failed multi-frame remap changed history");
    int64_t rebuilds=0;
    sam3::remap_multiplex_history(history,old_state,next_state,[&](const auto& frame,const auto& state){++rebuilds;const auto memory=core.encode_history(frame,state,policy,mode);return std::make_pair(memory.features,memory.position);},mode);
    TORCH_CHECK(rebuilds==2 && at::equal(history.conditioning[0].memory.select(0,0),original_memory.select(0,0)) && at::equal(history.conditioning[0].pointer.select(0,0),original_pointer.select(0,0)),"unchanged bucket was not preserved");
    TORCH_CHECK(history.conditioning[0].memory.device().is_cpu() && history.conditioning[0].memory.scalar_type()==at::kBFloat16 && (history.tracked[0].masks.high_res_mask.slice(0,3)==-1024).all().template item<bool>(),"storage or historical absence changed");
    request.index=2;request.initial=false;const auto interactive=features(),propagation=features();auto frame=core.forward(interactive,propagation,request,history,next_state,policy,mode);
    sam3::MultiplexMaskUpdate update;auto masks=at::rand({added,1,1008,1008},opts)>.5;
    core.update_masks(interactive,propagation,masks,indices,ids,frame,next_state,update,policy,mode);
    TORCH_CHECK(frame.masks.low_res_mask.size(0)==3+added && frame.memory.size(0)==next_state.bucket_count() && at::isfinite(frame.masks.low_res_mask).all().item<bool>(),"remapped history failed propagation/reconditioning");
    std::cout<<"native history remap passed: 2 retained frames, 3 -> "<<next_state.object_count()<<" objects, "<<next_state.bucket_count()<<" buckets; atomic rollback, BF16 CPU storage, exact unchanged bucket, re-encoding, propagation and reconditioning; no Python\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
