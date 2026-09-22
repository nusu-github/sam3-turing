#include "sam3/multiplex_frame.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <numeric>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==6 || argc==7,"usage: sam3_multiplex_frame STORE cpu|cuda fp32|fp16|bf16_reference OBJECTS FRAMES [math]");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    if(argc==7){TORCH_CHECK(std::string(argv[6])=="math","unknown backend");at::globalContext().setSDPUseFlash(false);at::globalContext().setSDPUseMemEfficient(false);at::globalContext().setSDPUseCuDNN(false);at::globalContext().setSDPUseMath(true);}
    const at::Device device(argv[2]);const std::string mode=argv[3];const auto objects=std::stoll(argv[4]),count=std::stoll(argv[5]);
    TORCH_CHECK(objects>=2 && count>=3,"probe requires at least two objects and three frames");
    const sam3::Sam31TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(argv[1])),device);
    const auto buckets=sam3::MultiplexController().get_state(objects,device,at::kFloat,true);
    const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
    sam3::MultiplexFrameHistory history;sam3::MultiplexFrameOptions policy;policy.offload_output=true;policy.temporal.memory_slots=3;policy.temporal.max_pointer_frames=4;
    const auto features=[&]{return sam3::TrackingFeatures{at::randn({1,256,72,72},opts),at::randn({1,256,72,72},opts),{at::randn({1,32,288,288},opts),at::randn({1,64,144,144},opts)}};};
    for(int64_t index=0;index<count;++index) {
      const auto interactive=features(),propagation=features();sam3::MultiplexFrameRequest request;request.index=index;request.frame_count=count;request.initial=index==0;
      if(index==0){request.points=at::rand({objects,17,2},opts)*1008;request.labels=at::ones({objects,17},opts.dtype(at::kInt));request.encode_memory=false;
        const auto preview=core.forward(interactive,propagation,request,history,buckets,policy,mode);
        request.previous_logits=preview.masks.low_res_mask.to(device);std::vector<int64_t> ids(objects);std::iota(ids.begin(),ids.end(),0);request.objects_to_interact=ids;request.encode_memory=true;
      }else if(index==1){request.objects_to_interact=std::vector<int64_t>{objects-1,0};request.points=at::rand({2,1,2},opts)*1008;request.labels=at::ones({2,1},opts.dtype(at::kInt));}
      auto out=core.forward(interactive,propagation,request,history,buckets,policy,mode);
      TORCH_CHECK(out.masks.low_res_mask.sizes()==at::IntArrayRef({objects,1,288,288}) && out.masks.high_res_mask.sizes()==at::IntArrayRef({objects,1,1008,1008}),"incorrect object/mask output size");
      TORCH_CHECK(out.pointer.sizes()==at::IntArrayRef({buckets.bucket_count(),16,256}) && out.memory.sizes()==at::IntArrayRef({buckets.bucket_count(),256,72,72}),"incorrect multiplexed memory");
      TORCH_CHECK(out.memory.device().is_cpu() && out.image.device().is_cpu() && at::isfinite(out.masks.low_res_mask).all().item<bool>(),"invalid offloaded output");
      TORCH_CHECK(out.conditioning_objects.size()==size_t(index==0?objects:index==1?2:0),"incorrect conditioning object set");
      std::cout<<"frame="<<index<<" objects="<<objects<<" buckets="<<buckets.bucket_count()<<" conditioning="<<out.conditioning_objects.size()<<" memory="<<out.memory.sizes()<<" storage=cpu\n";
      (index==0?history.conditioning:history.tracked).push_back(std::move(out));
    }
    std::cout<<"native SAM3.1 frame chain passed; synthetic features, no Python\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
