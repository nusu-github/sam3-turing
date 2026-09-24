#include "sam3/tracking_frame.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==6,"usage: sam3_tracking_frame STORE cpu|cuda fp32|fp16|bf16_reference OBJECTS FRAMES");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const at::Device device(argv[2]);const std::string mode=argv[3];
    const auto objects=std::stoll(argv[4]),count=std::stoll(argv[5]);TORCH_CHECK(objects>0 && count>0,"object and frame counts must be positive");
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    const sam3::Sam3TrackingFrame core(store,device);sam3::TrackingHistory history;
    sam3::TrackingFrameOptions policy;policy.offload_output=true;
    for (int64_t frame=0;frame<count;++frame) {
      const auto high=core.project_pyramid({at::randn({1,256,288,288},options),at::randn({1,256,144,144},options)},mode);
      const sam3::TrackingFeatures features{at::randn({1,256,72,72},options).expand({objects,-1,-1,-1}),
          at::randn({1,256,72,72},options).expand({objects,-1,-1,-1}),{high[0].expand({objects,-1,-1,-1}),high[1].expand({objects,-1,-1,-1})}};
      sam3::TrackingFrameRequest request;request.index=frame;request.frame_count=count;request.initial=frame==0;
      if (frame==0) {
        request.points=at::full({objects,1,2},504.,options);request.labels=at::ones({objects,1},options.dtype(at::kInt));request.encode_memory=false;
        const auto preview=core.forward(features,request,history,policy,mode);
        TORCH_CHECK(!preview.memory.defined(),"preview unexpectedly encoded memory");
        request.previous_logits=preview.low_mask.to(device).clamp(-32.,32.);
        request.points=at::cat({request.points,request.points+20.},1);request.labels=at::ones({objects,2},options.dtype(at::kInt));request.encode_memory=true;
      }
      auto output=core.forward(features,request,history,policy,mode);
      TORCH_CHECK(output.low_mask.sizes()==at::IntArrayRef({objects,1,288,288}) && output.memory.sizes()==at::IntArrayRef({objects,64,72,72}),"incorrect tracking output shape");
      TORCH_CHECK(output.low_mask.device().is_cpu() && output.memory.device().is_cpu(),"output offload failed");
      TORCH_CHECK(at::isfinite(output.low_mask).all().item<bool>() && at::isfinite(output.memory).all().item<bool>(),"non-finite tracking frame output");
      std::cout<<"frame="<<frame<<" objects="<<objects<<" refined="<<(frame==0)<<" masks="<<output.low_mask.sizes()<<" memory="<<output.memory.sizes()<<" storage=cpu\n";
      (frame==0?history.conditioning:history.tracked).push_back(std::move(output));
    }
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
