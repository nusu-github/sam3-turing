#include "sam3/multiplex_temporal.h"
#include "sam3/multiplex_decoder.h"
#include "sam3/memory_encoder.h"
#include "sam3/autocast.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==6 || argc==7,"usage: sam3_multiplex_temporal STORE cpu|cuda fp32|fp16|bf16_reference OBJECTS FRAMES [math]");
    c10::InferenceMode inference;at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    if (argc==7) {
      TORCH_CHECK(std::string(argv[6])=="math","unknown attention backend");
      at::globalContext().setSDPUseFlash(false);at::globalContext().setSDPUseMemEfficient(false);
      at::globalContext().setSDPUseCuDNN(false);at::globalContext().setSDPUseMath(true);
    }
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const at::Device device(argv[2]);const std::string mode=argv[3];
    TORCH_CHECK(mode=="fp32" || mode=="fp16" || mode=="bf16_reference","unknown precision");
    const auto objects=std::stoll(argv[4]),count=std::stoll(argv[5]);TORCH_CHECK(objects>0 && count>0,"object and frame counts must be positive");
    sam3::AutocastGuard autocast(device.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
    const sam3::MultiplexMemoryConditioner temporal(store,device);
    const sam3::MultiplexPropagationHeads propagation(store,device);
    const sam3::VideoInteractiveHeads interactive(store,"sam3.1",device);
    const sam3::MaskMemoryEncoder encoder(store,"sam3.1",device);
    const auto buckets=sam3::MultiplexController().get_state(objects,device,at::kFloat,true);
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    sam3::MultiplexTemporalState state;sam3::MultiplexTemporalOptions selection;
    for (int64_t frame=0;frame<count;++frame) {
      const auto image=at::randn({1,256,72,72},options),position=at::randn_like(image);
      const auto source=image.flatten(2).permute({2,0,1}),source_pos=position.flatten(2).permute({2,0,1});
      const std::vector<at::Tensor> raw_high={at::randn({1,256,288,288},options),at::randn({1,256,144,144},options)};
      sam3::VideoMaskOutput masks;sam3::MultiplexTemporalAssembly trace;at::Tensor conditioning;
      if (frame==0) {
        auto supplied=at::zeros({objects,1,1008,1008},options);
        for (int64_t i=0;i<objects;++i) supplied[i].slice(1,100+i%50,600+i%50).slice(2,150+i%30,650+i%30).fill_(1.);
        masks=interactive.use_mask_as_output(image,interactive.project_pyramid(raw_high,mode),supplied,mode);
        conditioning=at::arange(objects,options.dtype(at::kLong));
      } else {
        const auto conditioned=temporal.forward(source,source_pos,72,72,frame,count,false,false,true,state,buckets,selection,mode,&trace);
        masks=propagation.forward(buckets,conditioned,propagation.project_pyramid(raw_high,mode),mode);
      }
      const auto memory=encoder.encode_frame(image,masks.high_res_mask,masks.object_logits,{},buckets.mux_matrix(),conditioning,mode);
      const auto quality=std::get<0>(masks.iou.max(-1));
      sam3::MultiplexTemporalFrame saved{frame,memory.features.cpu(),memory.position.cpu(),
          buckets.mux(masks.object_pointer),sam3::memory_confidence(masks.object_logits,quality).cpu(),source.cpu(),source_pos.cpu()};
      (frame==0?state.conditioning:state.tracked).push_back(std::move(saved));
      TORCH_CHECK(at::isfinite(masks.low_res_mask).all().item<bool>() && at::isfinite(memory.features).all().item<bool>(),"non-finite multiplex temporal chain output");
      TORCH_CHECK(masks.high_res_mask.sizes()==at::IntArrayRef({objects,1,1008,1008}),"incorrect object output count");
      std::cout<<"frame="<<frame<<" objects="<<objects<<" buckets="<<buckets.bucket_count()<<" fuse="<<trace.fuse
               <<" pointer_tokens="<<trace.pointer_tokens<<" masks="<<masks.low_res_mask.sizes()<<" memory="<<memory.features.sizes()<<" spatial_and_image_storage=cpu\n";
    }
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
