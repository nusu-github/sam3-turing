#include "sam3/temporal_memory.h"
#include "sam3/video_heads.h"
#include "sam3/memory_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==5,"usage: sam3_temporal STORE cpu|cuda fp32|fp16|bf16_reference FRAMES");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const at::Device device(argv[2]);const std::string mode=argv[3];
    const auto count=std::stoll(argv[4]);TORCH_CHECK(count>0,"frame count must be positive");
    const sam3::Sam3MemoryConditioner temporal(store,device);
    const sam3::VideoInteractiveHeads heads(store,"sam3",device);
    const sam3::MaskMemoryEncoder memory(store,"sam3",device);
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    sam3::TemporalState state;sam3::TemporalOptions selection;
    for (int64_t frame=0;frame<count;++frame) {
      const auto image=at::randn({1,256,72,72},options),position=at::randn_like(image);
      const auto source=image.flatten(2).permute({2,0,1}),source_pos=position.flatten(2).permute({2,0,1});
      sam3::TemporalAssembly trace;
      const auto conditioned=temporal.forward(source,source_pos,72,72,frame,count,frame==0,false,true,state,selection,mode,&trace);
      const auto high=heads.project_pyramid({at::randn({1,256,288,288},options),at::randn({1,256,144,144},options)},mode);
      at::Tensor points,labels;
      if (frame==0) {points=at::full({1,1,2},504.,options);labels=at::ones({1,1},options.dtype(at::kInt));}
      const auto output=heads.forward(conditioned,high,points,labels,{},true,mode);
      const auto encoded=memory.encode_frame(image,output.high_res_mask,output.object_logits,{frame==0,false,0.},{},{},mode);
      const auto quality=std::get<0>(output.iou.max(-1));
      const auto confidence=sam3::memory_confidence(output.object_logits,quality);
      sam3::TemporalFrame saved{frame,encoded.features.cpu(),encoded.position.cpu(),output.object_pointer,confidence.cpu()};
      (frame==0?state.conditioning:state.tracked).push_back(std::move(saved));
      TORCH_CHECK(at::isfinite(output.low_res_mask).all().item<bool>() && at::isfinite(encoded.features).all().item<bool>(),"non-finite temporal chain output");
      std::cout<<"frame="<<frame<<" memory_frames="<<trace.plan.spatial.size()<<" pointer_tokens="<<trace.pointer_tokens
        <<" mask="<<output.low_res_mask.sizes()<<" stored_memory="<<encoded.features.sizes()<<" spatial_storage=cpu\n";
    }
    return 0;
  } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
