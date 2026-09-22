#include "sam3/video_heads.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==7,"usage: sam3_video_heads STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference BATCH POINTS");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);
    const auto batch=std::stoll(argv[5]),count=std::stoll(argv[6]),image_batch=model=="sam3"?batch:1;
    TORCH_CHECK(batch>0 && count>=0,"invalid prompt dimensions");
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    const sam3::VideoInteractiveHeads heads(store,model,device);
    const auto high=heads.project_pyramid({at::randn({image_batch,256,288,288},options),at::randn({image_batch,256,144,144},options)},mode);
    const auto image=at::randn({image_batch,256,72,72},options);
    const auto points=at::rand({batch,count,2},options)*1008;
    const auto labels=at::arange(batch*count,options.dtype(at::kInt)).reshape({batch,count}).remainder(5)-1;
    for (bool multi:{false,true}) {
      const auto out=heads.forward(image,high,points,labels,{},multi,mode);
      TORCH_CHECK(out.low_res_multimasks.sizes()==at::IntArrayRef({batch,multi?3:1,288,288}),"invalid mask shape");
      TORCH_CHECK(out.object_pointer.sizes()==at::IntArrayRef({batch,256}) && at::isfinite(out.object_pointer).all().item<bool>(),"invalid pointer");
      std::cout<<"multimask="<<multi<<" masks="<<out.low_res_multimasks.sizes()<<" selected="<<out.high_res_mask.sizes()<<" pointer="<<out.object_pointer.sizes()<<'\n';
    }
    auto mask=at::rand({batch,1,1152,1152},options)>.5;mask[0].zero_();
    const auto direct=heads.use_mask_as_output(image,high,mask,mode);
    TORCH_CHECK(direct.object_logits[0].item<float>()==-10. && (direct.high_res_mask[0]==-10.).all().item<bool>(),"empty direct-mask behavior changed");
    std::cout<<"direct_mask="<<direct.low_res_mask.sizes()<<" pointer="<<direct.object_pointer.sizes()<<'\n';return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
