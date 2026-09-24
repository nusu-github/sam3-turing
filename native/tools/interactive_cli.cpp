#include "sam3/interactive_decoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==7,"usage: sam3_interactive STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference BATCH POINTS");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);
    const auto batch=std::stoll(argv[5]),count=std::stoll(argv[6]);
    TORCH_CHECK(batch>0 && count>=0,"invalid prompt dimensions");
    const auto options=at::TensorOptions().dtype(at::kFloat).device(device);
    const sam3::InteractivePromptEncoder encoder(store,model,device);
    const sam3::InteractiveMaskDecoder decoder(store,model,device);
    const auto points=at::rand({batch,count,2},options)*1008;
    const auto labels=at::arange(batch*count,options.dtype(at::kLong)).reshape({batch,count}).remainder(5)-1;
    const auto prompts=encoder.forward({points,labels,{},at::randn({batch,1,288,288},options)},mode);
    const auto high=decoder.project_pyramid({at::randn({1,256,288,288},options),at::randn({1,256,144,144},options)},mode);
    const auto image=at::randn({1,256,72,72},options);
    for (bool multi : {false,true}) {
      const auto out=decoder.forward(image,prompts,high,multi,true,mode);
      TORCH_CHECK(out.masks.size(1)==(multi?3:1) && out.all_masks.size(1)==4,"mask candidate count mismatch");
      TORCH_CHECK(at::isfinite(out.masks).all().item<bool>() && at::isfinite(out.iou).all().item<bool>(),"non-finite interactive output");
      std::cout<<"multimask="<<multi<<" masks="<<out.masks.sizes()<<" all_masks="<<out.all_masks.sizes()<<" iou="<<out.iou.sizes()<<" tokens="<<out.tokens.sizes()<<'\n';
    }
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
