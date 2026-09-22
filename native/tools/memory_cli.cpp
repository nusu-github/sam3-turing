#include "sam3/memory_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==6,"usage: sam3_memory STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference OBJECTS");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);
    const auto objects=std::stoll(argv[5]);TORCH_CHECK(objects>0,"object count must be positive");
    const bool multiplex=model=="sam3.1";const auto batch=multiplex?(objects+15)/16:objects;
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    const sam3::MaskMemoryEncoder encoder(store,model,device);
    const auto image=at::randn({1,256,72,72},options),masks=at::randn({objects,1,1008,1008},options);
    const auto scores=at::randn({objects,1},options);
    at::Tensor matrix,conditions;
    if (multiplex) {
      matrix=at::zeros({batch*16,objects},options);
      matrix.diagonal().fill_(1.);
      conditions=at::arange(0,objects,2,options.dtype(at::kLong));
    }
    for (bool from_points:{false,true}) {
      const auto out=encoder.encode_frame(image,masks,scores,{from_points,true,0.},matrix,conditions,mode);
      TORCH_CHECK(out.features.sizes()==at::IntArrayRef({batch,multiplex?256:64,72,72}) && out.position.sizes()==out.features.sizes(),"invalid memory output shape");
      TORCH_CHECK(at::isfinite(out.features).all().item<bool>() && at::isfinite(out.position).all().item<bool>(),"non-finite memory output");
      std::cout<<"objects="<<objects<<" groups="<<batch<<" from_points="<<from_points<<" features="<<out.features.sizes()<<" position="<<out.position.sizes()<<'\n';
    }
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
