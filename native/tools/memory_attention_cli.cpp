#include "sam3/memory_attention.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==8 || argc==9,"usage: sam3_memory_attention STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference BATCH FRAMES POINTERS [math]");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const bool math=argc==9;TORCH_CHECK(!math || std::string(argv[8])=="math","unknown attention backend option");
    if (math) {
      at::globalContext().setSDPUseFlash(false);at::globalContext().setSDPUseMemEfficient(false);
      at::globalContext().setSDPUseCuDNN(false);at::globalContext().setSDPUseMath(true);
    }
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);
    const auto batch=std::stoll(argv[5]),frames=std::stoll(argv[6]),pointers=std::stoll(argv[7]);
    TORCH_CHECK(batch>0 && frames>=0 && pointers>=0 && frames+pointers>0,"invalid attention dimensions");
    const bool multiplex=model=="sam3.1";const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    const auto source=at::randn({5184,batch,256},options),source_pos=at::randn_like(source);
    const auto memory=at::randn({frames*5184+pointers,batch,multiplex?256:64},options),memory_pos=at::randn_like(memory);
    at::Tensor image,memory_image,memory_image_pos;
    if (multiplex) {
      image=at::randn({5184,1,256},options);memory_image=at::randn({frames*5184,1,256},options);memory_image_pos=at::randn_like(memory_image);
    }
    const sam3::MemoryAttention attention(store,model,device);
    const auto out=attention.forward(source,source_pos,memory,memory_pos,pointers,mode,image,memory_image,memory_image_pos);
    TORCH_CHECK(out.sizes()==source.sizes() && at::isfinite(out).all().item<bool>(),"invalid memory attention output");
    std::cout<<"grid=72 batch="<<batch<<" frames="<<frames<<" pointer_tokens="<<pointers<<" math_only="<<math<<" output="<<out.sizes()<<'\n';return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
