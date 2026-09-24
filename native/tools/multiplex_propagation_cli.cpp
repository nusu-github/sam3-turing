#include "sam3/multiplex_decoder.h"
#include "sam3/memory_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==5 || argc==6,"usage: sam3_multiplex_propagation STORE cpu|cuda fp32|fp16|bf16_reference OBJECTS [math]");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    if (argc==6) {
      TORCH_CHECK(std::string(argv[5])=="math","unknown attention backend");
      at::globalContext().setSDPUseFlash(false);at::globalContext().setSDPUseMemEfficient(false);
      at::globalContext().setSDPUseCuDNN(false);at::globalContext().setSDPUseMath(true);
    }
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const at::Device device(argv[2]);
    const std::string mode=argv[3];const auto objects=std::stoll(argv[4]);
    TORCH_CHECK(objects>0,"object count must be positive");
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    auto state=sam3::MultiplexController().get_state(objects,device,at::kFloat,true);
    const sam3::MultiplexPropagationHeads heads(store,device);
    const sam3::MaskMemoryEncoder memory(store,"sam3.1",device);
    const auto shared_image=at::randn({1,256,72,72},options);
    const auto high=heads.project_pyramid({at::randn({1,256,288,288},options),at::randn({1,256,144,144},options)},mode);
    const auto run=[&](const char* phase) {
      // Synthetic conditioned features; temporal memory selection is separate.
      const auto conditioned=at::randn({state.bucket_count(),256,72,72},options);
      const auto out=heads.forward(state,conditioned,high,mode);
      TORCH_CHECK(out.low_res_multimasks.sizes()==at::IntArrayRef({state.object_count(),3,288,288}),"invalid propagation candidate shape");
      TORCH_CHECK(out.high_res_mask.sizes()==at::IntArrayRef({state.object_count(),1,1008,1008}),"invalid selected propagation shape");
      TORCH_CHECK(out.object_pointer.sizes()==at::IntArrayRef({state.object_count(),256}) && at::isfinite(out.object_pointer).all().item<bool>(),"invalid propagation pointers");
      const auto encoded=memory.encode_frame(shared_image,out.high_res_mask,out.object_logits,{},state.mux_matrix(),{},mode);
      TORCH_CHECK(encoded.features.sizes()==at::IntArrayRef({state.bucket_count(),256,72,72}) && at::isfinite(encoded.features).all().item<bool>(),"invalid encoded propagation memory");
      std::cout<<"phase="<<phase<<" objects="<<state.object_count()<<" buckets="<<state.bucket_count()
               <<" candidates="<<out.low_res_multimasks.sizes()<<" selected="<<out.high_res_mask.sizes()
               <<" memory="<<encoded.features.sizes()<<'\n';
    };
    run("initial");
    if (objects>1) {
      state.remove_objects({0});state.add_objects(state.next_indices(1,true,true),std::nullopt,true,true);
      run("remove_and_add");
    }
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
