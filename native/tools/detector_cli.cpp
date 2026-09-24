#include "sam3/detector.h"
#include "sam3/geometry_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==5,"usage: sam3_detector_transformer STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference");
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[3]);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    const sam3::GeometryEncoder geometry(store,argv[2],device);
    const sam3::DetectorEncoder encoder(store,argv[2],device);
    const sam3::DetectorDecoder decoder(store,argv[2],device);
    const auto options=at::TensorOptions().dtype(at::kFloat).device(device);
    const auto image=at::arange(256*72*72,options).remainder(71).reshape({1,256,72,72})/71.;
    const auto positions=at::zeros_like(image);
    sam3::GeometryPrompt prompt {
      at::empty({0,1,2},options),at::empty({0,1},options.dtype(at::kLong)),at::empty({1,0},options.dtype(at::kBool)),
      at::full({2,1,4},.4,options),at::tensor({1,0},options.dtype(at::kLong)).view({2,1}),at::zeros({1,2},options.dtype(at::kBool))};
    const auto [embedding,padding]=geometry.forward(image,positions,prompt,argv[4]);
    const auto fused=encoder.forward(image,positions,embedding,padding,{},argv[4]);
    const auto decoded=decoder.forward(fused,padding,argv[4]);
    TORCH_CHECK(decoded.hidden.sizes()==at::IntArrayRef({6,200,1,256}),"lost decoder queries/layers");
    TORCH_CHECK(at::isfinite(decoded.hidden).all().item<bool>() && at::isfinite(decoded.references).all().item<bool>(),"nonfinite transformer outputs");
    std::cout << "memory=" << fused.memory.sizes() << " hidden=" << decoded.hidden.sizes()
              << " references=" << decoded.references.sizes() << " presence=" << decoded.presence_logits.sizes()
              << " hidden_sum=" << decoded.hidden.sum().item<double>() << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
