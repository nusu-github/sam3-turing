#include "sam3/geometry_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc == 5,"usage: sam3_geometry STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference");
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[3]);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    const sam3::GeometryEncoder encoder(store,argv[2],device);
    const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
    const auto features = at::arange(2*256*72*72,options).remainder(71).reshape({2,256,72,72})/71.;
    const auto positions = at::zeros_like(features);
    for (int64_t n : {0,3}) {
      sam3::GeometryPrompt prompt {
        at::full({n,2,2},.3,options), at::zeros({n,2},options.dtype(at::kLong)), at::zeros({2,n},options.dtype(at::kBool)),
        at::full({n,2,4},.4,options), at::ones({n,2},options.dtype(at::kBool)), at::zeros({2,n},options.dtype(at::kBool))};
      if (n) { prompt.point_padding[1][n-1] = true; prompt.box_padding[1][n-1] = true; }
      const auto [encoded,padding] = encoder.forward(features,positions,prompt,argv[4]);
      TORCH_CHECK(encoded.sizes() == at::IntArrayRef({2*n+1,2,256}) && at::isfinite(encoded).all().item<bool>(),"invalid geometry output");
      TORCH_CHECK(padding.sum().item<int64_t>() == (n ? 2 : 0),"lost per-image padding");
      std::cout << "geometry=" << encoded.sizes() << " sum=" << encoded.sum().item<double>()
                << " padded=" << padding.sum().item<int64_t>() << '\n';
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
