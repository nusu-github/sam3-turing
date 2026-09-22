#include "sam3/vision_encoder.h"
#include "sam3/preprocess.h"
#include <ATen/Context.h>
#include <iostream>
int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 5, "usage: sam3_vision STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference");
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[3]);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    const sam3::VisionEncoder encoder(store, argv[2], device);
    const auto pixels = at::arange(3*480*640,at::TensorOptions().dtype(at::kLong)).remainder(256).to(at::kByte).reshape({3,480,640}).to(device);
    const auto result = encoder.forward(sam3::preprocess_rgb(pixels),argv[4]);
    std::cout << "trunk=" << result.trunk.sizes() << " sum=" << result.trunk.sum().item<double>() << '\n';
    for (const auto& [head,features] : result.pyramid)
      for (size_t i = 0; i < features.size(); ++i)
        std::cout << head << '.' << i << '=' << features[i].sizes() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
