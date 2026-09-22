#include "sam3/text_encoder.h"
#include <ATen/Context.h>
#include <iostream>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc >= 5, "usage: sam3_text STORE sam3|sam3.1 cpu|cuda TOKEN_ID...");
    std::vector<int64_t> ids;
    for (int i = 4; i < argc; ++i) ids.push_back(std::stoll(argv[i]));
    at::globalContext().setAllowTF32CuBLAS(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    const sam3::TextEncoder encoder(store, argv[2], at::Device(argv[3]));
    const auto result = encoder.forward(at::tensor(ids, at::kLong).unsqueeze(0));
    const auto memory = std::get<1>(result);
    std::cout << "memory=" << memory.sizes() << " sum=" << memory.sum().item<double>() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
