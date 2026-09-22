#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2 || (std::string(argv[1]) != "cpu" && std::string(argv[1]) != "cuda")) {
    std::cerr << "usage: sam3_native_smoke cpu|cuda\n";
    return 2;
  }
  try {
    c10::InferenceMode guard;
    const auto device = at::Device(argv[1]);
    const auto options = at::TensorOptions().dtype(at::kFloat).device(device);
    const auto a = at::arange(6, options).reshape({2, 3});
    const auto actual = at::matmul(a, a.transpose(0, 1)).cpu();
    const auto expected = at::tensor({5.f, 14.f, 14.f, 50.f}).reshape({2, 2});
    if (!at::equal(actual, expected)) {
      std::cerr << "matrix product mismatch\n";
      return 1;
    }
    std::cout << "PASS: native ATen matmul on " << argv[1] << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
