#include "sam3/vision_position.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <iostream>
#ifdef SAM3_TEST_CUDA_STREAM
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
at::Tensor reference(const at::Tensor &x) {
  const auto b = x.size(0), h = x.size(2), w = x.size(3);
  const auto options = x.options().dtype(at::kFloat);
  auto y = at::arange(1, h + 1, options).view({1, h, 1}).repeat({b, 1, w});
  auto xx = at::arange(1, w + 1, options).view({1, 1, w}).repeat({b, h, 1});
  y = y / (y.slice(1, h - 1, h) + 1e-6) * (2 * std::acos(-1.0));
  xx = xx / (xx.slice(2, w - 1, w) + 1e-6) * (2 * std::acos(-1.0));
  auto dim = at::arange(128, options);
  dim = at::pow(10000.0, 2 * dim.floor_divide(2) / 128);
  const auto encode = [&](const at::Tensor &c) {
    auto angle = c.unsqueeze(-1) / dim;
    return at::stack({angle.slice(3, 0, 128, 2).sin(),
                      angle.slice(3, 1, 128, 2).cos()},
                     4)
        .flatten(3);
  };
  return at::cat({encode(y), encode(xx)}, 3)
      .permute({0, 3, 1, 2})
      .to(x.scalar_type());
}
int main(int argc, char **argv) {
  try {
    c10::InferenceMode inference;
    at::set_num_threads(4);
    const at::Device device(argc > 1 ? argv[1] : "cpu");
#ifdef SAM3_TEST_CUDA_STREAM
    c10::cuda::CUDAGuard device_guard(device);
    c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
#endif
    int count = 0;
    const auto check = [&](const at::Tensor &x) {
      auto a = reference(x), b = sam3::vision_position_encoding(x);
      TORCH_CHECK(a.sizes() == b.sizes() && a.strides() == b.strides() &&
                      a.scalar_type() == b.scalar_type() &&
                      a.device() == b.device(),
                  "position metadata");
      TORCH_CHECK(at::equal(a.contiguous().reshape({-1}).view(at::kByte),
                            b.contiguous().reshape({-1}).view(at::kByte)),
                  "position bits");
      ++count;
    };
    for (auto type : {at::kFloat, at::kHalf, at::kBFloat16, at::kDouble}) {
      const auto options = at::TensorOptions().device(device).dtype(type);
      for (auto shape : std::vector<std::vector<int64_t>>{{1, 1, 1},
                                                          {1, 7, 13},
                                                          {2, 24, 48},
                                                          {1, 36, 36},
                                                          {2, 72, 72},
                                                          {1, 144, 144},
                                                          {1, 288, 288},
                                                          {2, 288, 288},
                                                          {0, 7, 13},
                                                          {2, 0, 13},
                                                          {2, 7, 0}})
        check(at::empty({shape[0], 3, shape[1], shape[2]}, options));
      check(at::zeros({2, 3, 7, 13}, options).transpose(2, 3));
      check(at::empty({1, 0, 7, 13}, options));
    }
    for (auto type : {at::kFloat8_e4m3fn, at::kFloat8_e5m2})
      check(at::empty({1, 3, 7, 13},
                      at::TensorOptions().device(device).dtype(type)));
    auto x = at::ones({2, 3, 7, 13}, at::TensorOptions().device(device));
    auto a = sam3::vision_position_encoding(x);
    x.fill_(17);
    auto b = sam3::vision_position_encoding(x);
    TORCH_CHECK(at::equal(a, b), "positions depend on image values");
    a.zero_();
    TORCH_CHECK(at::equal(b, reference(x)), "outputs alias");
    TORCH_CHECK(at::equal(x, at::full_like(x, 17)), "input changed");
    for (auto invalid :
         {at::empty({1, 3, 7}, x.options()),
          at::empty({1, 3, 7, 13}, x.options().dtype(at::kLong))}) {
      bool rejected = false;
      try {
        sam3::vision_position_encoding(invalid);
      } catch (const c10::Error &) {
        rejected = true;
      }
      TORCH_CHECK(rejected, "invalid input accepted");
    }
    std::cout << "PASS position " << count
              << " exact cases, layout, empty/strided inputs, independent "
                 "outputs and invalid inputs\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
