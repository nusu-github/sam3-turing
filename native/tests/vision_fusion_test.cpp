#include "sam3/vision_fusion.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_TEST_CUDA_STREAM
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
namespace {
at::Tensor reference(const at::Tensor &x, const at::Tensor &g,
                     const at::Tensor &b, at::ScalarType t, bool windows) {
  auto y = at::layer_norm(x, {1024}, g, b, 1e-5);
  if (windows)
    y = y.view({x.size(0), x.size(1) / 24, 24, x.size(2) / 24, 24, 1024})
            .permute({0, 1, 3, 2, 4, 5})
            .reshape({-1, 24, 24, 1024});
  return y.to(t);
}
void compare(const at::Tensor &a, const at::Tensor &b) {
  TORCH_CHECK(a.sizes() == b.sizes() && a.strides() == b.strides() &&
                  a.scalar_type() == b.scalar_type(),
              "metadata mismatch: ", a.sizes(), " / ", b.sizes(), " strides ",
              a.strides(), " / ", b.strides());
  auto aa = a.cpu().contiguous(), bb = b.cpu().contiguous();
  if (!at::equal(aa.view(at::kByte), bb.view(at::kByte))) {
    std::cerr << "max error " << (aa - bb).abs().max().item<double>()
              << " unequal " << aa.ne(bb).sum().item<int64_t>() << '\n';
    TORCH_CHECK(false, "bitwise mismatch");
  }
}
void run(at::Device device) {
  c10::InferenceMode inference;
  at::set_num_threads(4);
  at::manual_seed(128);
  auto options = at::TensorOptions().device(device).dtype(at::kFloat);
  int count = 0, residual_count = 0;
  for (auto shape : std::vector<std::vector<int64_t>>{
           {1, 1, 1, 1024}, {1, 24, 48, 1024}, {2, 72, 72, 1024}}) {
    for (int layout = 0; layout < 4; ++layout) {
      auto x = at::randn(shape, options);
      auto g = at::randn({1024}, options), b = at::randn({1024}, options);
      if (layout == 1) {
        auto z = at::empty({x.numel() + 1}, options);
        x = z.slice(0, 1).view(shape);
        x.normal_();
      }
      if (layout == 2) {
        x = x.transpose(1, 2);
        g = at::randn({2048}, options).slice(0, 0, 2048, 2);
      }
      if (layout == 3) {
        x = x.slice(0, 0, 1).expand({3, x.size(1), x.size(2), 1024});
      }
      for (int distribution = 0; distribution < 4; ++distribution) {
        auto input = distribution == 0   ? x
                     : distribution == 1 ? x * 1e-7f + 100.f
                     : distribution == 2 ? x * 10000.f
                                         : at::zeros_like(x);
        for (auto type : {at::kFloat, at::kHalf, at::kBFloat16})
          for (bool windows : {false, true}) {
            if (windows && (input.size(1) % 24 || input.size(2) % 24))
              continue;
            auto expected = reference(input, g, b, type, windows);
            auto actual =
                sam3::vision_norm_projection(input, g, b, type, windows);
            compare(actual, expected);
            ++count;
            if (distribution == 0 || distribution == 2)
              for (auto attention_type :
                   {at::kFloat, at::kHalf, at::kBFloat16}) {
                const auto bs = input.size(0), hh = input.size(1),
                           ww = input.size(2);
                auto a =
                    at::randn(windows ? std::vector<int64_t>{bs * (hh / 24) *
                                                                 (ww / 24),
                                                             24, 24, 1024}
                                      : input.sizes().vec(),
                              options)
                        .to(attention_type);
                auto unfolded = a;
                if (windows)
                  unfolded = a.reshape({bs, hh / 24, ww / 24, 24, 24, 1024})
                                 .permute({0, 1, 3, 2, 4, 5})
                                 .reshape({bs, hh, ww, 1024});
                auto sum = input + unfolded;
                auto normalized =
                    at::layer_norm(sum, {1024}, g, b, 1e-5).to(type);
                auto fused =
                    sam3::vision_residual_norm(input, a, g, b, type, windows);
                compare(std::get<0>(fused), sum);
                compare(std::get<1>(fused), normalized);
                ++residual_count;
                auto prepared = sam3::vision_residual_norm_projection(
                    input, unfolded, g, b, type, windows);
                compare(std::get<0>(prepared), sum);
                compare(std::get<1>(prepared),
                        reference(sum, g, b, type, windows));
              }
          }
      }
    }
  }
  auto g = at::ones({1024}, options), b = at::zeros({1024}, options);
  auto empty = at::empty({0, 24, 24, 1024}, options);
  compare(sam3::vision_norm_projection(empty, g, b, at::kHalf, false),
          reference(empty, g, b, at::kHalf, false));
  bool rejected = false;
  try {
    sam3::vision_norm_projection(at::zeros({1, 2, 3, 1024}, options), g, b,
                                 at::kHalf, true);
  } catch (const c10::Error &) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "invalid window accepted");
  std::cout << "PASS " << count << " norm and " << residual_count
            << " each residual-norm and next-projection layout/distribution/dtype cases on " << device
            << '\n';
}
} // namespace
int main(int argc, char **argv) {
  try {
#ifdef SAM3_TEST_CUDA_STREAM
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromPool(false, 0));
    run(at::Device(at::kCUDA, 0));
#else
    run(at::Device(argc > 1 ? argv[1] : "cpu"));
#endif
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
