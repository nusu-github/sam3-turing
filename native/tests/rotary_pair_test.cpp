#include "sam3/rotary_pair.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_TEST_CUDA_STREAM
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
#include <limits>
at::Tensor ref(const at::Tensor &x, const at::Tensor &f) {
  return at::view_as_real(
             at::view_as_complex(x.to(at::kFloat)
                                     .reshape({x.size(0), x.size(1), x.size(2),
                                               x.size(3) / 2, 2})) *
             f.view({1, 1, x.size(2), x.size(3) / 2}))
      .flatten(3)
      .to(x.scalar_type());
}
void check_equal(const at::Tensor &a, const at::Tensor &b) {
  TORCH_CHECK(a.sizes() == b.sizes() && a.strides() == b.strides() &&
                  a.scalar_type() == b.scalar_type(),
              "pair layout");
  TORCH_CHECK(
      at::equal(a.contiguous().view(at::kByte), b.contiguous().view(at::kByte)),
      "pair bits; max=", (a - b).abs().max().item<double>(),
      " different=", a.ne(b).sum().item<int64_t>());
}
int main(int argc, char **argv) {
  try {
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::manual_seed(2951);
    const at::Device device(argc > 1 ? argv[1] : "cpu");
#ifdef SAM3_TEST_CUDA_STREAM
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromPool());
#endif
    int count = 0;
    for (auto type : {at::kFloat, at::kHalf, at::kBFloat16, at::kDouble})
      for (auto shape : std::vector<std::pair<int64_t, int64_t>>{
               {1, 1}, {1, 7}, {9, 576}, {1, 5184}, {2, 5184}}) {
        auto options = at::TensorOptions().device(device).dtype(type);
        auto data = at::randn({shape.first, shape.second, 3, 16, 64}, options);
        auto q = data.select(2, 0).permute({0, 2, 1, 3}),
             k = data.select(2, 1).permute({0, 2, 1, 3});
        auto saved = data.clone();
        auto angles = at::randn({shape.second, 32}, options.dtype(at::kFloat));
        auto f = at::polar(at::ones_like(angles), angles);
        for (int variant = 0; variant < 4; ++variant) {
          auto a = q, b = k, frequency = f;
          if (variant == 1) {
            a = q.contiguous();
            b = k.contiguous();
          }
          if (variant == 2)
            frequency = f.conj();
          if (variant == 3) {
            a = q.slice(2, 0, 0);
            b = k.slice(2, 0, 0);
            frequency = f.slice(0, 0, 0);
          }
          auto result = sam3::rotary_embedding_pair(a, b, frequency);
          check_equal(std::get<0>(result), ref(a, frequency));
          check_equal(std::get<1>(result), ref(b, frequency));
          auto retained = std::get<1>(result).clone();
          std::get<0>(result).fill_(0);
          TORCH_CHECK(at::equal(std::get<1>(result), retained),
                      "outputs alias");
          TORCH_CHECK(at::equal(data, saved), "input modified");
          ++count;
        }
      }
    auto options = at::TensorOptions().device(device).dtype(at::kFloat);
    auto data = at::zeros({1, 1, 3, 16, 64}, options);
    auto q = data.select(2, 0).permute({0, 2, 1, 3}),
         k = data.select(2, 1).permute({0, 2, 1, 3});
    q.fill_(-1);
    k.fill_(-1);
    q.slice(3, 0, 64, 2).fill_(0x1.000002p0f);
    k.copy_(q);
    auto f = at::view_as_complex(
        at::stack({at::ones({1, 32}, options),
                   at::full({1, 32}, 0x1.fffffep-1f, options)},
                  2));
    auto result = sam3::rotary_embedding_pair(q, k, f);
    check_equal(std::get<0>(result), ref(q, f));
    check_equal(std::get<1>(result), ref(k, f));
    ++count;
    const float inf = std::numeric_limits<float>::infinity();
    q.flatten().slice(0, 0, 8).copy_(
        at::tensor({0.f, -0.f, inf, -inf,
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::denorm_min(), 1e-30f, 1e30f})
            .to(device));
    k.copy_(q);
    result = sam3::rotary_embedding_pair(q, k, f);
    check_equal(std::get<0>(result), ref(q, f));
    check_equal(std::get<1>(result), ref(k, f));
    ++count;
    bool rejected = false;
    try {
      sam3::rotary_embedding_pair(q, k, at::real(f));
    } catch (const c10::Error &) {
      rejected = true;
    }
    TORCH_CHECK(rejected, "invalid frequency accepted");
    std::cout << "PASS rotary pair " << count
              << " exact cases, layout, independence, FMA, nonfinite, empty, "
                 "fallback and invalid inputs\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
