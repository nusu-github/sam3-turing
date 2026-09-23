#include "sam3/autocast.h"
#include "sam3/image_results.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_TEST_CUDA_STATS
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {
// Source processor semantics: select, resize the entire selection in one call,
// sigmoid, threshold. This oracle does not reproduce native chunking.
std::vector<sam3::ImageResult> reference(const sam3::DetectionOutput &in,
                                         const std::vector<int64_t> &hs,
                                         const std::vector<int64_t> &ws,
                                         double threshold, bool presence,
                                         const std::string &mode) {
  sam3::AutocastGuard guard(in.logits.device().type(), mode != "fp32",
                            mode == "fp16" ? at::kHalf : at::kBFloat16);
  auto scores = in.logits.sigmoid();
  if (presence)
    scores = scores * in.presence_logits.sigmoid().unsqueeze(1);
  scores = scores.squeeze(-1);
  std::vector<sam3::ImageResult> result;
  for (int64_t b = 0; b < in.logits.size(0); ++b) {
    auto keep = scores[b] > threshold;
    auto boxes = in.boxes[b].index({keep});
    auto center = boxes.slice(1, 0, 2), size = boxes.slice(1, 2, 4);
    auto xyxy = at::cat({center - size / 2, center + size / 2}, 1) *
                at::tensor({ws[b], hs[b], ws[b], hs[b]},
                           boxes.options().dtype(at::kLong));
    auto p = at::upsample_bilinear2d(in.masks[b].index({keep}).unsqueeze(1),
                                     {hs[b], ws[b]}, false)
                 .sigmoid();
    result.push_back({xyxy, scores[b].index({keep}), p, p > .5,
                      at::nonzero(keep).squeeze(1)});
  }
  return result;
}
void exact(const at::Tensor &a, const at::Tensor &b) {
  TORCH_CHECK(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type(),
              "result shape/dtype mismatch");
  // Compare finite values exactly; NaNs must occupy the same positions.
  auto same = a == b;
  if (a.is_floating_point())
    same = same | (at::isnan(a) & at::isnan(b));
  TORCH_CHECK(same.all().item<bool>(), "result values differ");
}
void sync(at::Device device) {
#ifdef SAM3_TEST_CUDA_STATS
  if (device.is_cuda())
    c10::cuda::device_synchronize();
#endif
}
sam3::DetectionOutput input(at::Device device, at::ScalarType type,
                            int64_t queries, int64_t height, int64_t width) {
  const auto options = at::TensorOptions().device(device).dtype(type);
  sam3::DetectionOutput in;
  in.logits = at::randn({2, queries, 1}, options);
  in.boxes = at::rand({2, queries, 4}, options.dtype(at::kFloat));
  // Noncontiguous input and ragged output sizes exercise indexing/strides.
  in.masks = at::randn({2, queries, width, height}, options).transpose(2, 3);
  in.presence_logits = at::zeros({2, 1}, options);
  if (queries > 3) {
    in.masks[0][0][0][0] = std::numeric_limits<float>::quiet_NaN();
    in.masks[0][1][0][0] = std::numeric_limits<float>::infinity();
    in.masks[0][2][0][0] = -std::numeric_limits<float>::infinity();
    in.masks[0][3].zero_();
  }
  if (queries > 8) {
    in.masks[0][4].fill_(1e-8);
    in.masks[0][5].fill_(-1e-8);
    in.masks[0][6].fill_(.0002);
    in.masks[0][7].fill_(-.0002);
    in.masks[0][8].fill_(1e4);
  }
  return in;
}
void test(at::Device device) {
  int cases = 0;
  for (auto type : {at::kFloat, at::kDouble, at::kHalf, at::kBFloat16}) {
    for (auto queries : {0, 1, 17, 201}) {
      const auto in = input(device, type, queries, 3, 5);
      for (const auto &mode : {"fp32", "fp16", "bf16_reference"}) {
        for (bool presence : {false, true}) {
          for (double threshold : {-1., .25, 2.}) {
            for (bool reduce : {false, true}) {
              const std::vector<int64_t> hs = reduce
                                                  ? std::vector<int64_t>{3, 1}
                                                  : std::vector<int64_t>{7, 13};
              const std::vector<int64_t> ws = reduce
                                                  ? std::vector<int64_t>{5, 2}
                                                  : std::vector<int64_t>{11, 9};
              const auto expected =
                  reference(in, hs, ws, threshold, presence, mode);
              for (auto chunk : {1, 3, 13, 512}) {
                const auto actual = sam3::postprocess_image(
                    in, hs, ws, threshold, presence, chunk, mode);
                TORCH_CHECK(actual.size() == expected.size(),
                            "batch size mismatch");
                for (size_t b = 0; b < actual.size(); ++b) {
                  exact(actual[b].boxes, expected[b].boxes);
                  exact(actual[b].scores, expected[b].scores);
                  exact(actual[b].mask_probabilities,
                        expected[b].mask_probabilities);
                  exact(actual[b].masks, expected[b].masks);
                  exact(actual[b].query_indices, expected[b].query_indices);
                }
                ++cases;
              }
            }
          }
        }
      }
    }
  }
  std::cout << "{\"device\":\"" << device << "\",\"cases\":" << cases
            << ",\"all_outputs_exact\":true}\n";
}
void benchmark(at::Device device) {
  TORCH_CHECK(device.is_cuda(), "benchmark requires CUDA timing support");
#ifndef SAM3_TEST_CUDA_STATS
  TORCH_CHECK(false, "compile with CUDA statistics support");
#else
  auto in = input(device, at::kHalf, 200, 288, 288);
  in.logits = in.logits.slice(0, 0, 1);
  in.boxes = in.boxes.slice(0, 0, 1);
  in.masks = at::randn({1, 200, 288, 288}, in.masks.options());
  in.presence_logits = in.presence_logits.slice(0, 0, 1);
  for (auto shape :
       {std::pair<int64_t, int64_t>{640, 540}, {1080, 1920}, {2160, 3840}}) {
    for (int64_t chunk : {1, 8, 32, 200}) {
      for (int i = 0; i < 3; ++i) {
        auto out = sam3::postprocess_image(in, {shape.first}, {shape.second},
                                           -1., true, chunk, "fp16");
        sync(device);
      }
      c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
      std::vector<double> seconds;
      for (int i = 0; i < 9; ++i) {
        sync(device);
        const auto start = std::chrono::steady_clock::now();
        auto out = sam3::postprocess_image(in, {shape.first}, {shape.second},
                                           -1., true, chunk, "fp16");
        sync(device);
        seconds.push_back(std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - start)
                              .count());
        TORCH_CHECK(out[0].scores.numel() == 200, "query truncation");
      }
      const auto stats =
          c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
      std::sort(seconds.begin(), seconds.end());
      std::cout << std::setprecision(10) << "{\"height\":" << shape.first
                << ",\"width\":" << shape.second << ",\"chunk\":" << chunk
                << ",\"queries\":200,\"repetitions\":9,\"median_seconds\":"
                << seconds[4] << ",\"min_seconds\":" << seconds[0]
                << ",\"max_seconds\":" << seconds[8]
                << ",\"peak_allocated_bytes\":" << stats.allocated_bytes[0].peak
                << "}" << std::endl;
    }
  }
#endif
}
} // namespace
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 2 ||
                    (argc == 3 && std::string(argv[2]) == "--benchmark"),
                "usage: image_results_test cpu|cuda [--benchmark]");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::manual_seed(392);
    const auto device =
        at::empty({0}, at::TensorOptions().device(at::Device(argv[1])))
            .device();
    c10::DeviceGuard guard(device);
    if (argc == 3)
      benchmark(device);
    else
      test(device);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
