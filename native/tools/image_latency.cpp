// Repeated complete image inference, with resident modules and cached text.
#include "ppm.h"
#include "sam3/grounding.h"
#include "sam3/image_results.h"
#include "sam3/ops.h"
#include "sam3/preprocess.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include "sam3/vision_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void sync() { c10::cuda::device_synchronize(); }
void binary(const std::filesystem::path& path, const at::Tensor& tensor) {
  auto value = tensor.cpu().contiguous();
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(value.const_data_ptr()), value.nbytes());
  TORCH_CHECK(out, "cannot write ", path.u8string());
}
}

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 8 || (argc == 9 && std::string(argv[8]) == "--profile"), "usage: sam3_image_latency STORE IMAGE.ppm BPE.gz PROMPT.txt WARMUPS REPEATS OUTPUT [--profile]");
    const bool profile = argc == 9;
    const bool capture = std::getenv("SAM3_PROFILE_CAPTURE") != nullptr;
    const bool cudnn_benchmark = std::getenv("SAM3_BENCH_CUDNN") != nullptr;
    const int warmups = std::stoi(argv[5]), repeats = std::stoi(argv[6]);
    TORCH_CHECK(warmups >= 0 && repeats > 0, "invalid repetition count");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setBenchmarkCuDNN(cudnn_benchmark);
    std::cout << "cuDNN available=" << at::Context::hasCuDNN()
              << " enabled=" << at::globalContext().userEnabledCuDNN()
              << " version=" << at::Context::versionRuntimeCuDNN()
              << " benchmark=" << cudnn_benchmark << std::endl;
    const at::Device device(at::kCUDA, 0);
    c10::cuda::set_device(0);
    const auto root = std::filesystem::u8path(argv[7]);
    std::filesystem::create_directories(root);
    const auto rgb = sam3::cli::read_ppm(std::filesystem::u8path(argv[2]));
    std::ifstream prompt_file(std::filesystem::u8path(argv[4]), std::ios::binary);
    TORCH_CHECK(prompt_file, "cannot read prompt");
    const std::string prompt_text((std::istreambuf_iterator<char>(prompt_file)), {});
    const auto setup_start = Clock::now();
    sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    sam3::Tokenizer tokenizer(std::filesystem::u8path(argv[3]));
    std::cout << "loading modules" << std::endl;
    sam3::VisionEncoder vision(store, "sam3", device, at::kHalf);
    sam3::TextEncoder text(store, "sam3", device, at::kHalf);
    sam3::GroundingDetector detector(store, "sam3", device);
    auto tokens = at::tensor(tokenizer.tokenize({prompt_text})[0], at::kLong).unsqueeze(0).to(device);
    auto encoded = text.forward(tokens, "fp16");
    const auto opt = at::TensorOptions().device(device);
    const auto ids = at::zeros({1}, opt.dtype(at::kLong));
    const auto labels = at::empty({0,1}, opt.dtype(at::kLong));
    const auto padding = at::empty({1,0}, opt.dtype(at::kBool));
    const sam3::GroundingPrompt prompt{ids, ids, std::get<1>(encoded), std::get<0>(encoded),
      {at::empty({0,1,2}, opt), labels, padding, at::empty({0,1,4}, opt), labels, padding}};
    sync();
    const double setup = seconds(setup_start);
    const auto baseline = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    std::array<at::cuda::CUDAEvent, 6> events;
    if (profile) for (auto& event : events) event = at::cuda::CUDAEvent(cudaEventDefault);
    std::array<double, 6> host_marks{};
    auto infer = [&]() {
      const auto begin = Clock::now();
      auto mark = [&](size_t i) {
        if (profile) { events[i].record(); host_marks[i] = seconds(begin); }
      };
      mark(0);
      auto uploaded = rgb.to(device);
      mark(1);
      auto input = sam3::preprocess_rgb(uploaded);
      mark(2);
      auto features = vision.forward(input, "fp16", {"convs"}, {2});
      input = at::Tensor();
      uploaded = at::Tensor();
      mark(3);
      auto pyramid = std::move(features.pyramid.at("convs"));
      pyramid.pop_back();
      auto raw = detector.forward(pyramid, features.positions[pyramid.size()-1], prompt, false, "fp16");
      TORCH_CHECK(raw.detection.logits.size(1) == 200, "query count changed");
      mark(4);
      auto output = sam3::postprocess_image(raw.detection, {rgb.size(1)}, {rgb.size(2)}, .5, true, 8, "fp16")[0];
      mark(5);
      return output;
    };
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    std::cout << "cold inference" << std::endl;
    sync();
    const auto cold_start = Clock::now();
    sam3::ImageResult result = infer();
    sync();
    const double cold = seconds(cold_start);
    const auto cold_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    result = {};
    for (int i=0; i<warmups; ++i) { auto warm = infer(); sync(); }
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    std::vector<double> samples;
    std::vector<std::array<double, 5>> gpu_stages, host_stages;
    std::vector<int64_t> counts;
    if (capture) C10_CUDA_CHECK(cudaProfilerStart());
    for (int i=0; i<repeats; ++i) {
      result = {};
      sync();
      const auto start = Clock::now();
      result = infer();
      sync();
      samples.push_back(seconds(start));
      if (profile) {
        std::array<double, 5> gpu{}, host{};
        for (size_t j = 0; j < 5; ++j) {
          gpu[j] = events[j].elapsed_time(events[j+1]);
          host[j] = 1000 * (host_marks[j+1] - host_marks[j]);
        }
        gpu_stages.push_back(gpu);
        host_stages.push_back(host);
      }
      counts.push_back(result.scores.numel());
      std::cout << "sample " << i << " seconds=" << samples.back() << " detections=" << counts.back() << std::endl;
    }
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    if (capture) C10_CUDA_CHECK(cudaProfilerStop());
    for (const auto& tensor : {result.scores, result.boxes, result.mask_probabilities})
      TORCH_CHECK(at::isfinite(tensor).all().item<bool>(), "nonfinite result");
    TORCH_CHECK(std::all_of(counts.begin(), counts.end(), [&](int64_t n){return n==counts.front();}), "unstable detection count");
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = (sorted[(sorted.size()-1)/2] + sorted[sorted.size()/2]) / 2;
    binary(root/"masks.u8.bin", result.masks.to(at::kByte));
    binary(root/"scores.f32.bin", result.scores.to(at::kFloat));
    binary(root/"boxes.f32.bin", result.boxes.to(at::kFloat));
    binary(root/"queries.i64.bin", result.query_indices);
    std::ofstream out(root/"metrics.json");
    out << std::setprecision(12) << "{\n\"model\":\"sam3\",\"mode\":\"fp16\",\"resolution\":1008,\"threshold\":0.5,"
        << "\"scope\":\"CPU RGB tensor transfer + preprocessing + vision + detector + postprocessing; cached text; no image feature reuse; excludes disk IO and setup\",\n"
        << "\"cudnn_available\":" << (at::Context::hasCuDNN()?"true":"false")
        << ",\"cudnn_runtime_version\":" << at::Context::versionRuntimeCuDNN()
        << ",\"cudnn_benchmark\":" << (cudnn_benchmark?"true":"false")
        << ",\"compute_storage\":\"fp16\",\"text_weights_resident\":true,\"height\":" << rgb.size(1) << ",\"width\":" << rgb.size(2)
        << ",\"count\":" << result.scores.numel() << ",\"finite\":true,\"warmups\":" << warmups << ",\"repeats\":" << repeats
        << ",\"setup_seconds\":" << setup << ",\"cold_seconds\":" << cold << ",\"median_wall_seconds\":" << median
        << ",\"allocated_before_bytes\":" << baseline.allocated_bytes[0].current
        << ",\"cold_peak_allocated_bytes\":" << cold_stats.allocated_bytes[0].peak
        << ",\"peak_allocated_bytes\":" << stats.allocated_bytes[0].peak
        << ",\"peak_reserved_bytes\":" << stats.reserved_bytes[0].peak << ",\"wall_seconds\":[";
    for (size_t i=0; i<samples.size(); ++i) { if(i)out<<',';out<<samples[i]; }
    out << "]}\n";
    TORCH_CHECK(out, "cannot write metrics");
    if (profile) {
      std::ofstream stages(root/"stages.json");
      stages << std::setprecision(12) << "{\"stages\":[\"upload\",\"preprocess\",\"vision\",\"detector\",\"postprocess\"],"
             << "\"method\":\"CUDA events on current stream; no added inter-stage synchronization. Host durations include enqueue and any internal blocking, not CPU compute alone. GPU intervals may include host launch gaps.\",";
      auto rows = [&](const char* name, const auto& values) {
        stages << '\"' << name << "\":[";
        for (size_t i=0; i<values.size(); ++i) {
          if(i) stages << ',';
          stages << '[';
          for(size_t j=0; j<5; ++j) { if(j) stages << ','; stages << values[i][j]; }
          stages << ']';
        }
        stages << ']';
      };
      rows("gpu_ms", gpu_stages); stages << ','; rows("host_ms", host_stages);
      stages << "}\n";
      TORCH_CHECK(stages, "cannot write stages");
    }
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
