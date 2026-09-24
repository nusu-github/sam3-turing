// Synchronized, whole-vision benchmark; all available heads and positions.
#include "ppm.h"
#include "sam3/preprocess.h"
#include "sam3/vision_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(
        argc == 8,
        "STORE MODEL MODE FRAME.ppm BATCH OUTPUT.json SAMPLES (CUDA device 0)");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    const at::Device device(at::kCUDA, 0);
    c10::DeviceGuard guard(device);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model = argv[2], mode = argv[3];
    const auto batch = std::stoll(argv[5]);
    const int samples = std::stoi(argv[7]);
    TORCH_CHECK((model == "sam3" || model == "sam3.1") && batch > 0 &&
                    samples > 0,
                "model/batch/samples");
    const auto sync = []() { c10::cuda::device_synchronize(); };
    const auto stats = []() {
      return c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    };
    at::empty({1}, at::TensorOptions().device(device));
    sync();
    const auto start = std::chrono::steady_clock::now();
    sam3::VisionEncoder encoder(sam3::WeightStore(argv[1]), model, device,
                                mode == "fp16" ? at::kHalf : at::kFloat);
    sync();
    const auto load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    const auto parameter_bytes = stats().allocated_bytes[0].current;
    const auto input =
        sam3::preprocess_rgb(sam3::cli::read_ppm(argv[4]).to(device))
            .repeat({batch, 1, 1, 1});
    for (int i = 0; i < 3; ++i) {
      auto out = encoder.forward(input, mode);
      sync();
    }
    std::ofstream report(argv[6]);
    TORCH_CHECK(report, "open output");
    report << std::setprecision(12) << "{\"model\":\"" << model
           << "\",\"mode\":\"" << mode << "\",\"batch\":" << batch
           << ",\"all_heads_and_positions\":true,\"parameter_bytes\":"
           << parameter_bytes << ",\"load_ms\":" << load_ms << ",\"samples\":[";
    for (int i = 0; i < samples; ++i) {
      sync();
      const auto before = stats().allocated_bytes[0].current;
      c10::cuda::CUDACachingAllocator::resetPeakStats(0);
      auto begin = std::chrono::steady_clock::now();
      auto out = encoder.forward(input, mode);
      sync();
      const auto ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
      const auto peak = stats().allocated_bytes[0].peak;
      if (i)
        report << ',';
      report << "{\"ms\":" << ms << ",\"peak_extra_bytes\":" << peak - before
             << ",\"peak_allocated_bytes\":" << peak << '}';
    }
    report << "]}\n";
    report.close();
    TORCH_CHECK(report, "write output");
    std::cout << "vision benchmark saved; runtime is native CUDA/LibTorch\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
