// Actual-weight equivalence and allocation/timing audit. No model output or
// inference resolution is reduced; only unconsumed position maps are omitted.
#include "ppm.h"
#include "sam3/preprocess.h"
#include "sam3/vision_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_BENCH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
namespace {
void sync(at::Device device) {
#ifdef SAM3_BENCH_CUDA
  if (device.is_cuda())
    c10::cuda::device_synchronize();
#endif
}
int64_t allocated(at::Device device, bool peak = false) {
#ifdef SAM3_BENCH_CUDA
  if (device.is_cuda()) {
    const auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    return peak ? stats.allocated_bytes[0].peak
                : stats.allocated_bytes[0].current;
  }
#endif
  return 0;
}
void reset(at::Device device) {
#ifdef SAM3_BENCH_CUDA
  if (device.is_cuda())
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
#endif
}
void assert_equal(const at::Tensor &a, const at::Tensor &b) {
  TORCH_CHECK(a.sizes() == b.sizes() && a.strides() == b.strides() &&
                  a.scalar_type() == b.scalar_type() &&
                  a.device() == b.device() && at::equal(a, b),
              "selected vision features differ");
}
struct Role {
  std::string name;
  std::vector<std::string> heads;
  std::vector<int64_t> positions;
};
} // namespace
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 9, "usage: vision_positions_benchmark STORE MODEL "
                           "DEVICE MODE FRAME.ppm BATCH REPEATS OUTPUT.json");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model = argv[2], mode = argv[4];
    TORCH_CHECK(model == "sam3" || model == "sam3.1", "invalid model");
    const auto device =
        at::empty({0}, at::TensorOptions().device(at::Device(argv[3])))
            .device();
#ifndef SAM3_BENCH_CUDA
    TORCH_CHECK(device.is_cpu(), "build with CUDA benchmarking support");
#endif
    const c10::DeviceGuard guard(device);
    const int64_t batch = std::stoll(argv[6]), repeats = std::stoll(argv[7]);
    TORCH_CHECK(batch > 0 && repeats > 0, "positive batch/repeats required");
    sam3::VisionEncoder encoder(
        sam3::WeightStore(std::filesystem::u8path(argv[1])), model, device);
    const auto rgb = sam3::cli::read_ppm(std::filesystem::u8path(argv[5]));
    const auto input =
        sam3::preprocess_rgb(rgb.to(device)).repeat({batch, 1, 1, 1});
    const std::string interactive =
        model == "sam3" ? "sam2_convs" : "interactive_convs";
    const std::vector<Role> roles = {
        {"video",
         model == "sam3" ? std::vector<std::string>{"convs", interactive}
                         : std::vector<std::string>{"convs", interactive,
                                                    "propagation_convs"},
         {2}},
        {"grounding", {"convs"}, {2}},
        {"interactive", {interactive}, {}}};
    std::ofstream out(std::filesystem::u8path(argv[8]));
    TORCH_CHECK(out, "cannot open report");
    out << std::setprecision(12) << "{\"model\":\"" << model << "\",\"mode\":\""
        << mode << "\",\"device\":\"" << device << "\",\"batch\":" << batch
        << ",\"repeats\":" << repeats
        << ",\"gpu_memory_measured\":" << (device.is_cuda() ? "true" : "false")
        << ",\"cases\":[";
    for (size_t r = 0; r < roles.size(); ++r) {
      const auto &role = roles[r];
      int64_t omitted = 0, tensors = 1;
      // Destroy equivalence outputs before allocation/timing samples.
      {
        const auto full = encoder.forward(input, mode, role.heads);
        const auto selected =
            encoder.forward(input, mode, role.heads, role.positions);
        assert_equal(full.trunk, selected.trunk);
        TORCH_CHECK(full.pyramid.size() == selected.pyramid.size() &&
                        full.positions.size() == selected.positions.size(),
                    "feature count changed");
        for (const auto &[head, levels] : full.pyramid) {
          TORCH_CHECK(levels.size() == selected.pyramid.at(head).size(),
                      "pyramid level count changed");
          for (size_t i = 0; i < levels.size(); ++i) {
            assert_equal(levels[i], selected.pyramid.at(head)[i]);
            ++tensors;
          }
        }
        for (size_t i = 0; i < full.positions.size(); ++i) {
          if (std::find(role.positions.begin(), role.positions.end(),
                        int64_t(i)) != role.positions.end()) {
            assert_equal(full.positions[i], selected.positions[i]);
            ++tensors;
          } else {
            TORCH_CHECK(!selected.positions[i].defined(),
                        "unrequested position computed");
            omitted += full.positions[i].nbytes();
          }
        }
      }
      // Reuse kernels/allocator after both policies have warmed up.
      for (bool selected : {false, true}) {
        auto warm =
            selected ? encoder.forward(input, mode, role.heads, role.positions)
                     : encoder.forward(input, mode, role.heads);
        sync(device);
      }
      if (r)
        out << ',';
      out << "{\"role\":\"" << role.name
          << "\",\"exact\":true,\"retained_tensors\":" << tensors
          << ",\"omitted_position_bytes\":" << omitted << ",\"samples\":[";
      bool first = true;
      for (int64_t i = 0; i < repeats; ++i)
        for (int order = 0; order < 2; ++order) {
          const bool selected = (i % 2 == 0 ? order : 1 - order) != 0;
          sync(device);
          const auto before = allocated(device);
          reset(device);
          const auto start = std::chrono::steady_clock::now();
          auto value = selected ? encoder.forward(input, mode, role.heads,
                                                  role.positions)
                                : encoder.forward(input, mode, role.heads);
          sync(device);
          const double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
          if (!first)
            out << ',';
          first = false;
          out << "{\"selected\":" << (selected ? "true" : "false")
              << ",\"ms\":" << ms
              << ",\"peak_extra_bytes\":" << allocated(device, true) - before
              << ",\"retained_extra_bytes\":" << allocated(device) - before
              << '}';
        }
      out << "]}";
      std::cout << model << ' ' << mode << " batch=" << batch << ' '
                << role.name << " exact, omitted=" << omitted << '\n';
    }
    // Invalid selectors must fail before attempting neural work.
    int rejected = 0;
    for (const auto &bad :
         std::vector<std::vector<int64_t>>{{-1}, {99}, {2, 2}}) {
      try {
        encoder.forward(at::Tensor(), mode, {}, bad);
      } catch (const c10::Error &error) {
        TORCH_CHECK(std::string(error.what()).find("position level") !=
                        std::string::npos,
                    "selector not validated before image");
        ++rejected;
      }
    }
    TORCH_CHECK(rejected == 3, "invalid position selection accepted");
    out << "],\"invalid_selectors_rejected\":3}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
