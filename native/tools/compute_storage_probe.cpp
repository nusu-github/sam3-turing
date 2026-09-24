// Actual-weight FP16 compute-storage equivalence fixture. CUDA-only diagnostic.
#include "ppm.h"
#include "sam3/preprocess.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
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
        argc == 9,
        "STORE MODEL float|half MODE FRAME BATCH BPE OUTPUT (CUDA device 0)");
    c10::InferenceMode guard;
    at::set_num_threads(4);
    const at::Device device(at::kCUDA, 0);
    c10::DeviceGuard dg(device);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string storage = argv[3], mode = argv[4];
    TORCH_CHECK(storage == "float" || storage == "half", "storage");
    const auto dtype = storage == "half" ? at::kHalf : at::kFloat;
    const int64_t batch = std::stoll(argv[6]);
    sam3::WeightStore store(argv[1]);
    std::filesystem::path dir(argv[8]);
    std::filesystem::create_directories(dir);
    std::ofstream meta(dir / "metadata.txt");
    auto save = [&](std::string name, const at::Tensor &t) {
      auto c = t.cpu().contiguous();
      std::ofstream f(dir / (name + ".bin"), std::ios::binary);
      f.write(static_cast<const char *>(c.const_data_ptr()), c.nbytes());
      TORCH_CHECK(f, "write failed");
      meta << name << ' ' << t.scalar_type() << ' ' << t.sizes() << ' '
           << t.strides() << '\n';
    };
    auto allocated = []() {
      return c10::cuda::CUDACachingAllocator::getDeviceStats(0)
          .allocated_bytes[0]
          .current;
    };
    auto sync = []() { c10::cuda::device_synchronize(); };
    at::empty({1}, at::TensorOptions().device(device));
    using Clock = std::chrono::steady_clock;
    sync();
    const auto before = allocated();
    const auto start = Clock::now();
    int invalid_storage_rejected = 0;
    for (auto bad : std::vector<std::pair<at::Device, at::ScalarType>>{
             {at::kCPU, at::kHalf},
             {device, at::kDouble},
             {device, at::kBFloat16}}) {
      try {
        sam3::VisionEncoder ignored(store, argv[2], bad.first, bad.second);
      } catch (const c10::Error &e) {
        TORCH_CHECK(std::string(e.what()).find("compute storage must") !=
                        std::string::npos,
                    "unexpected vision rejection");
        ++invalid_storage_rejected;
      }
      try {
        sam3::TextEncoder ignored(store, argv[2], bad.first, bad.second);
      } catch (const c10::Error &e) {
        TORCH_CHECK(std::string(e.what()).find("compute storage must") !=
                        std::string::npos,
                    "unexpected text rejection");
        ++invalid_storage_rejected;
      }
    }
    TORCH_CHECK(invalid_storage_rejected == 6, "invalid storage accepted");
    sam3::VisionEncoder v(store, argv[2], device, dtype);
    sync();
    const auto vision_bytes = allocated() - before;
    sam3::TextEncoder text(store, argv[2], device, dtype);
    sync();
    const auto total_bytes = allocated() - before;
    const auto load =
        std::chrono::duration<double>(Clock::now() - start).count();
    auto x = sam3::preprocess_rgb(sam3::cli::read_ppm(argv[5]).to(device))
                 .repeat({batch, 1, 1, 1});
    auto out = v.forward(x, mode);
    save("trunk", out.trunk);
    for (const auto &[head, levels] : out.pyramid)
      for (size_t i = 0; i < levels.size(); ++i)
        save(head + "." + std::to_string(i), levels[i]);
    for (size_t i = 0; i < out.positions.size(); ++i)
      save("position." + std::to_string(i), out.positions[i]);
    sam3::Tokenizer tokenizer(argv[7]);
    int checks = 0;
    for (const auto &prompts : std::vector<std::vector<std::string>>{
             {"person"}, {"", "a red car", "人物", "a dog beside a chair"}}) {
      auto ids = tokenizer.tokenize(prompts);
      std::vector<int64_t> flat;
      for (auto &row : ids)
        flat.insert(flat.end(), row.begin(), row.end());
      for (int64_t length : {1, 7, 32}) {
        auto tokens = at::tensor(flat, at::TensorOptions().dtype(at::kLong))
                          .view({int64_t(ids.size()), 32})
                          .slice(1, 0, length);
        for (auto type : {at::kLong, at::kInt}) {
          auto encoded = text.forward(tokens.to(type), mode);
          const auto name = "text" + std::to_string(checks++);
          save(name + ".padding", std::get<0>(encoded));
          save(name + ".memory", std::get<1>(encoded));
          save(name + ".embeddings", std::get<2>(encoded));
        }
      }
    }
    if (storage == "half") {
      for (const std::string other : {"fp32", "bf16_reference"}) {
        bool rejected = false;
        try {
          v.forward(x, other);
        } catch (const c10::Error &e) {
          rejected =
              std::string(e.what()).find("requires fp16") != std::string::npos;
        }
        TORCH_CHECK(rejected, "vision allowed incompatible mode");
        rejected = false;
        try {
          text.forward(at::ones({1, 1}, at::TensorOptions().dtype(at::kLong)),
                       other);
        } catch (const c10::Error &e) {
          rejected =
              std::string(e.what()).find("requires fp16") != std::string::npos;
        }
        TORCH_CHECK(rejected, "text allowed incompatible mode");
      }
    }
    std::ofstream report(dir / "measurements.json");
    report << std::setprecision(12)
           << "{\"vision_parameter_bytes\":" << vision_bytes
           << ",\"text_parameter_bytes\":" << total_bytes - vision_bytes
           << ",\"total_parameter_bytes\":" << total_bytes
           << ",\"load_seconds\":" << load << ",\"text_cases\":" << checks
           << "}\n";
    std::cout << "PASS " << argv[2] << ' ' << storage << ' ' << mode
              << " batch=" << batch << " bytes=" << total_bytes << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
