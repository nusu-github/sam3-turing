#include "sam3/grounding.h"
#include "sam3/image_results.h"
#include "sam3/media.h"
#include "sam3/ops.h"
#include "sam3/preprocess.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include "sam3/vision_encoder.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_BENCH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
namespace {
using Clock = std::chrono::steady_clock;
std::string read(const std::filesystem::path &path) {
  std::ifstream f(path, std::ios::binary);
  TORCH_CHECK(f, "cannot open ", path.u8string());
  return {std::istreambuf_iterator<char>(f), {}};
}
void sync(at::Device device) {
#ifdef SAM3_BENCH_CUDA
  if (device.is_cuda())
    c10::cuda::device_synchronize();
#endif
}
struct Memory {
  int64_t allocated = 0, reserved = 0;
};
Memory memory(at::Device device) {
#ifdef SAM3_BENCH_CUDA
  if (device.is_cuda()) {
    auto s = c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    return {s.allocated_bytes[0].peak, s.reserved_bytes[0].peak};
  }
#endif
  return {};
}
void reset_peak(at::Device device) {
#ifdef SAM3_BENCH_CUDA
  if (device.is_cuda())
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
#endif
}
struct Row {
  int64_t image, category;
  std::filesystem::path path, prompt;
};
} // namespace
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(
        argc == 8 ||
            (argc == 9 && std::string(argv[8]) == "--save-probabilities"),
        "usage: sam3_image_benchmark STORE sam3|sam3.1 "
        "DEVICE MODE MANIFEST.tsv BPE.gz OUTPUT [--save-probabilities]");
    const bool save_probabilities = argc == 9;
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model = argv[2], mode = argv[4];
    TORCH_CHECK(model == "sam3" || model == "sam3.1", "invalid model");
    TORCH_CHECK(mode == "fp16" || mode == "bf16_reference" || mode == "fp32",
                "invalid precision");
    const auto device =
        at::empty({0}, at::TensorOptions().device(at::Device(argv[3])))
            .device();
#ifndef SAM3_BENCH_CUDA
    TORCH_CHECK(!device.is_cuda(), "CUDA benchmarking requires CUDA "
                                   "timing/statistics support at build time");
#endif
    c10::DeviceGuard guard(device);
    const auto manifest = std::filesystem::u8path(argv[5]),
               root = std::filesystem::u8path(argv[7]);
    std::filesystem::create_directories(root);
    std::vector<Row> rows;
    std::istringstream input(read(manifest));
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream row(line);
      std::vector<std::string> fields;
      std::string field;
      while (std::getline(row, field, '\t'))
        fields.push_back(field);
      TORCH_CHECK(fields.size() == 4, "manifest requires image ID, category "
                                      "ID, image path and prompt path");
      auto image = std::stoll(fields[0]), category = std::stoll(fields[1]);
      TORCH_CHECK(image >= 0 && category >= 0, "negative manifest ID");
      auto path = std::filesystem::u8path(fields[2]),
           prompt = std::filesystem::u8path(fields[3]);
      rows.push_back(
          {image, category,
           path.is_absolute() ? path : manifest.parent_path() / path,
           prompt.is_absolute() ? prompt : manifest.parent_path() / prompt});
    }
    TORCH_CHECK(!rows.empty(), "empty manifest");
    const auto start = Clock::now();
    sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    sam3::Tokenizer tokenizer(std::filesystem::u8path(argv[6]));
    sam3::VisionEncoder vision(store, model, device);
    sam3::TextEncoder text_encoder(store, model, device);
    sam3::GroundingDetector detector(store, model, device);
    std::map<std::string, std::pair<at::Tensor, at::Tensor>> texts;
    for (const auto &row : rows) {
      const auto text = read(row.prompt);
      if (texts.count(text))
        continue;
      const auto ids = tokenizer.tokenize({text})[0];
      auto encoded = text_encoder.forward(
          at::tensor(ids, at::TensorOptions().dtype(at::kLong).device(device))
              .unsqueeze(0),
          mode);
      texts.emplace(text,
                    std::make_pair(std::get<0>(encoded), std::get<1>(encoded)));
    }
    sync(device);
    const double setup =
        std::chrono::duration<double>(Clock::now() - start).count();
    const auto o = at::TensorOptions().device(device);
    const auto ids = at::zeros({1}, o.dtype(at::kLong)),
               labels = at::empty({0, 1}, o.dtype(at::kLong)),
               padding = at::empty({1, 0}, o.dtype(at::kBool));
    const sam3::GeometryPrompt geometry{
        at::empty({0, 1, 2}, o), labels, padding,
        at::empty({0, 1, 4}, o), labels, padding};
    std::filesystem::path current;
    int64_t h = 0, w = 0, encodes = 0;
    std::vector<at::Tensor> pyramid;
    at::Tensor position;
    bool warmed = false;
    std::ofstream metrics(root / "metrics.jsonl");
    TORCH_CHECK(metrics, "cannot create metrics");
    metrics << std::setprecision(9);
    for (const auto &row : rows) {
      double image_seconds = 0;
      Memory image_peak;
      reset_peak(device);
      if (current != row.path) {
        sync(device);
        const auto before = Clock::now();
        sam3::MediaSource source(row.path, {true, 1});
        auto rgb = source.read(0).rgb;
        h = rgb.size(1);
        w = rgb.size(2);
        auto features = vision.forward(sam3::preprocess_rgb(rgb.to(device)),
                                       mode, {"convs"}, {2});
        pyramid = std::move(features.pyramid.at("convs"));
        if (model == "sam3")
          pyramid.pop_back();
        position = features.positions[pyramid.size() - 1];
        current = row.path;
        ++encodes;
        sync(device);
        image_seconds =
            std::chrono::duration<double>(Clock::now() - before).count();
        image_peak = memory(device);
      }
      const auto &text = texts.at(read(row.prompt));
      sam3::GroundingPrompt prompt{ids, ids, text.second, text.first, geometry};
      if (!warmed) {
        for (int i = 0; i < 2; ++i) {
          auto raw = detector.forward(pyramid, position, prompt,
                                      model == "sam3.1", mode);
          auto ignored = sam3::postprocess_image(raw.detection, {h}, {w}, -1,
                                                 model == "sam3", 8, mode);
        }
        sync(device);
        warmed = true;
        reset_peak(device);
      }
      const auto before = Clock::now();
      auto raw =
          detector.forward(pyramid, position, prompt, model == "sam3.1", mode);
      TORCH_CHECK(raw.detection.logits.size(1) == 200, "query count changed");
      auto results = sam3::postprocess_image(raw.detection, {h}, {w}, -1,
                                             model == "sam3", 8, mode);
      sync(device);
      const double seconds =
          std::chrono::duration<double>(Clock::now() - before).count();
      bool finite = true;
      for (const auto &value :
           {raw.detection.logits, raw.detection.boxes, raw.detection.masks,
            raw.detection.presence_logits, results[0].scores, results[0].boxes})
        if (value.defined())
          finite = finite && at::isfinite(value).all().item<bool>();
      TORCH_CHECK(finite, "nonfinite detector output");
      auto peak = memory(device);
      peak.allocated = std::max(peak.allocated, image_peak.allocated);
      peak.reserved = std::max(peak.reserved, image_peak.reserved);
      const auto &result = results[0];
      TORCH_CHECK(result.scores.numel() == 200,
                  "audit must retain all queries, including zero scores");
      auto packed =
          sam3::pack_masks(result.masks.squeeze(1)).cpu().contiguous();
      const auto scores = result.scores.cpu().to(at::kFloat).contiguous(),
                 boxes = result.boxes.cpu().to(at::kFloat).contiguous(),
                 queries = result.query_indices.cpu().contiguous();
      const auto prefix =
          std::to_string(row.image) + "-" + std::to_string(row.category);
      std::ofstream binary(root / (prefix + ".masks.bin"), std::ios::binary);
      binary.write(static_cast<const char *>(packed.const_data_ptr()),
                   packed.nbytes());
      TORCH_CHECK(binary, "cannot write masks");
      if (save_probabilities) {
        const auto probabilities = result.mask_probabilities.cpu().contiguous();
        std::ofstream output(root / (prefix + ".probabilities.bin"),
                             std::ios::binary);
        output.write(static_cast<const char *>(probabilities.const_data_ptr()),
                     probabilities.nbytes());
        TORCH_CHECK(output, "cannot write full probability maps");
      }
      std::ofstream json(root / (prefix + ".json"));
      TORCH_CHECK(json, "cannot write detections");
      json << std::setprecision(9) << "{\"image_id\":" << row.image
           << ",\"category_id\":" << row.category << ",\"height\":" << h
           << ",\"width\":" << w << ",\"mask_bytes\":" << packed.size(1);
      if (save_probabilities)
        json << ",\"probability_dtype\":\""
             << c10::toString(result.mask_probabilities.scalar_type())
             << "\",\"probability_bytes\":"
             << result.mask_probabilities.nbytes();
      json << ",\"detections\":[";
      for (int64_t i = 0; i < scores.numel(); ++i) {
        if (i)
          json << ',';
        json << "{\"query\":" << queries[i].item<int64_t>()
             << ",\"score\":" << scores[i].item<float>() << ",\"box\":[";
        for (int j = 0; j < 4; ++j) {
          if (j)
            json << ',';
          json << boxes[i][j].item<float>();
        }
        json << "]}";
      }
      json << "]}\n";
      TORCH_CHECK(json, "failed to write detections");
      metrics << "{\"image_id\":" << row.image
              << ",\"category_id\":" << row.category
              << ",\"image_seconds\":" << image_seconds
              << ",\"detector_postprocess_seconds\":" << seconds
              << ",\"peak_allocated_bytes\":" << peak.allocated
              << ",\"peak_reserved_bytes\":" << peak.reserved
              << ",\"finite\":true,\"queries\":200}\n";
      metrics.flush();
      std::cout << prefix << " queries=200 seconds=" << seconds
                << " peak=" << peak.allocated << std::endl;
    }
    std::ofstream summary(root / "summary.json");
    summary
        << "{\"model\":\"" << model << "\",\"mode\":\"" << mode
        << "\",\"cases\":" << rows.size() << ",\"visual_encodes\":" << encodes
        << ",\"text_prompts\":" << texts.size()
        << ",\"setup_seconds\":" << setup
        << ",\"warmup_detector_calls\":2,\"threshold\":-1,\"tf32\":false}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
