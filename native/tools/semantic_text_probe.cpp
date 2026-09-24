// Actual-weight semantic text reuse audit; temporary hard links preserve source
// files.
#include "ppm.h"
#include "sam3/ops.h"
#include "sam3/video_predictor.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_BENCH_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif
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
struct Temporary {
  std::filesystem::path root;
  explicit Temporary(const std::filesystem::path &parent) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (int i = 0; i < 100; ++i) {
      root = parent / ("weight-fixture-" + std::to_string(stamp) + "-" +
                       std::to_string(i));
      if (std::filesystem::create_directory(root))
        return;
    }
    TORCH_CHECK(false, "cannot create fixture directory");
  }
  ~Temporary() {
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }
};
void dump(const std::filesystem::path &dir, const std::string &tag,
          const sam3::VideoOutput &out) {
  std::map<std::string, at::Tensor> fields{
      {"ids", out.ids},
      {"scores", out.probabilities},
      {"boxes", out.boxes_xywh},
      {"masks", sam3::pack_masks(out.masks)},
      {"centers", out.centers}};
  for (const auto &[id, mask] : out.cached_masks)
    fields.emplace("cached-" + std::to_string(id), sam3::pack_masks(mask));
  std::ofstream metadata(dir / (tag + "-metadata.txt"));
  for (const auto &[name, value] : fields) {
    metadata << name << ' ' << value.scalar_type() << ' ' << value.device()
             << ' ' << value.sizes() << ' ' << value.strides() << '\n';
    auto cpu = value.cpu().contiguous();
    std::ofstream file(dir / (tag + "-" + name + ".bin"), std::ios::binary);
    file.write(static_cast<const char *>(cpu.const_data_ptr()), cpu.nbytes());
    TORCH_CHECK(file, "cannot save output");
  }
  for (const auto &[id, mask] : out.cached_masks)
    metadata << "dense-" << id << ' ' << mask.device() << ' ' << mask.sizes()
             << ' ' << mask.strides() << '\n';
  for (const auto &[key, value] : out.frame_stats)
    metadata << "stat-" << key << ' ' << value << '\n';
  metadata.close();
  TORCH_CHECK(metadata, "cannot save metadata");
}
} // namespace
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(
        argc == 10,
        "STORE MODEL DEVICE MODE FRAME.ppm BPE.gz OUTPUT EXPECT_REUSE SAMPLES");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model = argv[2];
    TORCH_CHECK(model == "sam3" || model == "sam3.1", "model");
    const auto device =
        at::empty({0}, at::TensorOptions().device(at::Device(argv[3])))
            .device();
    c10::DeviceGuard guard(device);
    const auto source = std::filesystem::u8path(argv[1]),
               root = std::filesystem::u8path(argv[7]);
    std::filesystem::create_directories(root / "outputs");
    const int expect = std::stoi(argv[8]), samples = std::stoi(argv[9]);
    TORCH_CHECK((expect == 0 || expect == 1) && samples > 0, "options");
    sam3::WeightStore original(source);
    Temporary temporary(root);
    std::set<std::string> shards, text_shards;
    for (const auto &[name, r] : original.records())
      if (name.compare(0, model.size() + 1, model + "/") == 0) {
        shards.insert(r.shard);
        const auto prefix = model + "/detector.backbone.language_backbone.";
        if (name.compare(0, prefix.size(), prefix) == 0)
          text_shards.insert(r.shard);
      }
    TORCH_CHECK(!text_shards.empty(), "no text shards");
    std::filesystem::copy_file(source / "weights.s3i",
                               temporary.root / "weights.s3i");
    // Hard links never modify source bytes. Keep fixture/output on the same
    // local filesystem as weights; only these temporary names are
    // hidden/restored.
    for (const auto &shard : shards)
      std::filesystem::create_hard_link(source / shard, temporary.root / shard);
    const auto hide = [&](bool hidden) {
      for (const auto &shard : text_shards)
        std::filesystem::rename(
            temporary.root / (hidden ? shard : shard + ".hidden"),
            temporary.root / (hidden ? shard + ".hidden" : shard));
    };
    auto rgb = sam3::cli::read_ppm(std::filesystem::u8path(argv[5]));
    int64_t provider_calls = 0;
    auto options = sam3::video_predictor_defaults(
        model == "sam3" ? sam3::AssociationPolicy::Sam3
                        : sam3::AssociationPolicy::Sam31);
    options.mode = argv[4];
    options.centers = true;
    options.sam31_session.history_directory = root / "history";
    sam3::VideoPredictor predictor(
        sam3::WeightStore(temporary.root), std::filesystem::u8path(argv[6]),
        [&](int64_t) {
          ++provider_calls;
          return rgb;
        },
        2, rgb.size(1), rgb.size(2), device, options);
    predictor.set_output_cache(sam3::VideoOutputCacheStorage::PackedCPU);
    const auto missing = [&](const sam3::VideoSemanticPrompt &p) {
      bool failed = false;
      try {
        predictor.add_prompt(0, p);
      } catch (const std::exception &e) {
        TORCH_CHECK(std::string(e.what()).find(temporary.root.string()) !=
                        std::string::npos,
                    "unexpected failure while text shard hidden: ", e.what());
        failed = true;
      }
      return failed;
    };
    const std::vector<std::optional<std::string>> texts = {
        "person",     "",     "visual",
        std::nullopt, "人物", "a person with a red bag"};
    int misses = 0, hits = 0;
    int64_t sequence = 0;
    for (const auto &text : texts) {
      predictor.reset();
      sam3::VideoSemanticPrompt p;
      p.text = text;
      p.boxes_xywh = at::tensor({.30, .15, .35, .70}, at::kFloat).view({1, 4});
      p.box_labels = at::tensor({1}, at::kLong);
      const auto tag = std::to_string(sequence++);
      dump(root / "outputs", tag + "-initial", predictor.add_prompt(0, p));
      auto replacement = p;
      replacement.box_labels = at::tensor({0}, at::kLong);
      replacement.visual_features =
          at::arange(512, at::kFloat).view({2, 1, 256}) / 512.;
      replacement.visual_padding = at::zeros({1, 2}, at::kBool);
      const auto encodes = predictor.visual_encodes(), reads = provider_calls;
      rgb = rgb.flip({2}).contiguous();
      hide(true);
      sam3::VideoOutput repeated;
      bool failed = false;
      try {
        repeated = predictor.add_prompt(1, replacement);
      } catch (const std::exception &e) {
        TORCH_CHECK(std::string(e.what()).find(temporary.root.string()) !=
                        std::string::npos,
                    "unexpected same-text failure: ", e.what());
        failed = true;
      }
      hide(false);
      TORCH_CHECK(failed == (expect == 0),
                  "same-text reuse expectation failed");
      if (failed)
        repeated = predictor.add_prompt(1, replacement);
      else
        ++hits;
      TORCH_CHECK(predictor.visual_encodes() == encodes + 1 &&
                      provider_calls == reads + 1,
                  "semantic replacement reused stale image features");
      dump(root / "outputs", tag + "-geometry-visual", repeated);
      // Invalid geometry must fail before resetting the current text/state.
      auto invalid = replacement;
      invalid.boxes_xywh = at::full({1, 4}, 2., at::kFloat);
      auto ids = predictor.metadata().object_ids();
      bool rejected = false;
      try {
        predictor.add_prompt(0, invalid);
      } catch (const c10::Error &) {
        rejected = true;
      }
      TORCH_CHECK(rejected && predictor.metadata().object_ids() == ids,
                  "invalid prompt reset state");
      auto changed = replacement;
      changed.text = text ? std::optional<std::string>(*text + " changed")
                          : std::optional<std::string>("");
      hide(true);
      TORCH_CHECK(missing(changed),
                  "changed optional text reused stale encoding");
      ++misses;
      hide(false);
      dump(root / "outputs", tag + "-changed",
           predictor.add_prompt(0, changed));
      predictor.reset();
      hide(true);
      TORCH_CHECK(missing(changed), "public reset retained text encoding");
      ++misses;
      hide(false);
      dump(root / "outputs", tag + "-reset", predictor.add_prompt(0, changed));
    }
    // Full semantic replacement timings include image/detector/tracker work.
    predictor.reset();
    sam3::VideoSemanticPrompt p;
    p.text = "person";
    p.boxes_xywh = at::tensor({.30, .15, .35, .70}, at::kFloat).view({1, 4});
    p.box_labels = at::tensor({1}, at::kLong);
    predictor.add_prompt(0, p);
    int settings_invalidations = 0;
    const auto invalidate = [&](const auto &change, const auto &restore,
                                const std::string &tag) {
      change();
      hide(true);
      const bool failed = missing(p);
      hide(false);
      restore();
      TORCH_CHECK(failed, "inference policy change reused stale text");
      ++settings_invalidations;
      ++misses;
      dump(root / "outputs", "settings-" + tag, predictor.add_prompt(0, p));
    };
    invalidate([] { at::set_num_threads(3); }, [] { at::set_num_threads(4); },
               "threads");
    invalidate([] { at::globalContext().setAllowTF32CuBLAS(true); },
               [] { at::globalContext().setAllowTF32CuBLAS(false); }, "tf32");
    const bool flash = at::globalContext().userEnabledFlashSDP();
    invalidate([&] { at::globalContext().setSDPUseFlash(!flash); },
               [&] { at::globalContext().setSDPUseFlash(flash); }, "attention");
    predictor.add_prompt(0, p);
    sync(device);
    std::ofstream report(root / "report.json");
    report << std::setprecision(12)
           << "{\"reuse_hits_with_hidden_shards\":" << hits
           << ",\"required_misses\":" << misses
           << ",\"settings_invalidations\":" << settings_invalidations
           << ",\"samples_ms\":[";
    for (int i = 0; i < samples; ++i) {
      p.box_labels = at::tensor({i % 2}, at::kLong);
      sync(device);
      auto before = std::chrono::steady_clock::now();
      auto out = predictor.add_prompt(i % 2, p);
      sync(device);
      auto ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - before)
                    .count();
      if (i)
        report << ',';
      report << ms;
      dump(root / "outputs", "timed-" + std::to_string(i), out);
    }
    report << "],\"visual_encodes\":" << predictor.visual_encodes()
           << ",\"provider_calls\":" << provider_calls << "}\n";
    report.close();
    TORCH_CHECK(report, "write report");
    predictor.reset();
    std::cout << "PASS geometry/visual replacement, optional text distinction, "
                 "changed-text miss, reset, invalid prompt, missing-file "
                 "recovery and fresh image reads\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
