#include "ppm.h"
#include "sam3/video_predictor.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <sstream>
namespace {
void same(const sam3::VideoOutput &a, const sam3::VideoOutput &b) {
  for (auto pair : {std::make_pair(a.ids, b.ids),
                    {a.probabilities, b.probabilities},
                    {a.boxes_xywh, b.boxes_xywh},
                    {a.masks, b.masks},
                    {a.centers, b.centers}})
    TORCH_CHECK(at::equal(pair.first.cpu(), pair.second.cpu()),
                "single/multiple logical rank output differs");
  TORCH_CHECK(a.cached_masks.size() == b.cached_masks.size(),
              "cache count differs");
  for (const auto &[id, mask] : a.cached_masks)
    TORCH_CHECK(at::equal(mask.cpu(), b.cached_masks.at(id).cpu()),
                "cached mask differs");
}
} // namespace
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 8,
                "usage: video_multidevice_probe STORE sam3|sam3.1 COORDINATOR "
                "MODE FRAME.ppm BPE.gz TRACKING_DEVICES");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    const std::string model = argv[2];
    TORCH_CHECK(model == "sam3" || model == "sam3.1", "unknown model");
    const auto device =
        at::empty({0}, at::TensorOptions().device(at::Device(argv[3])))
            .device();
    std::vector<at::Device> devices;
    std::istringstream names(argv[7]);
    std::string name;
    while (std::getline(names, name, ','))
      devices.push_back(
          at::empty({0}, at::TensorOptions().device(at::Device(name)))
              .device());
    TORCH_CHECK(devices.size() >= 2, "probe requires at least two ranks");
    const bool logical = std::all_of(devices.begin(), devices.end(),
                                     [&](auto d) { return d == device; });
    auto rgb = sam3::cli::read_ppm(std::filesystem::u8path(argv[5]));
    auto options = sam3::video_predictor_defaults(
        model == "sam3" ? sam3::AssociationPolicy::Sam3
                        : sam3::AssociationPolicy::Sam31);
    options.mode = argv[4];
    options.centers = true;
    sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    const auto vocab = std::filesystem::u8path(argv[6]);
    sam3::VideoPredictor multi(
        store, vocab, [&](int64_t) { return rgb; }, 3, rgb.size(1), rgb.size(2),
        device, options);
    multi.set_tracking_devices(devices);
    TORCH_CHECK(multi.tracking_devices() == devices,
                "device list not retained");
    bool caught = false;
    try {
      multi.set_tracking_devices({});
    } catch (const c10::Error &) {
      caught = true;
    }
    TORCH_CHECK(caught && multi.tracking_devices() == devices,
                "failed configuration changed ranks");
    std::unique_ptr<sam3::VideoPredictor> single;
    if (logical)
      single = std::make_unique<sam3::VideoPredictor>(
          store, vocab, [&](int64_t) { return rgb; }, 3, rgb.size(1),
          rgb.size(2), device, options);
    int comparisons = 0;
    const auto compare = [&](const sam3::VideoOutput &a,
                             const sam3::VideoOutput &b) {
      same(a, b);
      ++comparisons;
    };
    auto mask = at::zeros({rgb.size(1), rgb.size(2)}, at::kBool);
    mask.slice(0, rgb.size(1) / 4, rgb.size(1) / 2)
        .slice(1, rgb.size(2) / 4, rgb.size(2) / 2)
        .fill_(true);
    for (auto id : {int64_t(-7), int64_t(42), int64_t(9001)}) {
      auto out = multi.add_mask(0, id, mask);
      if (single)
        compare(out, single->add_mask(0, id, mask));
      TORCH_CHECK(at::equal(out.cached_masks.at(id).squeeze(0).cpu(), mask),
                  "annotation changed");
      for (const auto &[key, value] : out.cached_masks)
        TORCH_CHECK(value.device() == device,
                    "edit failed to collect masks on coordinator");
    }
    TORCH_CHECK(multi.metadata().ids_per_rank[1] == std::vector<int64_t>{42},
                "new edits did not select least-loaded rank");
    caught = false;
    try {
      multi.set_tracking_devices({device});
    } catch (const c10::Error &) {
      caught = true;
    }
    TORCH_CHECK(caught, "active device reconfiguration accepted");
    sam3::VideoPredictorPropagation request;
    request.start = 0;
    request.max_steps = 2;
    std::vector<sam3::VideoOutput> expected;
    if (single)
      single->propagate(request, [&](int64_t, const auto &out) {
        expected.push_back(out);
        return true;
      });
    int emitted = 0;
    multi.propagate(request, [&](int64_t, const auto &out) {
      if (single)
        compare(out, expected.at(emitted));
      ++emitted;
      return true;
    });
    TORCH_CHECK(emitted == 3, "partial propagation omitted frames");
    multi.remove_object(42);
    if (single)
      single->remove_object(42);
    TORCH_CHECK(multi.metadata().ids_per_rank[1].empty(),
                "remove touched wrong rank");
    sam3::TrackingPoints points;
    points.points = at::tensor({.45, .55}, at::kFloat).view({1, 2});
    points.labels = at::tensor({1}, at::kLong);
    auto edited = multi.add_points(1, 42, points);
    if (single)
      compare(edited, single->add_points(1, 42, points));
    TORCH_CHECK(multi.metadata().ids_per_rank[1] == std::vector<int64_t>{42},
                "point edit lost owner");
    request.start = 2;
    request.reverse = true;
    expected.clear();
    if (single)
      single->propagate(request, [&](int64_t, const auto &out) {
        expected.push_back(out);
        return true;
      });
    emitted = 0;
    multi.propagate(request, [&](int64_t, const auto &out) {
      if (single)
        compare(out, expected.at(emitted));
      ++emitted;
      return true;
    });
    TORCH_CHECK(emitted == 2, "reverse propagation range changed");
    if (single)
      compare(multi.fetch(1), single->fetch(1));
    // Cancel from the callback and resume the action route without retaining
    // partial output buffers.
    request.start = 0;
    request.reverse = false;
    emitted = 0;
    multi.propagate(request, [&](int64_t, const auto &) {
      ++emitted;
      multi.cancel();
      return true;
    });
    TORCH_CHECK(emitted == 1, "cancel ignored");
    multi.reset();
    TORCH_CHECK(multi.tracking_devices() == devices &&
                    multi.metadata().object_ids().empty() &&
                    multi.interaction().actions().empty() &&
                    multi.interaction().cached_frames().empty(),
                "reset lost config or retained observations");
    multi.set_tracking_devices({device});
    TORCH_CHECK(multi.metadata().ids_per_rank.size() == 1,
                "reset reconfiguration failed");
    auto restored = multi.add_mask(0, 42, mask);
    TORCH_CHECK(at::equal(restored.cached_masks.at(42).squeeze(0).cpu(), mask),
                "reconfigured owner unusable");
    multi.reset();
    multi.set_tracking_devices(devices);
    sam3::VideoSemanticPrompt prompt;
    prompt.text = "person";
    auto detection = multi.add_prompt(0, prompt);
    TORCH_CHECK(!multi.metadata().object_ids().empty(),
                "semantic fixture has no detections");
    const auto id = multi.metadata().object_ids().front();
    size_t expected_rank = 0;
    std::vector<size_t> counts;
    for (const auto &rank : multi.metadata().ids_per_rank) {
      auto count = rank.size();
      if (std::find(rank.begin(), rank.end(), id) != rank.end())
        --count;
      counts.push_back(count);
    }
    for (size_t r = 1; r < counts.size(); ++r)
      if (counts[r] < counts[expected_rank])
        expected_rank = r;
    multi.add_points(0, id, points, true, false, true);
    const auto &placed = multi.metadata().ids_per_rank[expected_rank];
    TORCH_CHECK(std::find(placed.begin(), placed.end(), id) != placed.end(),
                "stateless refinement did not reassign least-loaded rank");
    multi.reset();
    std::cout << "{\"model\":\"" << model << "\",\"mode\":\"" << argv[4]
              << "\",\"ranks\":" << devices.size()
              << ",\"all_ranks_on_coordinator\":"
              << (logical ? "true" : "false")
              << ",\"exact_single_rank_comparisons\":" << comparisons
              << ",\"lifecycle_pass\":true}" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
