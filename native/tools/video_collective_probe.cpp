// Actual tracker weights, controlled feature/mask inputs. This verifies rank
// execution semantics, not end-to-end visual accuracy or physical multi-GPU
// speed.
#include "sam3/video_collective.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <sstream>
#include <type_traits>
namespace {
sam3::TrackingFeatures features(at::Device device, int64_t frame) {
  auto o = at::TensorOptions().device(device).dtype(at::kFloat);
  return {at::full({1, 256, 72, 72}, frame * .01, o),
          at::zeros({1, 256, 72, 72}, o),
          {at::zeros({1, 32, 288, 288}, o), at::zeros({1, 64, 144, 144}, o)}};
}
template <class Rank>
sam3::VideoRankPrediction
reference_propagate(int64_t frame, bool reverse, const std::vector<Rank> &ranks,
                    const sam3::VideoMetadata &metadata, at::Device device,
                    int64_t cleanup) {
  std::vector<at::Tensor> masks, scores;
  for (size_t r = 0; r < ranks.size(); ++r) {
    std::vector<at::Tensor> low, logits;
    std::vector<int64_t> ids;
    sam3::TrackingPropagation request;
    request.start = frame;
    request.max_steps = 0;
    request.reverse = reverse;
    request.encode_memory = false;
    for (auto &session : *ranks[r].sessions)
      session->propagate(request, [&](const auto &v) {
        ids.insert(ids.end(), v.object_ids.begin(), v.object_ids.end());
        low.push_back(v.low_masks.squeeze(1));
        logits.push_back(v.object_logits.flatten());
        return true;
      });
    const auto o =
        at::TensorOptions().device(ranks[r].device).dtype(at::kFloat);
    auto m = at::empty({0, 288, 288}, o), s = at::empty({0}, o);
    if (!ids.empty()) {
      auto rows = at::tensor(
          sam3::video_memory_rows(ids, {metadata.ids_per_rank[r]})[0],
          o.dtype(at::kLong));
      m = sam3::clean_video_mask_scores(at::cat(low).unsqueeze(1), cleanup)
              .squeeze(1)
              .index_select(0, rows);
      s = at::cat(logits).index_select(0, rows);
    }
    if (ranks.size() > 1) {
      m = m.to(at::kFloat);
      s = s.to(at::kFloat);
    }
    // Independent serial oracle: ordinary ATen copy, no collective helpers.
    masks.push_back(m.to(device));
    scores.push_back(s.to(device));
  }
  return {metadata.object_ids(), at::cat(masks), at::cat(scores)};
}
template <class Rank>
void reference_execute(int64_t frame, sam3::VideoUpdatePlan &plan,
                       const sam3::VideoDetections &d,
                       const std::vector<Rank> &ranks,
                       const sam3::VideoUpdateOptions &options) {
  for (size_t r = 0; r < ranks.size(); ++r) {
    auto local = plan;
    const auto copy = [&](const at::Tensor &x) {
      return x.defined() ? x.to(ranks[r].device) : at::Tensor();
    };
    local.tracking_masks = copy(plan.tracking_masks);
    local.corrections.binary_masks = copy(plan.corrections.binary_masks);
    local.corrections.low_masks = copy(plan.corrections.low_masks);
    sam3::VideoDetections dd{copy(d.masks), copy(d.scores), copy(d.boxes),
                             copy(d.keep)};
    sam3::execute_video_update(frame, r, local, dd, *ranks[r].sessions,
                               *ranks[r].factory, options);
    plan.metadata.buckets_per_rank[r] = local.metadata.buckets_per_rank[r];
    plan.reconditioned.insert(local.reconditioned.begin(),
                              local.reconditioned.end());
  }
}
template <bool Mux>
void run(const sam3::WeightStore &store, const std::vector<at::Device> &devices,
         const std::string &mode) {
  using Core = std::conditional_t<Mux, sam3::Sam31TrackingFrame,
                                  sam3::Sam3TrackingFrame>;
  using Session = std::conditional_t<Mux, sam3::Sam31TrackingSession,
                                     sam3::Sam3TrackingSession>;
  using Sessions = std::vector<std::unique_ptr<Session>>;
  using Factory = std::function<std::unique_ptr<Session>()>;
  using Rank =
      std::conditional_t<Mux, sam3::Sam31VideoRank, sam3::Sam3VideoRank>;
  using Options = std::conditional_t<Mux, sam3::MultiplexSessionOptions,
                                     sam3::TrackingSessionOptions>;
  std::map<std::string, std::shared_ptr<Core>> cores;
  std::vector<Sessions> actual(devices.size()), expected(devices.size());
  std::vector<Factory> factories;
  factories.reserve(devices.size());
  for (auto device : devices) {
    auto &core = cores[device.str()];
    if (!core)
      core = std::make_shared<Core>(store, device);
    Options policy;
    policy.offload_state = true;
    if constexpr (Mux)
      policy.all_edits_conditioning = false;
    factories.push_back([core, device, mode, policy] {
      if constexpr (Mux)
        return std::make_unique<Session>(
            core,
            [device](int64_t f) {
              auto v = features(device, f);
              return sam3::MultiplexTrackingFeatures{v, v};
            },
            4, 73, 91, device, mode, policy);
      else
        return std::make_unique<Session>(
            core, [device](int64_t f) { return features(device, f); }, 4, 73,
            91, device, mode, policy);
    });
  }
  std::vector<Rank> ar, er;
  for (size_t i = 0; i < devices.size(); ++i) {
    ar.push_back({devices[i], &actual[i], &factories[i]});
    er.push_back({devices[i], &expected[i], &factories[i]});
  }
  const auto device = devices.front();
  const auto o = at::TensorOptions().device(device).dtype(at::kFloat);
  sam3::VideoUpdateOptions options;
  options.association.policy = options.recondition.policy =
      Mux ? sam3::AssociationPolicy::Sam31 : sam3::AssociationPolicy::Sam3;
  options.cleanup_area = Mux ? 0 : 16;
  auto metadata = sam3::initialize_video_metadata(devices.size(), device);
  auto masks = at::full({17, 288, 288}, -2., o);
  for (int64_t i = 0; i < 17; ++i)
    masks[i]
        .slice(0, 8 + (i / 5) * 60, 48 + (i / 5) * 60)
        .slice(1, 8 + (i % 5) * 50, 43 + (i % 5) * 50)
        .fill_(2.);
  sam3::VideoDetections d{masks, at::full({17}, .95, o), at::zeros({17, 4}, o),
                          at::ones({17}, o.dtype(at::kBool))};
  auto plan = sam3::plan_video_update(0, false, d, at::empty({0, 288, 288}, o),
                                      at::empty({0}, o), metadata, options);
  auto oracle = plan;
  reference_execute(0, oracle, d, er, options);
  sam3::execute_video_update_ranks(0, plan, d, ar, options);
  TORCH_CHECK(plan.metadata.ids_per_rank == oracle.metadata.ids_per_rank &&
                  plan.metadata.buckets_per_rank ==
                      oracle.metadata.buckets_per_rank,
              "birth rank metadata differs");
  metadata = plan.metadata;
  TORCH_CHECK(metadata.object_ids().size() == 17, "births truncated");
  int checks = 0;
  const auto prediction = [&](int64_t frame, bool reverse) {
    auto e = reference_propagate(frame, reverse, er, metadata, device,
                                 options.cleanup_area);
    auto a = sam3::propagate_video_tracking_ranks(frame, reverse, ar, metadata,
                                                  device, options.cleanup_area);
    TORCH_CHECK(
        a.ids == e.ids && a.masks.scalar_type() == e.masks.scalar_type() &&
            at::equal(a.masks, e.masks) && at::equal(a.logits, e.logits),
        "neural rank predictions differ");
    ++checks;
    return a;
  };
  auto p = prediction(1, false);
  auto none =
      sam3::VideoDetections{d.masks.slice(0, 0, 0), d.scores.slice(0, 0, 0),
                            d.boxes.slice(0, 0, 0), d.keep.slice(0, 0, 0)};
  plan = sam3::plan_video_update(1, false, none, p.masks, p.logits, metadata,
                                 options);
  TORCH_CHECK(plan.removed.empty(),
              "controlled first update unexpectedly removed objects");
  // Global visibility: only the first object wins. A rank-local approximation
  // would wrongly preserve another visible object on each nonempty worker.
  plan.tracking_masks = at::ones_like(p.masks);
  plan.tracking_masks[0].fill_(2.);
  const auto memory = sam3::prepare_video_memory(
      plan.tracking_masks, options.association.policy, true);
  TORCH_CHECK(memory.object_logits[0].item<float>() == 10 &&
                  memory.object_logits.slice(0, 1).eq(-10).all().item<bool>(),
              "global visibility fixture invalid");
  plan.corrections.ids = {metadata.object_ids().front(),
                          metadata.object_ids().back()};
  plan.corrections.binary_masks =
      at::stack({masks[plan.corrections.ids[0]].gt(0),
                 masks[plan.corrections.ids[1]].gt(0)});
  oracle = plan;
  reference_execute(1, oracle, none, er, options);
  sam3::execute_video_update_ranks(1, plan, none, ar, options);
  TORCH_CHECK(plan.reconditioned == oracle.reconditioned &&
                  plan.metadata.buckets_per_rank ==
                      oracle.metadata.buckets_per_rank,
              "reconditioning rank metadata differs");
  metadata = plan.metadata;
  p = prediction(2, false);
  prediction(1, true);
  options.hotstart.delay = 10;
  options.hotstart.unmatched_threshold = 1;
  plan = sam3::plan_video_update(2, false, none, at::ones_like(p.masks),
                                 p.logits, metadata, options);
  oracle = plan;
  reference_execute(2, oracle, none, er, options);
  sam3::execute_video_update_ranks(2, plan, none, ar, options);
  TORCH_CHECK(plan.metadata.object_ids().empty(),
              "controlled removal did not remove all objects");
  for (const auto &rank : actual)
    TORCH_CHECK(rank.empty(), "rank retained removed sessions");
  for (const auto &rank : expected)
    TORCH_CHECK(rank.empty(), "oracle retained removed sessions");
  std::cout << "{\"model\":\"" << (Mux ? "sam3.1" : "sam3") << "\",\"mode\":\""
            << mode << "\",\"ranks\":" << devices.size()
            << ",\"unique_devices\":" << cores.size()
            << ",\"objects\":17,\"neural_predictions_exact\":" << checks
            << ",\"births_reconditioning_global_memory_removal\":true,"
               "\"features\":\"synthetic\"}"
            << std::endl;
}
} // namespace
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 5, "usage: video_collective_probe STORE sam3|sam3.1 "
                           "DEVICE[,DEVICE...] MODE");
    c10::InferenceMode inference;
    at::set_num_threads(4);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    std::vector<at::Device> devices;
    std::istringstream input(argv[3]);
    std::string value;
    while (std::getline(input, value, ','))
      devices.push_back(
          at::empty({0}, at::TensorOptions().device(at::Device(value)))
              .device());
    TORCH_CHECK(!devices.empty(), "at least one device required");
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    if (std::string(argv[2]) == "sam3")
      run<false>(store, devices, argv[4]);
    else {
      TORCH_CHECK(std::string(argv[2]) == "sam3.1", "invalid model");
      run<true>(store, devices, argv[4]);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
