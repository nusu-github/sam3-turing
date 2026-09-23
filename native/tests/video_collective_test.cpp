#include "sam3/video_collective.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <limits>
template <class F> void rejected(F f) {
  bool caught = false;
  try {
    f();
  } catch (const c10::Error &) {
    caught = true;
  }
  TORCH_CHECK(caught, "invalid collective input was accepted");
}
int main(int argc, char **argv) {
  try {
    c10::InferenceMode inference;
    at::set_num_threads(2);
    const at::Device device(argc > 1 ? argv[1] : "cpu");
    int cases = 0;
    for (auto dtype : {at::kHalf, at::kBFloat16, at::kFloat, at::kDouble}) {
      const auto o = at::TensorOptions().device(device).dtype(dtype);
      auto masks = at::arange(30, o).reshape({2, 5, 3}).transpose(1, 2);
      auto scores = at::tensor({.125, .875}, o.dtype(at::kFloat)).to(dtype);
      sam3::VideoRankPrediction a{{9, -7}, masks, scores};
      auto one = sam3::gather_video_tracking({a}, {{-7, 9}}, device);
      TORCH_CHECK(one.ids == std::vector<int64_t>({-7, 9}) &&
                      one.masks.scalar_type() == dtype &&
                      one.logits.scalar_type() == dtype,
                  "single rank dtype/order changed");
      TORCH_CHECK(at::equal(one.masks[0], masks[1]) &&
                      at::equal(one.logits[1], scores[0]),
                  "single rank values changed");
      one.masks.zero_();
      TORCH_CHECK(masks.sum().item<double>() > 0,
                  "gather aliases caller storage");
      auto empty = sam3::VideoRankPrediction{
          {}, at::empty({0, 3, 5}, o), at::empty({0}, o)};
      const int64_t large = std::numeric_limits<int64_t>::max() - 1;
      auto cpu = sam3::VideoRankPrediction{
          {large},
          at::full({1, 3, 5}, -.3125, at::TensorOptions().dtype(dtype)),
          at::ones({1}, at::TensorOptions().dtype(dtype))};
      for (auto target : {at::Device(at::kCPU), device}) {
        auto out = sam3::gather_video_tracking({a, empty, cpu},
                                               {{-7, 9}, {}, {large}}, target);
        TORCH_CHECK(out.ids == std::vector<int64_t>({-7, 9, large}) &&
                        out.masks.scalar_type() == at::kFloat &&
                        out.logits.scalar_type() == at::kFloat,
                    "multi-rank FP32/order contract changed");
        auto expected = at::cat({masks[1].unsqueeze(0).cpu(),
                                 masks[0].unsqueeze(0).cpu(), cpu.masks})
                            .to(at::kFloat);
        TORCH_CHECK(at::equal(out.masks.cpu(), expected) &&
                        out.masks.is_contiguous(),
                    "mixed-device gather changed values");
        auto all_empty =
            sam3::gather_video_tracking({empty, empty}, {{}, {}}, target);
        TORCH_CHECK(all_empty.masks.sizes() == at::IntArrayRef({0, 3, 5}) &&
                        all_empty.masks.scalar_type() == at::kFloat,
                    "empty ranks lost shape/dtype");
        ++cases;
      }
      rejected([&] {
        sam3::gather_video_tracking({a, cpu}, {{large}, {-7, 9}}, device);
      });
      rejected([&] {
        sam3::gather_video_tracking({a, a}, {{9, -7}, {9, -7}}, device);
      });
      rejected([&] {
        auto bad = a;
        bad.logits = at::empty({1}, o);
        sam3::gather_video_tracking({bad}, {{9, -7}}, device);
      });
      rejected([&] {
        auto bad = empty;
        bad.masks = at::empty({0, 4, 5}, o);
        sam3::gather_video_tracking({a, bad}, {{9, -7}, {}}, device);
      });
      auto transported = sam3::transfer_video_tensor(masks, at::kCPU);
      TORCH_CHECK(at::equal(transported, masks.cpu()), "host transfer differs");
      transported.zero_();
      TORCH_CHECK(masks.sum().item<double>() > 0, "transport aliases source");
      auto returned = sam3::transfer_video_tensor(masks.cpu(), device);
      TORCH_CHECK(at::equal(returned.cpu(), masks.cpu()),
                  "device transfer differs");
    }
    sam3::Sam3VideoSessions left, right;
    sam3::Sam3SessionFactory factory =
        []() -> std::unique_ptr<sam3::Sam3TrackingSession> {
      TORCH_CHECK(false, "empty update must not create a session");
    };
    std::vector<sam3::Sam3VideoRank> ranks{{device, &left, &factory},
                                           {device, &right, &factory}};
    auto metadata = sam3::initialize_video_metadata(2, device);
    auto predictions =
        sam3::propagate_video_tracking_ranks(0, false, ranks, metadata, device);
    const auto o = at::TensorOptions().device(device).dtype(at::kFloat);
    sam3::VideoDetections d{at::empty({0, 288, 288}, o), at::empty({0}, o),
                            at::empty({0, 4}, o),
                            at::empty({0}, o.dtype(at::kBool))};
    auto plan = sam3::plan_video_update(0, false, d, predictions.masks,
                                        predictions.logits, metadata);
    sam3::execute_video_update_ranks(0, plan, d, ranks);
    rejected([&] {
      auto bad = ranks;
      bad[1].sessions = &left;
      sam3::propagate_video_tracking_ranks(0, false, bad, metadata, device);
    });
    rejected([&] {
      auto bad = ranks;
      bad[1].factory = nullptr;
      sam3::execute_video_update_ranks(0, plan, d, bad);
    });
    TORCH_CHECK(left.empty() && right.empty() &&
                    plan.metadata.object_ids().empty(),
                "empty update mutated rank collections");
    std::cout
        << "collective gather/transport/placement/empty execution passed: "
        << cases << " dtype/device combinations on " << device << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
