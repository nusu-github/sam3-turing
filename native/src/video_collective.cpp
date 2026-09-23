#include "sam3/video_collective.h"
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <set>

namespace sam3 {
namespace {
void device_check(at::Device d) {
  TORCH_CHECK(d.is_cpu() || d.is_cuda(), "video transport supports CPU/CUDA");
}
at::Device resolved_device(at::Device d) {
  device_check(d);
  return at::empty({0}, at::TensorOptions().device(d)).device();
}
void ids_check(const std::vector<std::vector<int64_t>> &expected) {
  std::set<int64_t> seen;
  for (const auto &rank : expected)
    for (auto id : rank)
      TORCH_CHECK(seen.insert(id).second, "rank object IDs must be unique");
}
template <class Rank>
std::vector<std::vector<int64_t>>
validate_ranks(const std::vector<Rank> &ranks) {
  TORCH_CHECK(!ranks.empty(), "at least one tracking rank is required");
  std::set<const void *> collections, sessions;
  std::vector<std::vector<int64_t>> ids;
  for (const auto &rank : ranks) {
    device_check(rank.device);
    TORCH_CHECK(rank.sessions && collections.insert(rank.sessions).second,
                "rank collections must be non-null and distinct");
    ids.emplace_back();
    for (const auto &session : *rank.sessions) {
      TORCH_CHECK(session && sessions.insert(session.get()).second,
                  "sessions must be non-null and distinct");
      const auto local = session->object_ids();
      ids.back().insert(ids.back().end(), local.begin(), local.end());
    }
  }
  ids_check(ids);
  return ids;
}
void placement_check(const std::vector<std::vector<int64_t>> &actual,
                     const std::vector<std::vector<int64_t>> &expected) {
  TORCH_CHECK(actual.size() == expected.size(),
              "rank count does not match metadata");
  ids_check(expected);
  for (size_t r = 0; r < actual.size(); ++r)
    TORCH_CHECK(
        actual[r].size() == expected[r].size() &&
            std::set<int64_t>(actual[r].begin(), actual[r].end()) ==
                std::set<int64_t>(expected[r].begin(), expected[r].end()),
        "tracking IDs do not match rank ", r);
}
template <class Rank>
VideoRankPrediction
propagate(int64_t frame, bool reverse, const std::vector<Rank> &ranks,
          const VideoMetadata &metadata, at::Device target, int64_t cleanup) {
  c10::InferenceMode inference;
  placement_check(validate_ranks(ranks), metadata.ids_per_rank);
  std::vector<VideoRankPrediction> predictions;
  for (const auto &rank : ranks) {
    c10::DeviceGuard guard(resolved_device(rank.device));
    const auto options =
        at::TensorOptions().device(rank.device).dtype(at::kFloat);
    VideoRankPrediction p{
        {}, at::empty({0, 288, 288}, options), at::empty({0}, options)};
    std::vector<at::Tensor> masks, scores;
    TrackingPropagation request;
    request.start = frame;
    request.max_steps = 0;
    request.reverse = reverse;
    request.encode_memory = false;
    for (auto &session : *rank.sessions)
      session->propagate(request, [&](const auto &value) {
        p.ids.insert(p.ids.end(), value.object_ids.begin(),
                     value.object_ids.end());
        masks.push_back(value.low_masks.squeeze(1));
        scores.push_back(value.object_logits.flatten());
        return true;
      });
    if (!p.ids.empty()) {
      p.masks = clean_video_mask_scores(at::cat(masks).unsqueeze(1), cleanup)
                    .squeeze(1);
      p.logits = at::cat(scores);
    }
    const auto resolved = at::empty({0}, options).device();
    TORCH_CHECK(p.masks.device() == resolved && p.logits.device() == resolved,
                "tracking result is on the wrong rank device");
    predictions.push_back(std::move(p));
  }
  return gather_video_tracking(predictions, metadata.ids_per_rank, target);
}
template <class Rank>
void execute(int64_t frame, VideoUpdatePlan &plan,
             const VideoDetections &detection, const std::vector<Rank> &ranks,
             const VideoUpdateOptions &options) {
  c10::InferenceMode inference;
  const auto local_ids = validate_ranks(ranks);
  TORCH_CHECK(ranks.size() == plan.metadata.ids_per_rank.size() &&
                  ranks.size() == plan.metadata.buckets_per_rank.size(),
              "plan rank count mismatch");
  std::set<int64_t> actual;
  for (size_t r = 0; r < ranks.size(); ++r) {
    TORCH_CHECK(ranks[r].factory && *ranks[r].factory,
                "each rank needs a session factory");
    const auto &final = plan.metadata.ids_per_rank[r];
    for (auto id : local_ids[r]) {
      actual.insert(id);
      TORCH_CHECK(plan.removed.count(id) ||
                      std::find(final.begin(), final.end(), id) != final.end(),
                  "surviving object placed on wrong rank");
    }
  }
  TORCH_CHECK(actual == std::set<int64_t>(plan.previous_ids.begin(),
                                          plan.previous_ids.end()),
              "rank sessions do not cover the plan's previous IDs");
  for (size_t r = 0; r < ranks.size(); ++r) {
    const auto &rank = ranks[r];
    const auto device = resolved_device(rank.device);
    c10::DeviceGuard guard(device);
    if (ranks.size() == 1) {
      bool local = true;
      for (const auto &tensor :
           {plan.tracking_masks, plan.corrections.binary_masks,
            plan.corrections.low_masks, detection.masks, detection.scores,
            detection.boxes, detection.keep})
        if (tensor.defined() && tensor.device() != device)
          local = false;
      if (local) {
        execute_video_update(frame, r, plan, detection, *rank.sessions,
                             *rank.factory, options);
        continue;
      }
    }
    auto local = plan;
    // Execution only reads these tensor payloads. Host metadata is
    // value-copied; unused coordinator tensor histories stay untouched on their
    // original device.
    local.tracking_masks =
        transfer_video_tensor(plan.tracking_masks, rank.device);
    local.corrections.binary_masks =
        transfer_video_tensor(plan.corrections.binary_masks, rank.device);
    local.corrections.low_masks =
        transfer_video_tensor(plan.corrections.low_masks, rank.device);
    const VideoDetections d{
        transfer_video_tensor(detection.masks, rank.device),
        transfer_video_tensor(detection.scores, rank.device),
        transfer_video_tensor(detection.boxes, rank.device),
        transfer_video_tensor(detection.keep, rank.device)};
    execute_video_update(frame, r, local, d, *rank.sessions, *rank.factory,
                         options);
    plan.metadata.buckets_per_rank[r] = local.metadata.buckets_per_rank[r];
    plan.reconditioned.insert(local.reconditioned.begin(),
                              local.reconditioned.end());
  }
}
} // namespace
at::Tensor transfer_video_tensor(const at::Tensor &input, at::Device target) {
  c10::InferenceMode inference;
  target = resolved_device(target);
  if (!input.defined())
    return {};
  device_check(input.device());
  if (input.device() == target)
    return input.clone();
  // Explicit host staging avoids an implicit CUDA peer-access requirement.
  const auto host = input.is_cuda() ? input.to(at::kCPU) : input;
  return host.to(target, host.scalar_type(), false, true);
}
VideoRankPrediction
gather_video_tracking(const std::vector<VideoRankPrediction> &ranks,
                      const std::vector<std::vector<int64_t>> &expected,
                      at::Device target) {
  c10::InferenceMode inference;
  target = resolved_device(target);
  TORCH_CHECK(!ranks.empty() && ranks.size() == expected.size(),
              "prediction rank count mismatch");
  std::vector<std::vector<int64_t>> ids;
  for (const auto &r : ranks)
    ids.push_back(r.ids);
  ids_check(ids);
  placement_check(ids, expected);
  int64_t h = -1, w = -1;
  for (const auto &r : ranks) {
    TORCH_CHECK(r.masks.defined() && r.masks.dim() == 3 &&
                    r.masks.is_floating_point() &&
                    r.masks.size(0) == int64_t(r.ids.size()),
                "invalid rank masks");
    TORCH_CHECK(r.logits.defined() && r.logits.dim() == 1 &&
                    r.logits.is_floating_point() &&
                    r.logits.size(0) == int64_t(r.ids.size()) &&
                    r.logits.device() == r.masks.device(),
                "invalid rank logits");
    if (h < 0) {
      h = r.masks.size(1);
      w = r.masks.size(2);
    }
    TORCH_CHECK(h > 0 && w > 0 && r.masks.size(1) == h && r.masks.size(2) == w,
                "rank mask spatial dimensions disagree");
  }
  VideoRankPrediction result;
  std::vector<at::Tensor> masks, scores;
  for (size_t i = 0; i < ranks.size(); ++i) {
    const auto &r = ranks[i];
    const auto rows = video_memory_rows(r.ids, {expected[i]})[0];
    const auto order = at::tensor(rows, r.masks.options().dtype(at::kLong));
    auto low = r.masks.index_select(0, order),
         logits = r.logits.index_select(0, order);
    if (ranks.size() > 1) {
      low = low.to(at::kFloat);
      logits = logits.to(at::kFloat);
    }
    masks.push_back(low.device() == target
                        ? std::move(low)
                        : transfer_video_tensor(low, target));
    scores.push_back(logits.device() == target
                         ? std::move(logits)
                         : transfer_video_tensor(logits, target));
    result.ids.insert(result.ids.end(), expected[i].begin(), expected[i].end());
  }
  result.masks = (ranks.size() == 1 ? masks[0] : at::cat(masks)).contiguous();
  result.logits =
      (ranks.size() == 1 ? scores[0] : at::cat(scores)).contiguous();
  return result;
}
VideoRankPrediction propagate_video_tracking_ranks(
    int64_t f, bool reverse, const std::vector<Sam3VideoRank> &r,
    const VideoMetadata &m, at::Device d, int64_t cleanup) {
  return propagate(f, reverse, r, m, d, cleanup);
}
VideoRankPrediction propagate_video_tracking_ranks(
    int64_t f, bool reverse, const std::vector<Sam31VideoRank> &r,
    const VideoMetadata &m, at::Device d, int64_t cleanup) {
  return propagate(f, reverse, r, m, d, cleanup);
}
void execute_video_update_ranks(int64_t f, VideoUpdatePlan &p,
                                const VideoDetections &d,
                                const std::vector<Sam3VideoRank> &r,
                                const VideoUpdateOptions &o) {
  execute(f, p, d, r, o);
}
void execute_video_update_ranks(int64_t f, VideoUpdatePlan &p,
                                const VideoDetections &d,
                                const std::vector<Sam31VideoRank> &r,
                                const VideoUpdateOptions &o) {
  execute(f, p, d, r, o);
}
} // namespace sam3
