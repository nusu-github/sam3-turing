#pragma once
#include "sam3/video_update.h"

namespace sam3 {
// Synchronous, same-process transport. Different CUDA devices exchange through
// CPU memory, so peer access/NCCL is not required. Returned storage is
// independent.
SAM3_NATIVE_EXPORT at::Tensor transfer_video_tensor(const at::Tensor &,
                                                    at::Device);
struct VideoRankPrediction {
  std::vector<int64_t> ids;
  at::Tensor masks, logits; // floating [N,H,W], [N]
};
// Restore metadata order within each rank; reject cross-rank/missing/duplicate
// IDs. Like the source distributed path, multiple ranks promote to FP32 before
// collection; one rank retains its input dtype. Empty ranks remain explicit.
SAM3_NATIVE_EXPORT VideoRankPrediction
gather_video_tracking(const std::vector<VideoRankPrediction> &,
                      const std::vector<std::vector<int64_t>> &expected_ids,
                      at::Device destination);

struct Sam3VideoRank {
  at::Device device = at::kCPU;
  Sam3VideoSessions *sessions = nullptr;
  const Sam3SessionFactory *factory = nullptr;
};
struct Sam31VideoRank {
  at::Device device = at::kCPU;
  Sam31VideoSessions *sessions = nullptr;
  const Sam31SessionFactory *factory = nullptr;
};
// Borrowed rank collections must be exclusive for the duration of each call.
// Real sessions propagate on their configured devices; this API does not create
// replicas of a complete predictor or change object grouping.
SAM3_NATIVE_EXPORT VideoRankPrediction propagate_video_tracking_ranks(
    int64_t frame, bool reverse, const std::vector<Sam3VideoRank> &,
    const VideoMetadata &, at::Device destination, int64_t cleanup_area = 0);
SAM3_NATIVE_EXPORT VideoRankPrediction propagate_video_tracking_ranks(
    int64_t frame, bool reverse, const std::vector<Sam31VideoRank> &,
    const VideoMetadata &, at::Device destination, int64_t cleanup_area = 0);
// Execute the global plan on every rank, retaining GLOBAL masks for visibility
// decisions before selecting local memory rows. Merge bucket/affected-ID
// updates into the coordinator's plan. A failure stops later ranks; earlier
// successful session mutations are not rolled back (same contract as
// execute_video_update).
SAM3_NATIVE_EXPORT void execute_video_update_ranks(
    int64_t frame, VideoUpdatePlan &, const VideoDetections &,
    const std::vector<Sam3VideoRank> &, const VideoUpdateOptions &options = {});
SAM3_NATIVE_EXPORT void
execute_video_update_ranks(int64_t frame, VideoUpdatePlan &,
                           const VideoDetections &,
                           const std::vector<Sam31VideoRank> &,
                           const VideoUpdateOptions &options = {});
} // namespace sam3
