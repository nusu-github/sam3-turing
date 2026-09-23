#pragma once
#include "sam3/video_objects.h"
#include "sam3/video_recondition.h"
#include "sam3/video_memory.h"
namespace sam3 {
struct VideoMetadata {
  std::vector<std::vector<int64_t>> ids_per_rank;
  std::vector<int64_t> buckets_per_rank;
  int64_t max_id=-1;
  std::map<int64_t,double> object_scores;
  std::map<int64_t,std::map<int64_t,at::Tensor>> frame_scores;
  std::map<int64_t,at::Tensor> last_occluded;
  HostHotstartState host;
  DeviceHotstartState device;
  ConfirmationState confirmation;
  SAM3_NATIVE_EXPORT std::vector<int64_t> object_ids() const;
};
struct VideoUpdateOptions {
  AssociationOptions association;
  HotstartOptions hotstart;
  ReconditionOptions recondition;
  bool boundary_filter=false,confirmation_enabled=false,warmup_complete=true;
  bool allow_unoccluded_suppression=false,reapply_no_object_pointer=false;
  double boundary_margin=.025,occlusion_threshold=0.;
  int64_t confirmation_threshold=3,cleanup_area=16,bucket_capacity=16;
  // Counts use FP32 independently of the neural execution mode by default.
  std::string policy_mode="fp32";
};
struct VideoDetections {at::Tensor masks,scores,boxes,keep;};
struct VideoUpdatePlan {
  VideoMetadata metadata;
  AssociationMetadata association;
  std::vector<int64_t> previous_ids,new_ids,new_ranks;
  std::set<int64_t> removed,reconditioned;
  ReconditionMasks corrections;
  ReconditionDecision correction_decision;
  // suppressed is a GPU hotstart candidate. SAM3.1 source computes it but
  // does not publish it as a predictor output hide list; use host.suppressed.
  at::Tensor tracking_masks,suppressed;
};
SAM3_NATIVE_EXPORT VideoMetadata initialize_video_metadata(int64_t ranks,at::Device device=at::kCPU);
// Pure planning: inputs and previous metadata remain unchanged. Masks/scores are
// in previous.object_ids() order. No detection/object cap. Caller supplies global
// predictions; rank placement is planning, not GPU communication.
SAM3_NATIVE_EXPORT VideoUpdatePlan plan_video_update(int64_t frame,bool reverse,
    const VideoDetections&,const at::Tensor& tracking_masks,const at::Tensor& tracking_scores,
    const VideoMetadata&,const VideoUpdateOptions& options={});
// Source final score write occurs AFTER planning/removal and can overwrite the
// removed object's frame score, while its persistent object score stays -10000.
SAM3_NATIVE_EXPORT void finalize_video_scores(VideoMetadata&,int64_t frame,
    const std::vector<int64_t>& previous_ids,const at::Tensor& tracking_logits);
// Executes correction/preflight -> memory -> births -> removals for one rank.
// Updates actual bucket count. Not an all-session transaction.
SAM3_NATIVE_EXPORT void execute_video_update(int64_t frame,int64_t rank,VideoUpdatePlan&,
    const VideoDetections&,Sam3VideoSessions&,const Sam3SessionFactory&,const VideoUpdateOptions& options={});
SAM3_NATIVE_EXPORT void execute_video_update(int64_t frame,int64_t rank,VideoUpdatePlan&,
    const VideoDetections&,Sam31VideoSessions&,const Sam31SessionFactory&,const VideoUpdateOptions& options={});
// Raw source build_outputs mapping, before predictor-level temporal buffering,
// confirmation/output filtering and user-action postprocessing. Removed IDs are
// intentionally retained here; their score/metadata governs downstream output.
SAM3_NATIVE_EXPORT std::map<int64_t,at::Tensor> build_video_outputs(const VideoUpdatePlan&,
    const VideoDetections&,int64_t height,int64_t width,const VideoUpdateOptions& options={});
}
