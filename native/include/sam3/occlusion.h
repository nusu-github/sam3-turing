#pragma once
#include "sam3/hotstart.h"
namespace sam3 {
struct OcclusionOptions {
  AssociationPolicy policy=AssociationPolicy::Sam3;
  double iou_threshold=.5;
  bool allow_unoccluded_to_suppress=false; // source SAM3.1 option
};
struct OcclusionResult {at::Tensor masks,suppressed,last_occluded;};
// Low-level policy always runs; the high-level host calls it only when enabled.
// SAM3 known=false means absent dictionary key (distinct from a stored -1).
// SAM3.1 removal overrides prior history, SAM3 removal overrides only absent keys.
// The source's finite 100000 removal sentinel and reverse comparisons are kept.
SAM3_NATIVE_EXPORT OcclusionResult update_occlusion(const at::Tensor& masks,
    const at::Tensor& last_occluded,const at::Tensor& removed,int64_t frame,
    bool reverse=false,const OcclusionOptions& options={},const at::Tensor& known={},
    const std::string& mode="fp32");
struct DeviceOcclusionResult {DeviceHotstartState state;at::Tensor masks,suppressed;};
SAM3_NATIVE_EXPORT DeviceOcclusionResult update_device_occlusion(const DeviceHotstartState&,
    const at::Tensor& masks,const at::Tensor& removed,int64_t frame,bool reverse,
    double threshold,bool allow_unoccluded=false,const std::string& mode="fp32");
struct HostOcclusionResult {std::map<int64_t,at::Tensor> history;at::Tensor masks,suppressed;};
SAM3_NATIVE_EXPORT HostOcclusionResult update_host_occlusion(const std::map<int64_t,at::Tensor>& history,
    const std::vector<int64_t>& ids,const at::Tensor& masks,const std::set<int64_t>& removed,
    int64_t frame,bool reverse,double threshold,const std::string& mode="fp32");

// Source mask_to_box semantics: bool [N,1,H,W] -> int32 [N,1,4], inclusive
// maxima, zero boxes for empty masks. Axis reductions avoid full coordinate grids.
SAM3_NATIVE_EXPORT at::Tensor tracking_mask_boxes(const at::Tensor& masks);
SAM3_NATIVE_EXPORT at::Tensor diagonal_box_iou(const at::Tensor& a,const at::Tensor& b);
// Source video-specific +0.1/-0.1 cleanup, including half-foreground-area rule.
// Unlike the image transform, sprinkle removal sees the already filled holes.
SAM3_NATIVE_EXPORT at::Tensor clean_video_mask_scores(const at::Tensor&,int64_t area,
    bool holes=true,bool sprinkles=true);
using ReconditionCandidates=std::vector<std::pair<int64_t,int64_t>>; // ordered (object ID,detection index)
SAM3_NATIVE_EXPORT ReconditionCandidates recondition_candidates(const AssociationMetadata&);
struct ReconditionOptions {
  AssociationPolicy policy=AssociationPolicy::Sam3;
  int64_t period=-1;
  double box_iou_threshold=0.,detection_score_threshold=0.;
};
struct ReconditionDecision {bool periodic=false,geometry=false;std::set<int64_t> geometry_ids;};
// Either trigger requests reconditioning of ALL candidates, not just geometry_ids.
// Source SAM3.1 gate uses first-pair IoU and any candidate's qualifying score.
SAM3_NATIVE_EXPORT ReconditionDecision recondition_decision(const ReconditionCandidates&,
    const std::vector<int64_t>& track_ids,const at::Tensor& detection_boxes,const at::Tensor& detection_scores,
    const at::Tensor& track_masks,int64_t frame,const ReconditionOptions& options={},const std::string& mode="fp32");
struct ReconditionMasks {
  std::vector<int64_t> ids,detection_indices,track_indices;
  at::Tensor binary_masks,low_masks;
};
// Prepares edits without mutating the inputs or running neural session updates.
// SAM3 tests raw track score > .8; SAM3.1 tests sigmoid(score) > .8 and merges
// low logits by sign agreement before source video cleanup. area is its source
// fill_hole_area; the source currently ignores separate sprinkle_removal_area.
SAM3_NATIVE_EXPORT ReconditionMasks prepare_recondition_masks(const ReconditionCandidates&,
    const std::vector<int64_t>& track_ids,const at::Tensor& detection_masks,const at::Tensor& track_scores,
    const at::Tensor& track_masks,int64_t input_height,int64_t input_width,
    AssociationPolicy policy,int64_t area=16,const std::string& mode="fp32");
struct ReconditionBatch {int64_t state_index;std::vector<int64_t> ids;at::Tensor masks;};
// SAM3 preserves per-candidate/per-state call order. SAM3.1 assigns each target
// to its first containing state, grouping batches in first-encounter order.
SAM3_NATIVE_EXPORT std::vector<ReconditionBatch> recondition_batches(const ReconditionMasks&,
    const std::vector<std::vector<int64_t>>& state_ids,AssociationPolicy policy,bool multiplex=true);
}
