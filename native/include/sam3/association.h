#pragma once
#include <ATen/ATen.h>
#include <map>
#include <string>
#include <vector>
#include "sam3_native_export.h"
namespace sam3 {
enum class AssociationPolicy { Sam3, Sam31 };
struct AssociationOptions {
  AssociationPolicy policy=AssociationPolicy::Sam3;
  double new_detection_threshold=.7,track_match_threshold=.5,detection_match_threshold=.1;
  double high_confidence_threshold=.8,iom_recondition_threshold=.8,iou_recondition_threshold=.8;
  bool use_iom=false;
  // Optional SAM3.1 source compile padding, never a limit/drop count. Zero disables.
  int64_t pad_tracks_to=0;
};
struct AssociationTensors {
  at::Tensor unmatched,nonempty,is_new,best_track,high_confidence,high_overlap,keep,matches;
  std::vector<at::Tensor> tensors() const {return {unmatched,nonempty,is_new,best_track,high_confidence,high_overlap,keep,matches};}
};
struct AssociationMetadata {
  std::vector<int64_t> new_detections,unmatched_tracks,empty_tracks;
  std::map<int64_t,std::vector<int64_t>> detection_to_tracks;
  // Source iteration overwrites earlier detections if a track occurs again.
  std::map<int64_t,int64_t> track_to_recondition_detection;
  // Python dict insertion order matters to the SAM3.1 first-pair IoU gate.
  std::vector<int64_t> recondition_order;
};
// Floating logits [N,H,W], scores[N], tracks[M,Ht,Wt]. Resize to smaller area
// before sign thresholding. SAM3 keep must be absent/all true (source prefilters).
// SAM3.1 accepts bool keep[N]. No cap on N or M. Empty-input policy is model-specific.
SAM3_NATIVE_EXPORT AssociationTensors associate_tracking(const at::Tensor& detections,
    const at::Tensor& scores,const at::Tensor& tracks,const at::Tensor& keep={},
    const AssociationOptions& options={},const std::string& mode="fp32");
SAM3_NATIVE_EXPORT AssociationMetadata realize_association(const AssociationTensors&,
    const std::vector<int64_t>& track_ids,AssociationPolicy policy);
// This plans placements only; it does not run multi-GPU inference/communication.
// capacity=1 is SAM3; SAM3.1 multiplex assigns entire capacity-sized groups.
SAM3_NATIVE_EXPORT std::vector<int64_t> assign_detection_devices(int64_t count,
    const std::vector<int64_t>& previous_workload,int64_t capacity=1);
SAM3_NATIVE_EXPORT at::Tensor detection_boundary_keep(const at::Tensor& boxes,double margin=.025);
}
