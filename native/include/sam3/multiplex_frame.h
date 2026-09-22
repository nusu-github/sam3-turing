#pragma once
#include "sam3/tracking_frame.h"
#include "sam3/multiplex_temporal.h"
#include "sam3/multiplex_decoder.h"
namespace sam3 {
struct MultiplexFrame {
  int64_t index=0;
  VideoMaskOutput masks;
  at::Tensor memory,memory_position,pointer,iou,confidence,image,image_position,input_masks;
  std::vector<int64_t> conditioning_objects;
};
struct MultiplexFrameHistory {std::vector<MultiplexFrame> conditioning,tracked;};
struct MultiplexFrameRequest {
  int64_t index=0,frame_count=1;
  bool initial=false,reverse=false,encode_memory=true;
  at::Tensor points,labels,mask,previous_logits;
  std::optional<std::vector<int64_t>> objects_to_interact;
};
struct MultiplexFrameOptions {
  MultiplexTemporalOptions temporal;
  bool offload_output=false,trim_history=false,save_image=true,non_overlap_memory=false;
  bool multimask=true,multimask_tracking=true,attenuate_iou_by_stability=false;
  int64_t multimask_min_points=0,multimask_max_points=1;
  double object_threshold=0.;
};
struct MultiplexMaskUpdate {
  bool append=false,encode_memory=true,masks_from_points=false;
  bool allow_new_buckets=false,prefer_new_buckets=false;
};
// SAM3.1 inference frame host with current-frame mask insertion/reconditioning.
// Session-level history remapping and singleton reintegration are separate.
class SAM3_NATIVE_EXPORT Sam31TrackingFrame {
 public:
  Sam31TrackingFrame(const WeightStore&,at::Device device=at::kCPU);
  std::vector<at::Tensor> project_interactive(const std::vector<at::Tensor>&,const std::string& mode="fp32") const;
  std::vector<at::Tensor> project_propagation(const std::vector<at::Tensor>&,const std::string& mode="fp32") const;
  MultiplexFrame forward(const TrackingFeatures& interactive,const TrackingFeatures& propagation,
      const MultiplexFrameRequest&,MultiplexFrameHistory&,const MultiplexState&,
      const MultiplexFrameOptions& options={},const std::string& mode="fp32") const;
  // Update a current frame after propagation. Previous history is not remapped.
  // Appends allocate consecutive indices, as in the source; supplied indices
  // identify rows only for reconditioning. Returns the actual affected indices.
  std::vector<int64_t> update_masks(const TrackingFeatures& interactive,const TrackingFeatures& propagation,
      const at::Tensor& masks,const std::vector<int64_t>& indices,
      const std::optional<std::vector<int64_t>>& object_ids,MultiplexFrame&,MultiplexState&,
      const MultiplexMaskUpdate&,const MultiplexFrameOptions& options={},const std::string& mode="fp32") const;
  // Rebuild dense history from retained full-resolution masks and shared image.
  MaskMemoryOutput encode_history(const MultiplexFrame&,const MultiplexState&,
      const MultiplexFrameOptions& options={},const std::string& mode="fp32") const;
 private:
  VideoInteractiveHeads interactive_;
  MultiplexPropagationHeads propagation_;
  MultiplexMemoryConditioner temporal_;
  MaskMemoryEncoder memory_;
  at::Tensor no_memory_;
};
}
