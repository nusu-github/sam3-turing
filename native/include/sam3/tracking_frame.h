#pragma once
#include "sam3/temporal_memory.h"
#include "sam3/video_heads.h"
#include "sam3/memory_encoder.h"
namespace sam3 {
struct TrackingFeatures {
  at::Tensor image,position; // BCHW, [B,256,72,72]; image shared by expansion upstream
  std::vector<at::Tensor> high; // projected [B,32,288,288], [B,64,144,144]
};
struct TrackingFrame {
  int64_t index=0;
  at::Tensor low_mask,high_mask,pointer,object_logits,iou,confidence,memory,memory_position;
};
struct TrackingHistory { std::vector<TrackingFrame> conditioning,tracked; };
struct TrackingFrameRequest {
  int64_t index=0,frame_count=1;
  bool initial=false,reverse=false,use_previous=true,encode_memory=true;
  at::Tensor points,labels,mask,previous_logits;
};
struct TrackingFrameOptions {
  TemporalOptions temporal;
  bool non_overlap_memory=false,offload_output=false,trim_history=false;
  bool multimask=true,multimask_tracking=true;
  int64_t multimask_min_points=0,multimask_max_points=1;
};
// SAM3's inference frame host. Session-level prompt accumulation, consolidation
// and original-resolution output are separate from this neural/state operation.
class SAM3_NATIVE_EXPORT Sam3TrackingFrame {
 public:
  Sam3TrackingFrame(const WeightStore&,at::Device device=at::kCPU);
  std::vector<at::Tensor> project_pyramid(const std::vector<at::Tensor>&,
      const std::string& mode="fp32") const;
  TrackingFrame forward(const TrackingFeatures&,const TrackingFrameRequest&,TrackingHistory&,
      const TrackingFrameOptions& options={},const std::string& mode="fp32") const;
  // Used by session consolidation after masks from all objects are finalized.
  MaskMemoryOutput encode_memory(const TrackingFeatures&,const at::Tensor& high_masks,
      const at::Tensor& object_logits,bool from_points,bool non_overlap=false,
      const std::string& mode="fp32") const;
 private:
  VideoInteractiveHeads heads_;
  Sam3MemoryConditioner temporal_;
  MaskMemoryEncoder memory_;
};
}
