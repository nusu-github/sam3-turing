#pragma once
#include "sam3/grounding.h"
#include "sam3/multiplex_vision.h"
#include "sam3/video_update.h"
namespace sam3 {
struct VideoFrameFeatures {
  std::vector<at::Tensor> detection_pyramid;
  at::Tensor detection_position;
  MultiplexTrackingFeatures tracking; // SAM3 uses the same neck for both entries.
};
// One trunk evaluation supplies detector and all required tracker necks. Modules
// are shared with sessions; no duplicate checkpoint variants or vision weights.
class SAM3_NATIVE_EXPORT VideoFrameEncoder {
 public:
  VideoFrameEncoder(std::shared_ptr<const VisionEncoder>,std::shared_ptr<const GroundingDetector>,
      std::shared_ptr<const Sam3TrackingFrame>,at::Device);
  VideoFrameEncoder(std::shared_ptr<const VisionEncoder>,std::shared_ptr<const GroundingDetector>,
      std::shared_ptr<const Sam31TrackingFrame>,at::Device);
  VideoFrameFeatures encode_rgb(const at::Tensor&,const std::string& mode="fp32") const;
  VideoFrameFeatures encode_preprocessed(const at::Tensor&,const std::string& mode="fp32") const;
  GroundingOutput detect(const VideoFrameFeatures&,const GroundingPrompt&,
      const std::string& mode="fp32") const;
 private:
  std::shared_ptr<const VisionEncoder> vision_;
  std::shared_ptr<const GroundingDetector> detector_;
  std::shared_ptr<const Sam3TrackingFrame> sam3_;
  std::shared_ptr<const Sam31TrackingFrame> sam31_;
  at::Device device_;
};
enum class VideoNmsMode { Greedy, Sam31Perflib, Sam31Batched };
struct VideoDetectionOptions {
  AssociationPolicy model=AssociationPolicy::Sam3;
  VideoNmsMode nms=VideoNmsMode::Greedy;
  double score_threshold=.5,nms_threshold=.1;
  bool use_iom=false,boundary_filter=false,allow_new_detections=true;
  double boundary_margin=.025;
  std::string policy_mode="fp32";
};
// Returns source NMS keep flags for one prompt, without mutating logits.
SAM3_NATIVE_EXPORT at::Tensor video_mask_nms(const at::Tensor& probabilities,
    const at::Tensor& masks,const VideoDetectionOptions& options={});
// SAM3 compacts selected queries; SAM3.1 keeps every query with a boolean keep
// mask, ordered with source argsort(keep, descending=True). All prompt batches
// are returned independently; no query/prompt cap beyond the model itself.
SAM3_NATIVE_EXPORT std::vector<VideoDetections> postprocess_video_detections(
    const DetectionOutput&,const VideoDetectionOptions& options={});
}
