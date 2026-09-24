#pragma once
#include "sam3/tracking_frame.h"
#include "sam3/vision_encoder.h"
#include <memory>
namespace sam3 {
// Reuses the visual backbone and tracking core; does not duplicate their weights.
class SAM3_NATIVE_EXPORT Sam3TrackingVision {
 public:
  Sam3TrackingVision(std::shared_ptr<const VisionEncoder>,std::shared_ptr<const Sam3TrackingFrame>,at::Device);
  TrackingFeatures encode_rgb(const at::Tensor&,const std::string& mode="fp32") const;
  // Already normalized [1,3,1008,1008], for decoder-specific preprocessing.
  TrackingFeatures encode_preprocessed(const at::Tensor&,const std::string& mode="fp32") const;
 private:
  std::shared_ptr<const VisionEncoder> vision_;
  std::shared_ptr<const Sam3TrackingFrame> core_;
  at::Device device_;
};
}
