#pragma once
#include "sam3/multiplex_session.h"
#include "sam3/vision_encoder.h"
namespace sam3 {
// One shared visual trunk produces both tracking necks; no per-task weights.
class SAM3_NATIVE_EXPORT Sam31TrackingVision {
 public:
  Sam31TrackingVision(std::shared_ptr<const VisionEncoder>,std::shared_ptr<const Sam31TrackingFrame>,at::Device);
  MultiplexTrackingFeatures encode_rgb(const at::Tensor&,const std::string& mode="fp32") const;
  MultiplexTrackingFeatures encode_preprocessed(const at::Tensor&,const std::string& mode="fp32") const;
 private:
  std::shared_ptr<const VisionEncoder> vision_;
  std::shared_ptr<const Sam31TrackingFrame> core_;
  at::Device device_;
};
}
