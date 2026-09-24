#include "sam3/tracking_vision.h"
#include "sam3/preprocess.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
Sam3TrackingVision::Sam3TrackingVision(std::shared_ptr<const VisionEncoder> vision,std::shared_ptr<const Sam3TrackingFrame> core,at::Device device)
    :vision_(std::move(vision)),core_(std::move(core)),device_(device) {TORCH_CHECK(vision_ && core_,"visual backbone and tracking core are required");}
TrackingFeatures Sam3TrackingVision::encode_rgb(const at::Tensor& pixels,const std::string& mode) const {
  return encode_preprocessed(preprocess_tracking_rgb(pixels),mode);
}
TrackingFeatures Sam3TrackingVision::encode_preprocessed(const at::Tensor& image,const std::string& mode) const {
  c10::InferenceMode inference;
  TORCH_CHECK(image.sizes()==at::IntArrayRef({1,3,1008,1008}) && image.scalar_type()==at::kFloat,"tracking image requires normalized F32 [1,3,1008,1008]");
  auto out=vision_->forward(image.to(device_),mode,{"sam2_convs"},{2});
  const auto& pyramid=out.pyramid.at("sam2_convs");
  TORCH_CHECK(pyramid.size()==4 && out.positions.size()==4,"expected SAM3 tracking neck before scalp=1");
  return {pyramid[2],out.positions[2],core_->project_pyramid({pyramid[0],pyramid[1]},mode)};
}
}
