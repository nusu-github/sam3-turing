#include "sam3/multiplex_vision.h"
#include "sam3/preprocess.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
Sam31TrackingVision::Sam31TrackingVision(std::shared_ptr<const VisionEncoder> vision,std::shared_ptr<const Sam31TrackingFrame> core,at::Device device)
    :vision_(std::move(vision)),core_(std::move(core)),device_(device) {TORCH_CHECK(vision_ && core_,"visual backbone and tracking core are required");}
MultiplexTrackingFeatures Sam31TrackingVision::encode_rgb(const at::Tensor& pixels,const std::string& mode) const {
  return encode_preprocessed(preprocess_tracking_rgb(pixels),mode);
}
MultiplexTrackingFeatures Sam31TrackingVision::encode_preprocessed(const at::Tensor& image,const std::string& mode) const {
  c10::InferenceMode inference;
  TORCH_CHECK(image.sizes()==at::IntArrayRef({1,3,1008,1008}) && image.scalar_type()==at::kFloat,"tracking image requires normalized F32 [1,3,1008,1008]");
  auto out=vision_->forward(image.to(device_),mode,{"interactive_convs","propagation_convs"},{2});
  const auto& interactive=out.pyramid.at("interactive_convs");
  const auto& propagation=out.pyramid.at("propagation_convs");
  TORCH_CHECK(interactive.size()==3 && propagation.size()==3 && out.positions.size()==3,"expected SAM3.1 tracking necks with three levels");
  return {{interactive[2],out.positions[2],core_->project_interactive({interactive[0],interactive[1]},mode)},
          {propagation[2],out.positions[2],core_->project_propagation({propagation[0],propagation[1]},mode)}};
}
}
