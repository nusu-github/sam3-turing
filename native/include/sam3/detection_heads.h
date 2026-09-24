#pragma once
#include "sam3/detector.h"
namespace sam3 {
struct DetectionOutput {
  at::Tensor logits;         // [B,200,1]
  at::Tensor boxes;          // [B,200,4], normalized cxcywh
  at::Tensor boxes_xyxy;     // [B,200,4], normalized xyxy
  at::Tensor masks;          // [B,200,Hmask,Wmask], raw logits
  at::Tensor semantic;       // [B,1,Hmask,Wmask], raw logits
  at::Tensor queries;        // [B,200,256]
  at::Tensor presence_logits;// [B,1]
  at::Tensor presence;       // [1,B,256]
};
class SAM3_NATIVE_EXPORT DetectionHeads {
 public:
  DetectionHeads(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  // Pyramid is the original post-scalp detection pyramid. image_ids maps each
  // prompt batch item to a source image; repeated/reordered IDs are supported.
  // joint_scores follows the source video detector option (image default false).
  DetectionOutput forward(const std::vector<at::Tensor>& pyramid,const at::Tensor& image_ids,
      const FusionFeatures&,const at::Tensor& prompt_padding,const DecoderFeatures&,
      bool joint_scores=false,const std::string& mode="fp32") const;
 private:
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
}
