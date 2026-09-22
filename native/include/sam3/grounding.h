#pragma once
#include "sam3/geometry_encoder.h"
#include "sam3/detection_heads.h"
namespace sam3 {
struct GroundingPrompt {
  at::Tensor image_ids;       // [B] source image indices
  at::Tensor text_ids;        // [B] source text indices
  at::Tensor text_features;   // [L,Ntext,256]
  at::Tensor text_padding;    // [Ntext,L]
  GeometryPrompt geometry;
  at::Tensor visual_features; // optional [Nvisual,B,256]
  at::Tensor visual_padding;  // optional [B,Nvisual]
  at::Tensor previous_mask;   // optional [H*W,B,256], added only for geometry
  bool use_text=true;
};
struct GroundingOutput {
  DetectionOutput detection;
  FusionFeatures encoded;
  DecoderFeatures decoded;
};
// Reusable detector modules. Vision/text modules are independent and may be
// loaded/unloaded by the host without duplicating checkpoint files.
class SAM3_NATIVE_EXPORT GroundingDetector {
 public:
  GroundingDetector(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  GroundingOutput forward(const std::vector<at::Tensor>& pyramid,const at::Tensor& positions,
      const GroundingPrompt&,bool joint_scores=false,const std::string& mode="fp32") const;
 private:
  GeometryEncoder geometry_;
  DetectorEncoder encoder_;
  DetectorDecoder decoder_;
  DetectionHeads heads_;
  at::Device device_;
};
}
