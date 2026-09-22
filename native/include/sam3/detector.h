#pragma once
#include "sam3/weights.h"
namespace sam3 {
struct FusionFeatures {
  at::Tensor memory;          // [H*W,B,256]
  at::Tensor padding;         // [H*W,B], undefined when no image padding
  at::Tensor positions;      // [H*W,B,256]
  at::Tensor prompt;         // [L,B,256], unchanged by this encoder
  at::Tensor level_start;    // [1], int64
  at::Tensor spatial_shapes; // [1,2], int64 (H,W)
  at::Tensor valid_ratios;    // [B,1,2] (valid W/H fractions)
};
class SAM3_NATIVE_EXPORT DetectorEncoder {
 public:
  DetectorEncoder(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  // One feature scale is the original checkpoint architecture. Prompt lengths
  // and per-image padding are unrestricted. No source tensor is modified.
  FusionFeatures forward(const at::Tensor& image,const at::Tensor& positions,
      const at::Tensor& prompt,const at::Tensor& prompt_padding,
      const at::Tensor& image_padding={},const std::string& mode="fp32") const;
 private:
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};

struct DecoderFeatures {
  at::Tensor hidden;         // [6,200,B,256], all normalized layer outputs
  at::Tensor references;     // [6,200,B,4], anchors before each refinement
  at::Tensor presence_logits;// [6,1,B]
  at::Tensor presence;       // [1,B,256], final presence features
};
class SAM3_NATIVE_EXPORT DetectorDecoder {
 public:
  DetectorDecoder(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  // Optional diagnostic trace retains layer tensors; omitted for normal inference.
  DecoderFeatures forward(const FusionFeatures&,const at::Tensor& prompt_padding,
                          const std::string& mode="fp32",std::map<std::string,at::Tensor>* trace=nullptr) const;
 private:
  at::Tensor relative_position_bias(const at::Tensor& boxes,int64_t h,int64_t w) const;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
}
