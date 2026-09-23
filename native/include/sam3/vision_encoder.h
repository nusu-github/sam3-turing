#pragma once
#include "sam3/weights.h"

namespace sam3 {
struct VisionFeatures {
  at::Tensor trunk;
  std::map<std::string, std::vector<at::Tensor>> pyramid;
  std::vector<at::Tensor> positions;
};

class SAM3_NATIVE_EXPORT VisionEncoder {
 public:
  VisionEncoder(const WeightStore& store, const std::string& model, at::Device device = at::kCPU);
  // Preprocessed RGB [B,3,1008,1008], matching the original model resolution.
  // Modes: fp32, fp16 (Turing path), bf16_reference (original fused MLP).
  // All available necks are returned by default; selected heads reuse one trunk.
  VisionFeatures forward(const at::Tensor& image, const std::string& mode = "fp32",
                         const std::vector<std::string>& heads = {}) const;
  // Select only consumed position maps. Result positions retain their level
  // indices; unrequested entries are undefined. Empty selects none. Feature
  // pyramids/trunk and the original overload are unchanged.
  VisionFeatures forward(const at::Tensor& image, const std::string& mode,
                         const std::vector<std::string>& heads,
                         const std::vector<int64_t>& position_levels) const;
 private:
  const at::Tensor& weight(const std::string& name) const;
  at::Tensor norm(const at::Tensor& x, const std::string& prefix) const;
  at::Tensor linear(const at::Tensor& x, const std::string& prefix) const;
  at::Tensor attention(const at::Tensor& x, const std::string& prefix) const;
  at::Tensor block(const at::Tensor& x, int64_t layer, bool fused_bf16) const;
  at::Tensor neck(const at::Tensor& x, const std::string& head, int64_t level) const;
  at::Tensor position(const at::Tensor& x) const;
  std::map<std::string, at::Tensor> weights_;
  std::vector<std::string> heads_;
  int64_t levels_;
  at::Device device_;
};
}
