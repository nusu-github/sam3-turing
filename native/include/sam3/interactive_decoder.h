#pragma once
#include "sam3/interactive_prompt.h"
namespace sam3 {
struct InteractiveMaskOutput {
  at::Tensor masks;         // [B,1|3,4H,4W], logits
  at::Tensor iou;           // [B,1|3]
  at::Tensor tokens;        // [B,1|3,256], source object-pointer tokens
  at::Tensor object_logits; // [B,1]
  at::Tensor all_masks;     // [B,4,4H,4W]
  at::Tensor all_iou;       // [B,4]
  at::Tensor all_tokens;    // [B,4,256]
};
class SAM3_NATIVE_EXPORT InteractiveMaskDecoder {
 public:
  InteractiveMaskDecoder(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  // Project the two high-resolution 256-channel visual maps once per image.
  std::vector<at::Tensor> project_pyramid(const std::vector<at::Tensor>&,
      const std::string& mode="fp32") const;
  InteractiveMaskOutput forward(const at::Tensor& image,const InteractiveEmbeddings&,
      const std::vector<at::Tensor>& high_resolution,bool multimask,bool repeat_image=false,
      const std::string& mode="fp32",bool dynamic_stability=true,
      double stability_delta=.05,double stability_threshold=.98) const;
 private:
  at::Tensor attention(const at::Tensor&,const at::Tensor&,const at::Tensor&,const std::string&) const;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
  bool sigmoid_iou_;
};
}
