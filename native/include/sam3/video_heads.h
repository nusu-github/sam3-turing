#pragma once
#include "sam3/interactive_decoder.h"
namespace sam3 {
struct VideoMaskOutput {
  at::Tensor low_res_multimasks,high_res_multimasks,iou;
  at::Tensor low_res_mask,high_res_mask,object_pointer,object_logits;
};
// SAM3 per-object heads and SAM3.1 interactive heads. Multiplex propagation
// and temporal memory scheduling are separate modules, not implemented here.
class SAM3_NATIVE_EXPORT VideoInteractiveHeads {
 public:
  VideoInteractiveHeads(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  std::vector<at::Tensor> project_pyramid(const std::vector<at::Tensor>&,
      const std::string& mode="fp32") const;
  // Coordinates are in the 1008x1008 model image. SAM3 uses one image per
  // prompt batch entry; SAM3.1 repeats one image for multiple objects.
  VideoMaskOutput forward(const at::Tensor& image,const std::vector<at::Tensor>& projected_high,
      const at::Tensor& points={},const at::Tensor& labels={},const at::Tensor& masks={},
      bool multimask=false,const std::string& mode="fp32",double object_threshold=0.) const;
  VideoMaskOutput use_mask_as_output(const at::Tensor& image,const std::vector<at::Tensor>& projected_high,
      const at::Tensor& mask,const std::string& mode="fp32",double object_threshold=0.) const;
 private:
  at::Tensor gate_pointer(const at::Tensor& pointer,const at::Tensor& present) const;
  InteractivePromptEncoder encoder_;
  InteractiveMaskDecoder decoder_;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
  bool multiplex_;
};
}
