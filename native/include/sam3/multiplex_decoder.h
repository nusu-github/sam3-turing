#pragma once
#include "sam3/multiplex.h"
#include "sam3/video_heads.h"
namespace sam3 {
struct MultiplexMaskOutput {
  at::Tensor masks,iou,tokens,object_logits;
};
// SAM3.1's shipped propagation decoder: 16 slots, three distinct mask tokens
// per slot, separate IoU/object tokens, and no single-mask token.
class SAM3_NATIVE_EXPORT MultiplexMaskDecoder {
 public:
  MultiplexMaskDecoder(const WeightStore&,at::Device device=at::kCPU);
  std::vector<at::Tensor> project_pyramid(const std::vector<at::Tensor>&,
      const std::string& mode="fp32") const;
  MultiplexMaskOutput forward(const at::Tensor& image,const at::Tensor& position,
      const std::vector<at::Tensor>& projected_high,const at::Tensor& extra_embeddings={},
      const std::string& mode="fp32") const;
 private:
  at::Tensor attention(const at::Tensor&,const at::Tensor&,const at::Tensor&,const std::string&) const;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
class SAM3_NATIVE_EXPORT MultiplexPropagationHeads {
 public:
  MultiplexPropagationHeads(const WeightStore&,at::Device device=at::kCPU);
  std::vector<at::Tensor> project_pyramid(const std::vector<at::Tensor>&,
      const std::string& mode="fp32") const;
  at::Tensor dense_position(const std::string& mode="fp32") const;
  // One conditioned 72x72 image feature per bucket; high-resolution maps may
  // be shared across buckets. Outputs are demuxed back to valid object order.
  VideoMaskOutput forward(const MultiplexState&,const at::Tensor& image,
      const std::vector<at::Tensor>& projected_high,const std::string& mode="fp32",
      double object_threshold=0.,bool attenuate_iou_by_stability=false) const;
 private:
  MultiplexMaskDecoder decoder_;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
}
