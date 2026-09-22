#pragma once
#include "sam3/weights.h"
namespace sam3 {
struct MaskMemoryOutput { at::Tensor features,position; };
struct MemoryFrameOptions {
  bool from_points=false;
  bool non_overlap=false;
  double object_threshold=0.; // SAM3.1 only; SAM3 uses zero
};
class SAM3_NATIVE_EXPORT MaskMemoryEncoder {
 public:
  MaskMemoryEncoder(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  // Neural module: masks have one channel for SAM3, 32 for SAM3.1 (16 mask
  // slots + 16 conditioning channels). Image features have 256 channels;
  // one shared image may broadcast across objects/buckets, including CPU staging.
  MaskMemoryOutput forward(const at::Tensor& image,const at::Tensor& masks,
      bool skip_mask_sigmoid=false,const std::string& mode="fp32") const;
  // Tracker host: masks [objects,1,H,W], scores [objects,1]. SAM3.1's
  // mux matrix is [buckets*16,objects], matching MultiplexState.mux_matrix.
  // It preserves source matmul/AMP semantics, including padding slots.
  MaskMemoryOutput encode_frame(const at::Tensor& image,const at::Tensor& masks,
      const at::Tensor& object_logits,const MemoryFrameOptions& options={},
      const at::Tensor& mux_matrix={},const at::Tensor& conditioning_objects={},
      const std::string& mode="fp32") const;
 private:
  std::map<std::string,at::Tensor> weights_;
  at::Tensor absent_embedding_,position_;
  at::Device device_;
  bool multiplex_;
};
}
