#pragma once
#include "sam3/weights.h"
namespace sam3 {
class SAM3_NATIVE_EXPORT MemoryAttention {
 public:
  MemoryAttention(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  // Sequence-first [tokens,batch,channels]. Memory channels: SAM3=64,
  // SAM3.1=256. Spatial memory length is a multiple of the square query grid;
  // trailing object-pointer tokens are excluded from rotary encoding.
  // Position tensors may share a singleton batch, following source broadcasting.
  // SAM3.1 also uses image streams, shared (batch=1) or batched. Its memory
  // image may omit the trailing pointer tokens; the source padding is retained.
  at::Tensor forward(const at::Tensor& source,const at::Tensor& source_position,
      const at::Tensor& memory,const at::Tensor& memory_position,int64_t pointer_tokens=0,
      const std::string& mode="fp32",const at::Tensor& image={},
      const at::Tensor& memory_image={},const at::Tensor& memory_image_position={},
      std::vector<at::Tensor>* trace=nullptr) const;
 private:
  at::Tensor attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
      const at::Tensor& frequencies,int64_t pointer_tokens,bool repeat) const;
  std::map<std::string,at::Tensor> weights_;
  at::Tensor frequencies_;
  at::Device device_;
  bool multiplex_;
};
}
