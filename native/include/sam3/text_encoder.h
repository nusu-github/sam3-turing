#pragma once
#include "sam3/weights.h"
#include <tuple>

namespace sam3 {
// Token-level VE encoder. Tokenization is provided by the separate Tokenizer.
// Accepts arbitrary int64/int32 token batches up to the model context.
class SAM3_NATIVE_EXPORT TextEncoder {
 public:
  TextEncoder(const WeightStore& store, const std::string& model, at::Device device = at::kCPU);
  // kHalf stores CUDA projection parameters at the forward FP16 compute dtype;
  // norms/embeddings stay at source precision and only fp16 mode is permitted.
  // kFloat and the original constructor retain forward precision switching.
  TextEncoder(const WeightStore& store, const std::string& model,
              at::Device device, at::ScalarType compute_storage);
  // padding mask [B,L], resized memory [L,B,256], input embeds [L,B,1024].
  std::tuple<at::Tensor, at::Tensor, at::Tensor> forward(const at::Tensor& tokens,const std::string& mode="fp32") const;
 private:
  const at::Tensor& weight(const std::string& name) const;
  at::Tensor norm(const at::Tensor& x, const std::string& name) const;
  at::Tensor linear(const at::Tensor& x, const std::string& name) const;
  std::map<std::string, at::Tensor> weights_;
  int64_t width_ = 0, layers_ = 0, heads_ = 16, context_ = 0;
  at::Device device_;
};
}
