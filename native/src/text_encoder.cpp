#include "sam3/text_encoder.h"
#include "sam3/autocast.h"
#include <c10/core/InferenceMode.h>
#include <limits>

namespace sam3 {
TextEncoder::TextEncoder(const WeightStore& store, const std::string& model, at::Device device)
    : device_(device) {
  const auto prefix = model + "/detector.backbone.language_backbone.";
  auto values = store.read_prefix(prefix, device);
  for (auto& [name, value] : values) weights_.emplace(name.substr(prefix.size()), std::move(value));
  width_ = weight("encoder.token_embedding.weight").size(1);
  context_ = weight("encoder.positional_embedding").size(0);
  TORCH_CHECK(width_ == 1024 && context_ == 32, "unexpected SAM3 VE text encoder configuration");
  for (; weights_.count("encoder.transformer.resblocks." + std::to_string(layers_) + ".ln_1.weight"); ++layers_) {}
  TORCH_CHECK(layers_ == 24, "expected all 24 SAM3 VE text layers");
}
const at::Tensor& TextEncoder::weight(const std::string& name) const {
  const auto found = weights_.find(name);
  TORCH_CHECK(found != weights_.end(), "missing text weight: ", name);
  return found->second;
}
at::Tensor TextEncoder::norm(const at::Tensor& x, const std::string& name) const {
  return at::layer_norm(x, {width_}, weight(name + ".weight"), weight(name + ".bias"), 1e-5);
}
at::Tensor TextEncoder::linear(const at::Tensor& x, const std::string& name) const {
  return at::linear(x, weight(name + ".weight"), weight(name + ".bias"));
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> TextEncoder::forward(const at::Tensor& tokens,const std::string& mode) const {
  c10::InferenceMode guard;
  TORCH_CHECK(mode=="fp32" || mode=="fp16" || mode=="bf16_reference","unknown text precision mode");
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(tokens.dim() == 2 && (tokens.scalar_type() == at::kLong || tokens.scalar_type() == at::kInt),
              "text tokens must be int32 or int64 [B,L]");
  TORCH_CHECK(tokens.size(1) > 0 && tokens.size(1) <= context_, "token length exceeds model context");
  const auto ids = tokens.to(device_);
  const auto embedded = at::embedding(weight("encoder.token_embedding.weight"), ids);
  const auto batch = ids.size(0), length = ids.size(1), head_dim = width_ / heads_;
  auto x = embedded + weight("encoder.positional_embedding").slice(0, 0, length);
  const auto mask = at::full({length, length}, -std::numeric_limits<float>::infinity(), x.options()).triu_(1);
  for (int64_t layer = 0; layer < layers_; ++layer) {
    const auto prefix = "encoder.transformer.resblocks." + std::to_string(layer);
    // Follow upstream MultiheadAttention's sequence-first projection layout.
    const auto normalized = norm(x, prefix + ".ln_1").transpose(0, 1);
    const auto projected = at::linear(normalized, weight(prefix + ".attn.in_proj_weight"), weight(prefix + ".attn.in_proj_bias"));
    const auto qkv = projected.reshape({length, batch, 3, heads_, head_dim}).permute({2,1,3,0,4});
    const auto attended = at::scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], mask, 0.0, false);
    const auto flattened = attended.permute({2,0,1,3}).contiguous().view({length * batch, width_});
    const auto attention = linear(flattened, prefix + ".attn.out_proj").view({length, batch, width_}).transpose(0,1);
    x = x + attention;
    x = x + linear(at::gelu(linear(norm(x, prefix + ".ln_2"), prefix + ".mlp.c_fc"), "none"), prefix + ".mlp.c_proj");
  }
  const auto memory = norm(x, "encoder.ln_final").transpose(0,1);
  // VE returns token memory, not the pooled text_projection result. The latter
  // is unused by upstream VETextEncoder.forward and is intentionally not run.
  return {ids.eq(0), linear(memory, "resizer"), embedded.transpose(0,1)};
}
}
