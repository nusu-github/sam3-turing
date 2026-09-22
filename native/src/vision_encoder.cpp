#include "sam3/vision_encoder.h"
#include "sam3/autocast.h"
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <set>

namespace sam3 {
VisionEncoder::VisionEncoder(const WeightStore& store, const std::string& model, at::Device device)
    : device_(device) {
  const auto prefix = model + "/detector.backbone.vision_backbone.";
  auto values = store.read_prefix(prefix, device);
  for (auto& [name, tensor] : values) weights_.emplace(name.substr(prefix.size()), std::move(tensor));
  TORCH_CHECK(weight("trunk.patch_embed.proj.weight").sizes() == at::IntArrayRef({1024,3,14,14}), "unexpected SAM3 patch embedding");
  TORCH_CHECK(weight("trunk.pos_embed").sizes() == at::IntArrayRef({1,577,1024}), "unexpected absolute positions");
  heads_ = {"convs"};
  if (weights_.count("sam2_convs.0.conv_1x1.weight")) heads_.push_back("sam2_convs");
  if (weights_.count("interactive_convs.0.conv_1x1.weight")) heads_.push_back("interactive_convs");
  if (weights_.count("propagation_convs.0.conv_1x1.weight")) heads_.push_back("propagation_convs");
  levels_ = weights_.count("convs.3.conv_1x1.weight") ? 4 : 3;
  for (int i = 0; i < 32; ++i) weight("trunk.blocks." + std::to_string(i) + ".attn.qkv.weight");
}
const at::Tensor& VisionEncoder::weight(const std::string& name) const {
  const auto it = weights_.find(name);
  TORCH_CHECK(it != weights_.end(), "missing vision weight: ", name);
  return it->second;
}
at::Tensor VisionEncoder::norm(const at::Tensor& x, const std::string& prefix) const {
  return at::layer_norm(x, {1024}, weight(prefix + ".weight"), weight(prefix + ".bias"), 1e-5);
}
at::Tensor VisionEncoder::linear(const at::Tensor& x, const std::string& prefix) const {
  return at::linear(x, weight(prefix + ".weight"), weight(prefix + ".bias"));
}
at::Tensor VisionEncoder::attention(const at::Tensor& x, const std::string& prefix) const {
  const auto b = x.size(0), h = x.size(1), w = x.size(2), length = h * w;
  const auto qkv = linear(x, prefix + ".qkv").reshape({b,length,3,16,64}).permute({2,0,3,1,4});
  const auto& frequencies = weight(prefix + ".freqs_cis");
  TORCH_CHECK(frequencies.sizes() == at::IntArrayRef({length,32}), "RoPE token grid mismatch");
  const auto rotate = [&](const at::Tensor& value) {
    const auto complex = at::view_as_complex(value.to(at::kFloat).reshape({b,16,length,32,2}));
    return at::view_as_real(complex * frequencies.view({1,1,length,32})).flatten(3).to(value.scalar_type());
  };
  const auto q = rotate(qkv[0]), k = rotate(qkv[1]);
  const auto attended = at::scaled_dot_product_attention(q, k, qkv[2]);
  const auto output = attended.view({b,16,h,w,64}).permute({0,2,3,1,4}).reshape({b,h,w,1024});
  return linear(output, prefix + ".proj");
}
at::Tensor VisionEncoder::block(const at::Tensor& input, int64_t layer, bool fused_bf16) const {
  const auto prefix = "trunk.blocks." + std::to_string(layer);
  auto x = norm(input, prefix + ".norm1");
  const auto b = x.size(0), h = x.size(1), w = x.size(2);
  const bool windowed = (layer + 1) % 8 != 0;
  const auto hp = ((h + 23) / 24) * 24, wp = ((w + 23) / 24) * 24;
  if (windowed) {
    if (hp != h || wp != w) x = at::constant_pad_nd(x, {0,0,0,wp-w,0,hp-h}, 0);
    x = x.view({b,hp/24,24,wp/24,24,1024}).permute({0,1,3,2,4,5}).reshape({-1,24,24,1024});
  }
  x = attention(x, prefix + ".attn");
  if (windowed) {
    x = x.reshape({b,hp/24,wp/24,24,24,1024}).permute({0,1,3,2,4,5}).reshape({b,hp,wp,1024});
    x = x.slice(1,0,h).slice(2,0,w);
  }
  x = input + x;
  const auto normalized = norm(x, prefix + ".norm2");
  at::Tensor hidden;
  if (fused_bf16) {
    // Reference-only path: exactly reproduce upstream's forced BF16 epilogue.
    const auto flat = normalized.to(at::kBFloat16).view({-1,1024});
    hidden = at::_addmm_activation(weight(prefix + ".mlp.fc1.bias").to(at::kBFloat16), flat,
      weight(prefix + ".mlp.fc1.weight").to(at::kBFloat16).t(), 1, 1, true).view({b,h,w,4736});
  } else {
    hidden = at::gelu(linear(normalized, prefix + ".mlp.fc1"), "none");
  }
  return x + linear(hidden, prefix + ".mlp.fc2");
}
at::Tensor VisionEncoder::neck(const at::Tensor& input, const std::string& head, int64_t level) const {
  const auto prefix = head + "." + std::to_string(level);
  auto x = input;
  const auto transpose = [&](const at::Tensor& in, const std::string& name) {
    return at::conv_transpose2d(in, weight(name + ".weight"), weight(name + ".bias"), {2,2}, {0,0}, {0,0}, 1, {1,1});
  };
  if (level == 0) {
    x = transpose(x, prefix + ".dconv_2x2_0");
    x = transpose(at::gelu(x, "none"), prefix + ".dconv_2x2_1");
  } else if (level == 1) {
    x = transpose(x, prefix + ".dconv_2x2");
  } else if (level == 3) {
    x = at::max_pool2d(x, {2,2}, {2,2});
  }
  x = at::conv2d(x, weight(prefix + ".conv_1x1.weight"), weight(prefix + ".conv_1x1.bias"));
  return at::conv2d(x, weight(prefix + ".conv_3x3.weight"), weight(prefix + ".conv_3x3.bias"), {1,1}, {1,1});
}
at::Tensor VisionEncoder::position(const at::Tensor& x) const {
  const auto b = x.size(0), h = x.size(2), w = x.size(3);
  const auto options = x.options().dtype(at::kFloat);
  auto y = at::arange(1,h+1,options).view({1,h,1}).repeat({b,1,w});
  auto xx = at::arange(1,w+1,options).view({1,1,w}).repeat({b,h,1});
  y = y / (y.slice(1,h-1,h) + 1e-6) * (2 * std::acos(-1.0));
  xx = xx / (xx.slice(2,w-1,w) + 1e-6) * (2 * std::acos(-1.0));
  auto dim = at::arange(128,options);
  dim = at::pow(10000.0, 2 * dim.floor_divide(2) / 128);
  const auto encode = [&](const at::Tensor& coordinate) {
    const auto angle = coordinate.unsqueeze(-1) / dim;
    return at::stack({angle.slice(3,0,128,2).sin(), angle.slice(3,1,128,2).cos()},4).flatten(3);
  };
  return at::cat({encode(y),encode(xx)},3).permute({0,3,1,2}).to(x.scalar_type());
}
VisionFeatures VisionEncoder::forward(const at::Tensor& image, const std::string& mode,
                                      const std::vector<std::string>& heads) const {
  c10::InferenceMode inference;
  TORCH_CHECK(image.dim() == 4 && image.size(0) > 0 && image.size(1) == 3 &&
    image.size(2) == 1008 && image.size(3) == 1008 && image.is_floating_point(),
    "vision expects normalized floating RGB [B,3,1008,1008]");
  TORCH_CHECK(mode == "fp32" || mode == "fp16" || mode == "bf16_reference", "invalid vision precision mode");
  AutocastGuard autocast(device_.type(), mode != "fp32", mode == "fp16" ? at::kHalf : at::kBFloat16);
  const auto selected = heads.empty() ? heads_ : heads;
  std::set<std::string> unique;
  for (const auto& name : selected) {
    TORCH_CHECK(std::find(heads_.begin(),heads_.end(),name) != heads_.end(), "unknown vision head: ", name);
    TORCH_CHECK(unique.insert(name).second, "duplicate vision head");
  }
  auto x = at::conv2d(image.to(device_,at::kFloat), weight("trunk.patch_embed.proj.weight"), std::nullopt, {14,14}).permute({0,2,3,1});
  const auto pos = weight("trunk.pos_embed").slice(1,1).reshape({1,24,24,1024}).permute({0,3,1,2});
  x = x + pos.repeat({1,1,4,4}).slice(2,0,72).slice(3,0,72).permute({0,2,3,1});
  x = norm(x,"trunk.ln_pre");
  for (int64_t i = 0; i < 32; ++i) x = block(x,i,mode == "bf16_reference");
  VisionFeatures result;
  result.trunk = x.permute({0,3,1,2});
  for (const auto& head : selected) {
    auto& pyramid = result.pyramid[head];
    for (int64_t level = 0; level < levels_; ++level) {
      auto output = neck(result.trunk,head,level);
      if (result.positions.size() <= static_cast<size_t>(level)) result.positions.push_back(position(output));
      pyramid.push_back(std::move(output));
    }
  }
  return result;
}
}
