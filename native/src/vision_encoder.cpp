#include "sam3/vision_encoder.h"
#include "sam3/autocast.h"
#include "sam3/vision_position.h"
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <set>
#include <algorithm>
#include "profile_range.h"
#ifdef SAM3_EXPERIMENT_INT8_GEMM
#include "sam3/int4_experiment.h"
#endif

#include "vision_experiments.h"
#ifdef SAM3_WITH_CUDA
#include "vision_quantization.h"
#endif

namespace sam3 {
VisionEncoder::VisionEncoder(const WeightStore& store, const std::string& model, at::Device device)
    : VisionEncoder(store, model, device, at::kFloat) {}
VisionEncoder::VisionEncoder(const WeightStore& store, const std::string& model,
                             at::Device device, at::ScalarType compute_storage)
    : device_(device) {
  TORCH_CHECK(compute_storage == at::kFloat ||
    (compute_storage == at::kHalf && device.is_cuda()),
    "vision compute storage must be Float, or Half on CUDA");
  const auto prefix = model + "/detector.backbone.vision_backbone.";
  for (auto it = store.records().lower_bound(prefix);
       it != store.records().end() && it->first.compare(0,prefix.size(),prefix) == 0; ++it) {
    const auto name = it->first.substr(prefix.size());
    auto tensor = store.read(it->first, device);
    const bool projection = name == "trunk.patch_embed.proj.weight" ||
      (name.compare(0,13,"trunk.blocks.") == 0 &&
       (name.find(".attn.qkv.") != std::string::npos ||
        name.find(".attn.proj.") != std::string::npos ||
        name.find(".mlp.fc1.") != std::string::npos ||
        name.find(".mlp.fc2.") != std::string::npos)) ||
      name.compare(0,6,"convs.") == 0 || name.compare(0,11,"sam2_convs.") == 0 ||
      name.compare(0,18,"interactive_convs.") == 0 ||
      name.compare(0,18,"propagation_convs.") == 0;
    // Convert on the same device as autocast, one tensor at a time. This avoids
    // a full FP32 GPU copy and preserves the original device conversion rules.
    if (compute_storage == at::kHalf && projection) {
      TORCH_CHECK(tensor.scalar_type() == at::kFloat, "expected FP32 vision compute parameter: ", name);
      tensor = tensor.to(at::kHalf);
    }
    weights_.emplace(name, std::move(tensor));
  }
  TORCH_CHECK(weight("trunk.patch_embed.proj.weight").sizes() == at::IntArrayRef({1024,3,14,14}), "unexpected SAM3 patch embedding");
  TORCH_CHECK(weight("trunk.pos_embed").sizes() == at::IntArrayRef({1,577,1024}), "unexpected absolute positions");
  heads_ = {"convs"};
  if (weights_.count("sam2_convs.0.conv_1x1.weight")) heads_.push_back("sam2_convs");
  if (weights_.count("interactive_convs.0.conv_1x1.weight")) heads_.push_back("interactive_convs");
  if (weights_.count("propagation_convs.0.conv_1x1.weight")) heads_.push_back("propagation_convs");
  levels_ = weights_.count("convs.3.conv_1x1.weight") ? 4 : 3;
  for (int i = 0; i < 32; ++i) weight("trunk.blocks." + std::to_string(i) + ".attn.qkv.weight");
#ifdef SAM3_WITH_CUDA
  const auto experiment = detail::read_experiment("SAM3_EXPERIMENT_MLP");
  if(detail::int4_fc2_layer(31))TORCH_CHECK(experiment=="int8_boundary","INT4 FC2 requires int8_boundary MLP");
  if (experiment == "int8" || experiment == "int8_boundary") {
    TORCH_CHECK(device.is_cuda() && compute_storage == at::kHalf,
                "experimental int8 requires CUDA FP16 compute storage");
    for (int layer=0; layer<32; ++layer) for (int fc=1; fc<=2; ++fc) {
      if(!detail::mlp_int8_layer(layer))continue;
      const auto name = "trunk.blocks." + std::to_string(layer) + ".mlp.fc" + std::to_string(fc);
      const auto& w = weight(name + ".weight");
      auto q=at::empty(w.sizes(),w.options().dtype(at::kChar));
      auto scales=at::empty({w.size(0)},w.options().dtype(at::kFloat));
#ifdef SAM3_EXPERIMENT_INT8_GEMM
      if(fc==2 && detail::int4_fc2_layer(layer) && !detail::int4_activation_only()) {
        if(detail::int4_rotation())std::tie(q,scales)=int4_quant_rht(w,detail::int4_rotation());
        else std::tie(q,scales)=int4_quant(w,detail::int4_mse_mode());
        if(detail::int4_weight_only())q=detail::unpack_int4_diagnostic(q);
      }
      else
#endif
      approx_quant(w,q,scales);
#ifdef SAM3_EXPERIMENT_INT8_GEMM
      if(fc==2 && detail::int4_affine_mode()) {
        auto expanded=q.scalar_type()==at::kByte?detail::unpack_int4_diagnostic(q):q;
        weights_.emplace(name+".sum",expanded.sum(1,false,at::kInt).contiguous());
      }
#endif
      weights_.emplace(name + ".int8",std::move(q));
      weights_.emplace(name + ".scale",std::move(scales));
    }
  }
  const auto projection_mode = detail::checked_experiment("SAM3_EXPERIMENT_PROJECTION",
      {"exact", "qkv", "proj", "both"}, "invalid SAM3_EXPERIMENT_PROJECTION: ");
  if(projection_mode!="exact") {
    TORCH_CHECK(device.is_cuda() && compute_storage==at::kHalf,
                "experimental projection INT8 requires CUDA FP16 compute storage");
    for(int layer=0;layer<32;++layer) for(const std::string part : {"qkv","proj"}) {
      if(projection_mode!="both" && projection_mode!=part)continue;
      const auto name="trunk.blocks."+std::to_string(layer)+".attn."+part;
      const auto& w=weight(name+".weight");
      auto q=at::empty(w.sizes(),w.options().dtype(at::kChar));
      auto scales=at::empty({w.size(0)},w.options().dtype(at::kFloat));
      approx_quant(w,q,scales);
      weights_.emplace(name+".int8",std::move(q));
      weights_.emplace(name+".scale",std::move(scales));
    }
  }
#endif
}
const at::Tensor& VisionEncoder::weight(const std::string& name) const {
  const auto it = weights_.find(name);
  TORCH_CHECK(it != weights_.end(), "missing vision weight: ", name);
  return it->second;
}
at::Tensor VisionEncoder::norm(const at::Tensor& x, const std::string& prefix) const {
  return at::layer_norm(x, {1024}, weight(prefix + ".weight"), weight(prefix + ".bias"), 1e-5);
}
at::Tensor VisionEncoder::neck(const at::Tensor& input, const std::string& head, int64_t level) const {
  detail::ProfileRange range("vision.neck." + std::to_string(level));
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
#ifdef SAM3_FUSE_VISION_POSITION
  return vision_position_encoding(x);
#else
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

#endif
}
VisionFeatures VisionEncoder::forward(const at::Tensor& image, const std::string& mode,
                                      const std::vector<std::string>& heads) const {
  std::vector<int64_t> positions;
  for (int64_t level = 0; level < levels_; ++level) positions.push_back(level);
  return forward(image,mode,heads,positions);
}
VisionFeatures VisionEncoder::forward(const at::Tensor& image, const std::string& mode,
    const std::vector<std::string>& heads, const std::vector<int64_t>& position_levels) const {
  detail::ProfileRange range("vision.total");
  c10::InferenceMode inference;
  std::set<int64_t> selected_positions;
  for (auto level : position_levels) {
    TORCH_CHECK(level >= 0 && level < levels_, "position level outside vision pyramid");
    TORCH_CHECK(selected_positions.insert(level).second, "duplicate position level");
  }
  TORCH_CHECK(image.dim() == 4 && image.size(0) > 0 && image.size(1) == 3 &&
    image.size(2) == 1008 && image.size(3) == 1008 && image.is_floating_point(),
    "vision expects normalized floating RGB [B,3,1008,1008]");
  TORCH_CHECK(mode == "fp32" || mode == "fp16" || mode == "bf16_reference", "invalid vision precision mode");
  TORCH_CHECK(weight("trunk.patch_embed.proj.weight").scalar_type() != at::kHalf || mode == "fp16",
              "Half vision compute storage requires fp16 mode");
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
  at::Tensor prepared_projection;
  for (int64_t i = 0; i < 32; ++i)
    x = block(x,i,mode == "bf16_reference", &prepared_projection);
  VisionFeatures result;
  result.positions.resize(levels_);
  result.trunk = x.permute({0,3,1,2});
  for (const auto& head : selected) {
    auto& pyramid = result.pyramid[head];
    for (int64_t level = 0; level < levels_; ++level) {
      auto output = neck(result.trunk,head,level);
      if (selected_positions.count(level) && !result.positions[level].defined()) result.positions[level] = position(output);
      pyramid.push_back(std::move(output));
    }
  }
  return result;
}
}
