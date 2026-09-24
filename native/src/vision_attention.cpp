#include "sam3/vision_encoder.h"
#include "sam3/rotary.h"
#include "sam3/rotary_pair.h"
#include "profile_range.h"
#ifdef SAM3_EXPERIMENT_KITCHEN_ATTENTION
#include "sam3/kitchen_attention.h"
#endif

#include "vision_experiments.h"
#ifdef SAM3_WITH_CUDA
#include "vision_quantization.h"
#endif

namespace sam3 {
at::Tensor VisionEncoder::linear(const at::Tensor& x, const std::string& prefix) const {
#ifdef SAM3_WITH_CUDA
  if(prefix.find(".attn.")!=std::string::npos && weights_.count(prefix+".int8")) {
    const auto& w = weight(prefix + ".int8");
    auto result = detail::quantized_linear(x, w, weight(prefix + ".scale"),
        weight(prefix + (weights_.count(prefix+".calib_bias")?".calib_bias":".bias")), false, "vision.projection");
    auto shape=x.sizes().vec();shape.back()=w.size(0);
    return result.view(shape);
  }
#endif
  return at::linear(x, weight(prefix + ".weight"), weight(prefix + ".bias"));
}
at::Tensor VisionEncoder::attention(const at::Tensor& x, const std::string& prefix) const {
  const auto b = x.size(0), h = x.size(1), w = x.size(2), length = h * w;
  const auto& frequencies = weight(prefix + ".freqs_cis");
  TORCH_CHECK(frequencies.sizes() == at::IntArrayRef({length,32}), "RoPE token grid mismatch");
  at::Tensor q,k,v;
  bool fused=false;
#ifdef SAM3_WITH_CUDA
  const auto fusion = detail::checked_experiment("SAM3_EXPERIMENT_QKV_ROPE",
      {"exact", "fused"}, "invalid QKV RoPE experiment: ");
  // The constructor validates fusion mode. A scoped-out QKV keeps the exact
  // linear/rotary path and has no integer weight allocation.
  if(fusion=="fused" && weights_.count(prefix+".qkv.int8")) {
    const auto name=prefix+".qkv";
    TORCH_CHECK(x.is_cuda() && weights_.count(name+".int8"),"QKV RoPE fusion requires QKV INT8");
    auto packed=detail::profile_call("vision.qkv_rope",[&] {
      auto flat=x.to(at::kHalf).reshape({-1,1024}).contiguous();
      auto [quant, scales] = detail::quantize_rows(flat, "vision.projection");
      auto accum=detail::profile_call("vision.projection.int8_gemm",[&]{return at::_int_mm(quant,weight(name+".int8").t());});
      const auto& bias=weight(name+(weights_.count(name+".calib_bias")?".calib_bias":".bias"));
      return detail::profile_call("vision.projection.restore_rope",[&]{return approx_restore_rope(accum,scales,weight(name+".scale"),bias,frequencies,b);});
    });
    q=packed[0];k=packed[1];v=packed[2];fused=true;
  }
#endif
  if(!fused) {
    const auto qkv=detail::profile_call("vision.qkv",[&]{return linear(x,prefix+".qkv");}).reshape({b,length,3,16,64}).permute({2,0,3,1,4});
#ifdef SAM3_FUSE_VISION_QK
    std::tie(q,k)=detail::profile_call("vision.rope",[&]{return rotary_embedding_pair(qkv[0],qkv[1],frequencies);});
#else
    q=rotary_embedding(qkv[0],frequencies);k=rotary_embedding(qkv[1],frequencies);
#endif
    v=qkv[2];
  }
  const auto attended = detail::profile_call(length == 576 ? "vision.sdpa.local" : "vision.sdpa.global", [&] {
#ifdef SAM3_EXPERIMENT_KITCHEN_ATTENTION
    // kitchen/kitchen_rot replace the 5184-token global layers; *_all also the local ones.
    const auto mode = detail::checked_experiment("SAM3_EXPERIMENT_ATTENTION",
        {"exact", "kitchen", "kitchen_rot", "kitchen_all", "kitchen_rot_all"},
        "invalid attention experiment: ");
    if(mode!="exact" && detail::attention_int8_layer(detail::attention_block_index(prefix)) &&
        (length==5184 || mode=="kitchen_all" || mode=="kitchen_rot_all")) {
      const auto layout = detail::checked_experiment("SAM3_EXPERIMENT_KITCHEN_LAYOUT",
          {"head", "sequence"}, "invalid kitchen output layout: ", "head");
      return kitchen_attention(q,k,v,mode=="kitchen_rot" || mode=="kitchen_rot_all",layout=="sequence");
    }
#endif
    return at::scaled_dot_product_attention(q, k, v);
  });
  const auto output = detail::profile_call("vision.attention_layout", [&] { return attended.view({b,16,h,w,64}).permute({0,2,3,1,4}).reshape({b,h,w,1024}); });
  return detail::profile_call("vision.attention_projection", [&] { return linear(output, prefix + ".proj"); });
}
} // namespace sam3
