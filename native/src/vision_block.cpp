#include "sam3/vision_encoder.h"
#include "sam3/autocast.h"
#include "sam3/vision_fusion.h"
#include "profile_range.h"
#ifdef SAM3_EXPERIMENT_INT8_GEMM
#include "sam3/int4_experiment.h"
#endif

#include "vision_experiments.h"
#include "vision_calibration.h"
#ifdef SAM3_WITH_CUDA
#include "vision_quantization.h"
#include "vision_qkv_observer.h"
#endif

namespace sam3 {
at::Tensor VisionEncoder::block(const at::Tensor& input, int64_t layer, bool fused_bf16) const {
  return block(input, layer, fused_bf16, nullptr);
}
at::Tensor VisionEncoder::block(const at::Tensor& input, int64_t layer, bool fused_bf16, at::Tensor* prepared) const {
  detail::ProfileRange block_range("vision.block." + std::to_string(layer));
  const auto prefix = "trunk.blocks." + std::to_string(layer);
  const auto b = input.size(0), h = input.size(1), w = input.size(2);
  const bool windowed = (layer + 1) % 8 != 0;
#ifdef SAM3_FUSE_VISION_NORM
  const bool fuse_norm = input.is_cuda() && input.scalar_type() == at::kFloat &&
      h % 24 == 0 && w % 24 == 0;
#else
  const bool fuse_norm = false;
#endif
  const auto projection_type = at::autocast::is_autocast_enabled(at::kCUDA)
      ? (fused_bf16 ? at::kBFloat16 : at::kHalf) : at::kFloat;
  auto x = fuse_norm && prepared && prepared->defined() ? std::move(*prepared)
      : fuse_norm ? vision_norm_projection(input, weight(prefix + ".norm1.weight"),
      weight(prefix + ".norm1.bias"), projection_type, windowed) : norm(input, prefix + ".norm1");
  const auto hp = ((h + 23) / 24) * 24, wp = ((w + 23) / 24) * 24;
  if (windowed && !fuse_norm) {
    if (hp != h || wp != w) x = at::constant_pad_nd(x, {0,0,0,wp-w,0,hp-h}, 0);
    x = x.view({b,hp/24,24,wp/24,24,1024}).permute({0,1,3,2,4,5}).reshape({-1,24,24,1024});
  }
  detail::observe_qkv(x,weight(prefix+".attn.qkv.weight"),int(layer));
#ifdef SAM3_WITH_CUDA
  detail::observe_qkv_error(input,x,weight(prefix+".norm1.weight"),weight(prefix+".norm1.bias"),
      weight(prefix+".attn.qkv.weight"),weight(prefix+".attn.qkv.bias"),int(layer),windowed);
#endif
  x = attention(x, prefix + ".attn");
  at::Tensor normalized;
  if (fuse_norm) {
    std::tie(x, normalized) = vision_residual_norm(input, x,
        weight(prefix + ".norm2.weight"), weight(prefix + ".norm2.bias"),
        projection_type, windowed);
  } else {
    if (windowed) {
      x = x.reshape({b,hp/24,wp/24,24,24,1024}).permute({0,1,3,2,4,5}).reshape({b,hp,wp,1024});
      x = x.slice(1,0,h).slice(2,0,w);
    }
    x = input + x;
    normalized = norm(x, prefix + ".norm2");
  }
  at::Tensor hidden, projection;
  if (fused_bf16) {
    // Reference-only path: exactly reproduce upstream's forced BF16 epilogue.
    const auto flat = normalized.to(at::kBFloat16).view({-1,1024});
    hidden = at::_addmm_activation(weight(prefix + ".mlp.fc1.bias").to(at::kBFloat16), flat,
      weight(prefix + ".mlp.fc1.weight").to(at::kBFloat16).t(), 1, 1, true).view({b,h,w,4736});
  } else {
    // Opt-in local experiment. Unset/exact retains the original arithmetic.
    auto mode = detail::checked_experiment("SAM3_EXPERIMENT_MLP",
        {"exact", "int8_boundary"}, "invalid SAM3_EXPERIMENT_MLP: ");
    if(mode=="int8_boundary" && !detail::mlp_int8_layer(layer))mode="exact";
    if (mode == "int8_boundary") {
#ifdef SAM3_WITH_CUDA
      const auto& mlp_part=detail::mlp_int8_part();
      if (mlp_part!="fc1") {
        const auto fc1=prefix+".mlp.fc1",fc2=prefix+".mlp.fc2";
        const bool calibrated=weights_.count(fc2+".calib_r")!=0;
        const auto& fc2_bias=weight(fc2+(calibrated?".calib_bias":".bias"));
        at::Tensor qhidden,shidden,int4_offset;
        if(mlp_part=="fc2") {
          // Preserve the exact FP16 FC1 output, then fuse GELU, optional affine,
          // and FC2 activation quantization. Only FC2 uses integer GEMM here.
          auto pre=detail::profile_call("vision.mlp.fc1.fp16",[&]{return linear(normalized,fc1);});
          pre=pre.reshape({-1,4736}).contiguous();
          qhidden=at::empty(pre.sizes(),pre.options().dtype(at::kChar));
          shidden=at::empty({pre.size(0)},pre.options().dtype(at::kFloat));
          detail::profile_call("vision.mlp.gelu_quant",[&]{
            approx_gelu_quant(pre,calibrated?weight(fc2+".calib_r"):at::Tensor(),
                calibrated?weight(fc2+".calib_shift"):at::Tensor(),qhidden,shidden);return 0;
          });
        } else {
          detail::ProfileRange fc1_range("vision.mlp.fc1");
          const auto flat=normalized.to(at::kHalf).reshape({-1,1024}).contiguous();
          auto [q, scales] = detail::quantize_rows(flat, "vision.mlp.fc1");
          auto accum=detail::profile_call("vision.mlp.fc1.int8_gemm",[&] {return at::_int_mm(q,weight(fc1+".int8").t());});
          if(!detail::int4_fc2_layer(layer) || detail::int4_weight_only()) {
            qhidden=at::empty(accum.sizes(),flat.options().dtype(at::kChar));
            shidden=at::empty_like(scales);
          }
          detail::profile_call("vision.mlp.boundary",[&] {
            if(calibrated) {
              approx_restore_quant_affine(accum,scales,weight(fc1+".scale"),weight(fc1+".bias"),
                  weight(fc2+".calib_r"),weight(fc2+".calib_shift"),qhidden,shidden);
              return 0;
            }
#ifdef SAM3_EXPERIMENT_INT8_GEMM
            if(detail::int4_fc2_layer(layer) && !detail::int4_weight_only()) {
              if(detail::int4_affine_mode())std::tie(qhidden,shidden,int4_offset)=int4_boundary_affine(accum,scales,weight(fc1+".scale"),weight(fc1+".bias"),detail::int4_mse_mode());
              else if(detail::int4_rotation())std::tie(qhidden,shidden)=int4_boundary_rht(accum,scales,weight(fc1+".scale"),weight(fc1+".bias"),detail::int4_rotation());
              else std::tie(qhidden,shidden)=int4_boundary(accum,scales,weight(fc1+".scale"),weight(fc1+".bias"));
              if(detail::int4_activation_only())qhidden=detail::unpack_int4_diagnostic(qhidden);
            }
            else
#endif
            approx_restore_quant(accum,scales,weight(fc1+".scale"),weight(fc1+".bias"),qhidden,shidden);return 0;
          });
        }
        detail::ProfileRange fc2_range("vision.mlp.fc2");
        auto accum=detail::profile_call((detail::int4_fc2_layer(layer) && !detail::int4_weight_only() && !detail::int4_activation_only())?"vision.mlp.fc2.int4_gemm":"vision.mlp.fc2.int8_gemm",[&] {
#ifdef SAM3_EXPERIMENT_INT8_GEMM
          if(detail::int4_fc2_layer(layer) && !detail::int4_weight_only() && !detail::int4_activation_only())return int4_mm(qhidden,weight(fc2+".int8"));
#endif
          return at::_int_mm(qhidden,weight(fc2+".int8").t());});
#ifdef SAM3_EXPERIMENT_INT8_GEMM
        if(int4_offset.defined())detail::profile_call("vision.mlp.fc2.affine_correction",[&]{int4_correct(accum,int4_offset,weight(fc2+".sum"));return 0;});
#endif
        const auto fc2_mode = detail::checked_experiment("SAM3_EXPERIMENT_FC2_NORM",
            {"exact", "fused"}, "invalid FC2 norm experiment: ");
        if(fc2_mode=="fused" && fuse_norm && prepared && layer<31 && projection_type==at::kHalf) {
          const auto next="trunk.blocks."+std::to_string(layer+1)+".norm1";
          at::Tensor result;
          std::tie(result,*prepared)=detail::profile_call("vision.mlp.fc2.restore_norm",[&]{
            return approx_fc2_norm(x,accum,shidden,weight(fc2+".scale"),fc2_bias,weight(next+".weight"),weight(next+".bias"),(layer+2)%8!=0);
          });
          return result;
        }
        auto result=at::empty(accum.sizes(),normalized.options().dtype(at::kHalf));
        detail::profile_call("vision.mlp.fc2.restore",[&] {
          approx_restore(accum,shidden,weight(fc2+".scale"),fc2_bias,result,false);return 0;
        });
        projection=result.view({b,h,w,1024});
      } else {
        // FC1-only isolation: unfused INT8 FC1 with restore/GELU; FC2 stays FP16 below.
        const auto fc1=prefix+".mlp.fc1";
        detail::ProfileRange projection_range("vision.mlp.fc1");
        hidden = detail::quantized_linear(normalized, weight(fc1 + ".int8"), weight(fc1 + ".scale"),
            weight(fc1 + ".bias"), true, "vision.mlp.fc1").view({b,h,w,4736});
      }
#else
      TORCH_CHECK(false,"experimental int8 requires CUDA");
#endif
    } else {
      // The fresh linear result has no other consumers. Reuse its allocation
      // for exact GELU, keeping the FP16 rounding point before activation.
      hidden = detail::profile_call("vision.mlp.fc1",[&] { return linear(normalized, prefix + ".mlp.fc1"); });
      detail::profile_call("vision.mlp.gelu",[&] { at::gelu_(hidden, "none"); return 0; });
    }
  }
  if (!projection.defined()) {
    detail::observe_fc2(hidden,weight(prefix+".mlp.fc2.weight"),int(layer));
    projection = detail::profile_call("vision.mlp.fc2",[&] { return linear(hidden, prefix + ".mlp.fc2"); });
  }
  if (fuse_norm && prepared && layer < 31) {
    // Complete this block's residual and prepare the next QKV input in one
    // pass. Release consumed MLP intermediates before allocating the pair.
    hidden = at::Tensor();
    normalized = at::Tensor();
    const auto next = "trunk.blocks." + std::to_string(layer + 1) + ".norm1";
    at::Tensor result;
    std::tie(result, *prepared) = vision_residual_norm_projection(x, projection,
        weight(next + ".weight"), weight(next + ".bias"), projection_type,
        (layer + 2) % 8 != 0);
    return result;
  }
  if (prepared) *prepared = at::Tensor();
  return x + projection;
}
} // namespace sam3
