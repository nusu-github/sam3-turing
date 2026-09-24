#pragma once
#include "vision_calibration.h"
#include "vision_quantization.h"
#include "sam3/autocast.h"
#include "sam3/vision_fusion.h"

namespace sam3::detail {
// Calibration-only shadow calculation. Never returns a tensor to the model.
inline void observe_qkv_error(const at::Tensor& input,const at::Tensor& normalized,
    const at::Tensor& gamma,const at::Tensor& beta,const at::Tensor& weight,
    const at::Tensor& bias,int layer,bool windowed) {
  const auto root=read_experiment("SAM3_EXPERIMENT_OBSERVE_QKV_ERROR","");
  if(root.empty())return;
  static std::mutex mutex;
  static std::set<std::pair<std::string,int>> written;
  const std::lock_guard<std::mutex> lock(mutex);
  if(written.count({root,layer}))return;
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kFloat &&
      input.sizes()==at::IntArrayRef({1,72,72,1024}) && normalized.scalar_type()==at::kHalf,
      "QKV error observation requires fused FP16 normalization on one 72x72 image");
  const auto calibration=read_experiment("SAM3_EXPERIMENT_QKV_SHADOW_CALIBRATION","");
  TORCH_CHECK(!calibration.empty(),"QKV error observation requires shadow calibration");
  // Shadow setup must use the same FP32 arithmetic as model setup, outside autocast.
  AutocastGuard no_autocast(at::kCUDA,false,at::kHalf);
  const auto path=std::filesystem::u8path(calibration);
  const auto affine=read_qkv_data(layer,path,true,input.device());
  const auto r=affine[0].contiguous(),shift=affine[1].contiguous();
  const auto mean=read_qkv_data(layer,path,false,input.device())[0];
  auto [g,b]=fold_qkv_norm(gamma,beta,r,shift);
  const auto transformed=vision_norm_projection(input,g,b,at::kHalf,windowed);
  const auto w=(weight.to(at::kFloat)*r).to(at::kHalf).contiguous();
  auto q=at::empty(w.sizes(),w.options().dtype(at::kChar));
  auto scales=at::empty({w.size(0)},w.options().dtype(at::kFloat));
  TORCH_CHECK(at::isfinite(w).all().item<bool>(),"QKV shadow weight overflows FP16");
  approx_quant(w,q,scales);
  const auto base_bias=linear_mean_bias(weight,bias,q,scales,mean,r,shift).to(at::kHalf).contiguous();
  TORCH_CHECK(at::isfinite(base_bias).all().item<bool>(),"QKV shadow bias overflows FP16");
  const auto candidate=quantized_linear(transformed,q,scales,base_bias,false,false,"vision.qkv_shadow");
  const auto reference=at::linear(normalized.reshape({5184,1024}),weight,bias);
  const auto stats=qkv_error_statistics(reference,candidate,base_bias).cpu().contiguous();
  std::filesystem::create_directories(std::filesystem::u8path(root));
  const auto output=std::filesystem::u8path(root)/("qkv_error-"+std::to_string(layer)+".f32.bin");
  TORCH_CHECK(!std::filesystem::exists(output),"refusing to overwrite QKV error observation");
  std::ofstream out(output,std::ios::binary);
  out.write(static_cast<const char*>(stats.const_data_ptr()),stats.nbytes());
  TORCH_CHECK(out,"cannot write QKV error observation");
  written.insert({root,layer});
}
} // namespace sam3::detail
