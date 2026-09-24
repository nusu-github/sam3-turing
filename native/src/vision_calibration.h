#pragma once
// Opt-in research data only. No public ABI or checkpoint-format change.
#include <ATen/ATen.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include "vision_experiments.h"

namespace sam3::detail {
inline void observe_linear(const at::Tensor& value, const at::Tensor& weight, int layer,
    const char* variable, const std::string& kind) {
  const auto root = read_experiment(variable, "");
  if (root.empty()) return;
  // A calibration process owns exactly one image. Avoid duplicate cold/timed writes.
  static std::mutex mutex;
  static std::set<std::pair<std::string, std::string>> written;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto filename=kind + "-" + std::to_string(layer) + ".f32.bin";
  if (written.count({root, filename})) return;
  // QKV's native FP16 linear consumes the projection after its Half cast.
  const auto projected=kind=="qkv" ? value.to(at::kHalf) : value;
  const auto flat = projected.reshape({-1,value.size(-1)}).to(at::kFloat);
  const auto stats = at::stack({flat.amin(0),flat.amax(0),flat.mean(0),
      flat.square().mean(0).sqrt(),weight.to(at::kFloat).abs().amax(0)}).cpu().contiguous();
  TORCH_CHECK(at::isfinite(stats).all().item<bool>(), "nonfinite calibration data");
  std::filesystem::create_directories(std::filesystem::u8path(root));
  const auto path = std::filesystem::u8path(root) / filename;
  TORCH_CHECK(!std::filesystem::exists(path), "refusing to overwrite calibration observation: ",path.u8string());
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(stats.const_data_ptr()),stats.nbytes());
  TORCH_CHECK(out, "cannot write calibration observation");
  written.insert({root,filename});
}
inline void observe_fc2(const at::Tensor& value, const at::Tensor& weight, int layer) {
  observe_linear(value,weight,layer,"SAM3_EXPERIMENT_OBSERVE_FC2","fc2");
}
inline void observe_qkv(const at::Tensor& value, const at::Tensor& weight, int layer) {
  observe_linear(value,weight,layer,"SAM3_EXPERIMENT_OBSERVE_QKV","qkv");
}

inline std::pair<at::Tensor,at::Tensor> read_fc2_calibration(int layer, at::Device device) {
  const auto root = read_experiment("SAM3_EXPERIMENT_FC2_CALIBRATION", "");
  if(root.empty()) return {};
  constexpr int64_t width=4736;
  const auto path=std::filesystem::u8path(root)/"fc2-affine.f32.bin";
  TORCH_CHECK(std::filesystem::file_size(path)==32*2*width*sizeof(float),"invalid FC2 calibration size");
  auto values=at::empty({2,width},at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  std::ifstream in(path,std::ios::binary);
  in.seekg(layer*2*width*sizeof(float));
  in.read(static_cast<char*>(values.mutable_data_ptr()),values.nbytes());
  TORCH_CHECK(in && at::isfinite(values).all().item<bool>() && (values[0]>0).all().item<bool>(),
      "invalid FC2 scale/shift calibration");
  values=values.to(device);
  return {values[0].contiguous(),values[1].contiguous()};
}

inline at::Tensor read_fc2_mean(int layer, at::Device device) {
  constexpr int64_t width=4736;
  const auto root=read_experiment("SAM3_EXPERIMENT_FC2_CALIBRATION", "");
  const auto path=std::filesystem::u8path(root)/"fc2-mean.f32.bin";
  TORCH_CHECK(std::filesystem::file_size(path)==32*width*sizeof(float),"invalid FC2 mean size");
  auto mean=at::empty({width},at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  std::ifstream in(path,std::ios::binary);
  in.seekg(layer*width*sizeof(float));
  in.read(static_cast<char*>(mean.mutable_data_ptr()),mean.nbytes());
  TORCH_CHECK(in && at::isfinite(mean).all().item<bool>(),"invalid FC2 mean data");
  return mean.to(device);
}

// Correct expected output from weight quantization only. Activation quantization
// and changed inputs from earlier quantized blocks are deliberately not modeled.
inline at::Tensor linear_mean_bias(const at::Tensor& original_weight,const at::Tensor& bias,
    const at::Tensor& q,const at::Tensor& scales,const at::Tensor& mean,
    const at::Tensor& r,const at::Tensor& shift) {
  const auto dequantized=q.to(at::kFloat)*scales.unsqueeze(1);
  return bias.to(at::kFloat) + at::mv(original_weight.to(at::kFloat),mean) -
      at::mv(dequantized,mean/r-shift);
}
inline at::Tensor fc2_mean_bias(const at::Tensor& original_weight,const at::Tensor& bias,
    const at::Tensor& q,const at::Tensor& scales,const at::Tensor& mean,
    const at::Tensor& r,const at::Tensor& shift) {
  return linear_mean_bias(original_weight,bias,q,scales,mean,r,shift);
}

inline at::Tensor read_qkv_data(int layer, const std::filesystem::path& root,
    bool affine, at::Device device) {
  constexpr int64_t width=1024;
  TORCH_CHECK(layer>=0 && layer<32,"invalid QKV calibration layer");
  const int64_t rows=affine?2:1;
  const auto path=root/(affine?"qkv-affine.f32.bin":"qkv-mean.f32.bin");
  TORCH_CHECK(std::filesystem::file_size(path)==32*rows*width*sizeof(float),"invalid QKV calibration size");
  auto values=at::empty({rows,width},at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  std::ifstream in(path,std::ios::binary);
  in.seekg(layer*rows*width*sizeof(float));
  in.read(static_cast<char*>(values.mutable_data_ptr()),values.nbytes());
  TORCH_CHECK(in && at::isfinite(values).all().item<bool>(),"invalid QKV calibration data");
  TORCH_CHECK(!affine || (values[0]>0).all().item<bool>(),"QKV scales must be positive");
  return values.to(device);
}

inline at::Tensor read_qkv_output_bias(int layer, const std::filesystem::path& root,
    at::Device device) {
  constexpr int64_t width=3072;
  TORCH_CHECK(layer>=0 && layer<32,"invalid QKV output bias layer");
  const auto path=root/"qkv-output-bias.f32.bin";
  TORCH_CHECK(std::filesystem::file_size(path)==32*width*sizeof(float),"invalid QKV output bias size");
  auto values=at::empty({width},at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  std::ifstream in(path,std::ios::binary);
  in.seekg(layer*width*sizeof(float));
  in.read(static_cast<char*>(values.mutable_data_ptr()),values.nbytes());
  TORCH_CHECK(in && at::isfinite(values).all().item<bool>(),"invalid QKV output bias data");
  values=values.to(device,at::kHalf).contiguous();
  TORCH_CHECK(at::isfinite(values).all().item<bool>(),"QKV output bias overflows FP16");
  return values;
}

// Sufficient statistics of the actual rounded projection outputs, before RoPE.
// Rows: mean error, mean squared error, reference mean, candidate mean, base bias.
inline at::Tensor qkv_error_statistics(const at::Tensor& reference,
    const at::Tensor& candidate,const at::Tensor& bias) {
  TORCH_CHECK(reference.dim()==2 && reference.sizes()==candidate.sizes() &&
      reference.size(0)>0 && bias.dim()==1 && bias.size(0)==reference.size(1),
      "QKV output statistics shape mismatch");
  const auto ref=reference.to(at::kFloat),actual=candidate.to(at::kFloat);
  const auto diff=ref-actual;
  auto stats=at::stack({diff.mean(0),diff.square().mean(0),ref.mean(0),actual.mean(0),bias.to(at::kFloat)});
  TORCH_CHECK(at::isfinite(stats).all().item<bool>(),"nonfinite QKV output statistics");
  return stats;
}

// LN'(u) = LN(u)/r - shift in exact arithmetic. The fused normalization
// rounds only after this transformed affine, unlike transforming a Half LN.
inline std::pair<at::Tensor,at::Tensor> fold_qkv_norm(const at::Tensor& gamma,
    const at::Tensor& beta,const at::Tensor& r,const at::Tensor& shift) {
  TORCH_CHECK(gamma.dim()==1 && gamma.sizes()==beta.sizes() && gamma.sizes()==r.sizes() &&
      gamma.sizes()==shift.sizes(),"QKV norm calibration shape mismatch");
  TORCH_CHECK(at::isfinite(r).all().item<bool>() && (r>0).all().item<bool>() &&
      at::isfinite(shift).all().item<bool>(),"invalid QKV norm calibration");
  auto g=(gamma/r).contiguous(),b=(beta/r-shift).contiguous();
  TORCH_CHECK(at::isfinite(g).all().item<bool>() && at::isfinite(b).all().item<bool>(),
      "QKV folded normalization overflows");
  return {g,b};
}
} // namespace sam3::detail
