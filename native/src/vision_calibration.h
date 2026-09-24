#pragma once
// Opt-in research data only. No public ABI or checkpoint-format change.
#include <ATen/ATen.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include "vision_experiments.h"

namespace sam3::detail {
inline void observe_fc2(const at::Tensor& value, const at::Tensor& weight, int layer) {
  const auto root = read_experiment("SAM3_EXPERIMENT_OBSERVE_FC2", "");
  if (root.empty()) return;
  // A calibration process owns exactly one image. Avoid duplicate cold/timed writes.
  static std::mutex mutex;
  static std::set<std::pair<std::string, int>> written;
  const std::lock_guard<std::mutex> lock(mutex);
  if (written.count({root, layer})) return;
  const auto flat = value.reshape({-1,value.size(-1)}).to(at::kFloat);
  const auto stats = at::stack({flat.amin(0),flat.amax(0),flat.mean(0),
      flat.square().mean(0).sqrt(),weight.to(at::kFloat).abs().amax(0)}).cpu().contiguous();
  TORCH_CHECK(at::isfinite(stats).all().item<bool>(), "nonfinite calibration data");
  std::filesystem::create_directories(std::filesystem::u8path(root));
  const auto path = std::filesystem::u8path(root) / ("fc2-" + std::to_string(layer) + ".f32.bin");
  TORCH_CHECK(!std::filesystem::exists(path), "refusing to overwrite calibration observation: ",path.u8string());
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(stats.const_data_ptr()),stats.nbytes());
  TORCH_CHECK(out, "cannot write calibration observation");
  written.insert({root,layer});
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
} // namespace sam3::detail
