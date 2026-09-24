#pragma once
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <vector>

namespace sam3::detail {
// Cached neural features must not cross a change in Torch inference policy.
// These are process-wide settings; callers must not change them concurrently
// with inference, as required by sam3_runtime_configure's existing contract.
using InferenceSettings = std::vector<int64_t>;
inline InferenceSettings inference_settings() {
  auto &c = at::globalContext();
  InferenceSettings result{
      at::get_num_threads(),
      int64_t(at::get_num_interop_threads()),
      int64_t(c.float32MatmulPrecision()),
      int64_t(
          c.float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL)),
      int64_t(
          c.float32Precision(at::Float32Backend::CUDA, at::Float32Op::CONV)),
      int64_t(c.float32Precision(at::Float32Backend::MKLDNN,
                                 at::Float32Op::MATMUL)),
      int64_t(
          c.float32Precision(at::Float32Backend::MKLDNN, at::Float32Op::CONV)),
      int64_t(c.allowFP16ReductionCuBLAS()),
      int64_t(c.allowBF16ReductionCuBLAS()),
      c.allowFP16AccumulationCuBLAS(),
      int64_t(c.blasPreferredBackend()),
      int64_t(c.linalgPreferredBackend()),
      c.userEnabledCuDNN(),
      c.userEnabledMkldnn(),
      c.userEnabledNNPACK(),
      c.benchmarkCuDNN(),
      c.benchmarkLimitCuDNN(),
      c.deterministicCuDNN(),
      c.deterministicMkldnn(),
      c.deterministicAlgorithms(),
      c.deterministicAlgorithmsWarnOnly(),
      c.userEnabledFlashSDP(),
      c.userEnabledMemEfficientSDP(),
      c.userEnabledMathSDP(),
      c.userEnabledCuDNNSDP(),
      c.userEnabledOverrideableSDP(),
      c.allowFP16BF16ReductionMathSDP()};
  for (auto backend : c.sDPPriorityOrder())
    result.push_back(int64_t(backend));
  return result;
}
} // namespace sam3::detail
