#include "../src/inference_settings.h"
#include <algorithm>
#include <iostream>

int main() {
  try {
    at::set_num_threads(2);
    auto &context = at::globalContext();
    context.setAllowTF32CuBLAS(false);
    context.setAllowTF32CuDNN(false);
    const auto original = sam3::detail::inference_settings();
    TORCH_CHECK(original == sam3::detail::inference_settings(),
                "unstable inference settings");
    int checks = 0;
    const auto check = [&](const auto &change, const auto &restore) {
      change();
      TORCH_CHECK(original != sam3::detail::inference_settings(),
                  "settings change was not observed");
      restore();
      TORCH_CHECK(original == sam3::detail::inference_settings(),
                  "settings were not restored");
      ++checks;
    };
    check([] { at::set_num_threads(3); }, [] { at::set_num_threads(2); });
    check([&] { context.setAllowTF32CuBLAS(true); },
          [&] { context.setAllowTF32CuBLAS(false); });
    check([&] { context.setAllowTF32CuDNN(true); },
          [&] { context.setAllowTF32CuDNN(false); });
    const bool flash = context.userEnabledFlashSDP();
    check([&] { context.setSDPUseFlash(!flash); },
          [&] { context.setSDPUseFlash(flash); });
    const bool math = context.userEnabledMathSDP();
    check([&] { context.setSDPUseMath(!math); },
          [&] { context.setSDPUseMath(math); });
    const bool mkldnn = context.userEnabledMkldnn();
    check([&] { context.setUserEnabledMkldnn(!mkldnn); },
          [&] { context.setUserEnabledMkldnn(mkldnn); });
    const bool benchmark = context.benchmarkCuDNN();
    check([&] { context.setBenchmarkCuDNN(!benchmark); },
          [&] { context.setBenchmarkCuDNN(benchmark); });
    const bool deterministic = context.deterministicAlgorithms(),
               warn = context.deterministicAlgorithmsWarnOnly();
    check([&] { context.setDeterministicAlgorithms(!deterministic, warn); },
          [&] { context.setDeterministicAlgorithms(deterministic, warn); });
    const auto reduction = context.allowFP16ReductionCuBLAS();
    const bool reduced =
        reduction == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
    const bool split =
        reduction !=
        at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK;
    check([&] { context.setAllowFP16ReductionCuBLAS(!reduced); },
          [&] { context.setAllowFP16ReductionCuBLAS(reduced, split); });
    std::vector<int64_t> order;
    for (auto backend : context.sDPPriorityOrder())
      order.push_back(int64_t(backend));
    auto reversed = order;
    std::reverse(reversed.begin(), reversed.end());
    check([&] { context.setSDPPriorityOrder(reversed); },
          [&] { context.setSDPPriorityOrder(order); });
    std::cout << "PASS " << checks
              << " inference policy changes and restoration\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
