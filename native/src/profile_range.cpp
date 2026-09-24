#include "profile_range.h"
#include <cstdlib>
#if defined(SAM3_WITH_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <nvtx3/nvToolsExt.h>
#define SAM3_HAS_NVTX_RANGES
#endif

namespace sam3::detail {
ProfileRange::ProfileRange(const std::string& name) {
#ifdef SAM3_HAS_NVTX_RANGES
  static const bool enabled = [] {
    const auto* value = std::getenv("SAM3_PROFILE_NVTX");
    return value && std::string(value) == "1";
  }();
  active_ = enabled;
  if (active_) nvtxRangePushA(name.c_str());
#endif
}
ProfileRange::~ProfileRange() {
#ifdef SAM3_HAS_NVTX_RANGES
  if (active_) nvtxRangePop();
#endif
}
}
