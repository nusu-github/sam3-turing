#pragma once
#include <ATen/autocast_mode.h>

namespace sam3 {
// Scoped, thread-local autocast state; never leak a mode into another module.
class AutocastGuard {
 public:
  AutocastGuard(at::DeviceType device, bool enabled, at::ScalarType dtype)
      : device_(device), enabled_(at::autocast::is_autocast_enabled(device)),
        dtype_(at::autocast::get_autocast_dtype(device)) {
    at::autocast::set_autocast_enabled(device, enabled);
    if (enabled) at::autocast::set_autocast_dtype(device, dtype);
    at::autocast::increment_nesting();
  }
  ~AutocastGuard() {
    if (at::autocast::decrement_nesting() == 0) at::autocast::clear_cache();
    at::autocast::set_autocast_enabled(device_, enabled_);
    at::autocast::set_autocast_dtype(device_, dtype_);
  }
  AutocastGuard(const AutocastGuard&) = delete;
  AutocastGuard& operator=(const AutocastGuard&) = delete;
 private:
  at::DeviceType device_;
  bool enabled_;
  at::ScalarType dtype_;
};
}
