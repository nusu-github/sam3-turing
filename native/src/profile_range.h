#pragma once
#include <string>

namespace sam3::detail {
// Opt-in diagnostics only; no synchronization and no tensor changes.
class ProfileRange {
 public:
  explicit ProfileRange(const std::string& name);
  ~ProfileRange();
  ProfileRange(const ProfileRange&) = delete;
  ProfileRange& operator=(const ProfileRange&) = delete;
 private:
  bool active_ = false;
};
template<class F> auto profile_call(const std::string& name, F&& fn) {
  ProfileRange range(name);
  return fn();
}
}
