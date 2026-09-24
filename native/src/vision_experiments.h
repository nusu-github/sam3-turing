#pragma once
// Internal experiment selection shared by weight preparation and block execution.
// Preserve first-use environment caching and validation from the original encoder.
#include <c10/util/Exception.h>
#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <cstdlib>
#include <string>

namespace sam3::detail {
// No cache here: only callers with a local static freeze their first value.
// An explicitly empty environment variable remains empty, not the fallback.
inline std::string read_experiment(const char* name, const char* fallback = "exact") {
  const auto* value = std::getenv(name);
  return value ? value : fallback;
}
inline void check_experiment(const std::string& value,
    std::initializer_list<std::string_view> allowed, const char* error) {
  TORCH_CHECK(std::find(allowed.begin(), allowed.end(), value) != allowed.end(), error, value);
}
inline std::string checked_experiment(const char* name,
    std::initializer_list<std::string_view> allowed, const char* error,
    const char* fallback = "exact") {
  auto value = read_experiment(name, fallback);
  check_experiment(value, allowed, error);
  return value;
}

inline bool mlp_int8_layer(int layer) {
  static const std::string scope=read_experiment("SAM3_EXPERIMENT_MLP_SCOPE", "all");
  check_experiment(scope, {"all", "first8", "first16", "first24", "last8", "last16",
      "last24", "global", "local"}, "invalid MLP INT8 scope: ");
  return scope=="all" || (scope=="first8" && layer<8) || (scope=="first16" && layer<16) || (scope=="first24" && layer<24) || (scope=="last8" && layer>=24) || (scope=="last16" && layer>=16) || (scope=="last24" && layer>=8) || (scope=="global" && (layer+1)%8==0) || (scope=="local" && (layer+1)%8!=0);
}
inline const std::string& mlp_int8_part() {
  static const std::string part=read_experiment("SAM3_EXPERIMENT_MLP_PART", "both");
  check_experiment(part, {"both", "fc1", "fc2"}, "invalid MLP INT8 part: ");
  return part;
}
inline const std::string& int4_fc2_mode() {
  static const std::string mode=read_experiment("SAM3_EXPERIMENT_INT4_FC2");
  check_experiment(mode, {"exact", "all", "last8", "last4", "last2", "last", "w4", "a4",
      "affine", "a4_affine", "mse", "a4_mse", "w4_mse", "rht16", "rht64"},
      "invalid INT4 FC2 mode: ");
#ifndef SAM3_EXPERIMENT_INT8_GEMM
  TORCH_CHECK(mode=="exact","INT4 requires the development CUTLASS build");
#endif
  return mode;
}
inline bool int4_fc2_layer(int layer) {
  const auto& mode=int4_fc2_mode();
  return mode=="rht16" || mode=="rht64" || mode=="mse" || mode=="a4_mse" || mode=="w4_mse" || mode=="affine" || mode=="a4_affine" || mode=="w4" || mode=="a4" || mode=="all" || (mode=="last8" && layer>=24) || (mode=="last4" && layer>=28) || (mode=="last2" && layer>=30) || (mode=="last" && layer==31);
}
inline int int4_rotation(){return int4_fc2_mode()=="rht16"?16:int4_fc2_mode()=="rht64"?64:0;}
inline bool int4_weight_only(){return int4_fc2_mode()=="w4" || int4_fc2_mode()=="w4_mse";}
inline bool int4_activation_only(){return int4_fc2_mode()=="a4" || int4_fc2_mode()=="a4_affine" || int4_fc2_mode()=="a4_mse";}
inline bool int4_affine_mode(){return int4_fc2_mode()=="affine" || int4_fc2_mode()=="a4_affine" || int4_fc2_mode()=="mse" || int4_fc2_mode()=="a4_mse";}
inline bool int4_mse_mode(){return int4_fc2_mode()=="mse" || int4_fc2_mode()=="a4_mse" || int4_fc2_mode()=="w4_mse";}
} // namespace sam3::detail
