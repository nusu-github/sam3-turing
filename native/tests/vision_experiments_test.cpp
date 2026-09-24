#include "../src/vision_experiments.h"
#include <cstdlib>
#include <iostream>

namespace {
void setting(const char* name, const char* value) {
#ifdef _WIN32
  TORCH_CHECK(_putenv_s(name, value ? value : "") == 0, "cannot set test environment");
#else
  TORCH_CHECK((value ? setenv(name, value, 1) : unsetenv(name)) == 0,
              "cannot set test environment");
#endif
}
template<class F> void rejects(F&& fn) {
  bool rejected = false;
  try { fn(); } catch (const c10::Error&) { rejected = true; }
  TORCH_CHECK(rejected, "invalid experiment was accepted");
}
}

int main(int argc, char** argv) {
  try {
    using namespace sam3::detail;
    const std::string mode = argc > 1 ? argv[1] : "valid";
    if (mode == "invalid_scope") {
      setting("SAM3_EXPERIMENT_MLP_SCOPE", "typo");
      rejects([] { mlp_int8_layer(0); });
      setting("SAM3_EXPERIMENT_MLP_SCOPE", "all");
      rejects([] { mlp_int8_layer(0); }); // Invalid first value remains cached.
    } else if (mode == "invalid_int4") {
      setting("SAM3_EXPERIMENT_INT4_FC2", "typo");
      rejects([] { int4_fc2_mode(); });
      setting("SAM3_EXPERIMENT_INT4_FC2", "exact");
      rejects([] { int4_fc2_mode(); });
    } else if (mode == "invalid_part") {
      setting("SAM3_EXPERIMENT_MLP_PART", "typo");
      rejects([] { mlp_int8_part(); });
      setting("SAM3_EXPERIMENT_MLP_PART", "both");
      rejects([] { mlp_int8_part(); });
    } else {
      TORCH_CHECK(mode == "valid", "unknown test mode");
      setting("SAM3_EXPERIMENT_MLP", nullptr);
      setting("SAM3_EXPERIMENT_MLP_PART", nullptr);
      TORCH_CHECK(mlp_int8_part()=="both","default MLP part changed");
      TORCH_CHECK(read_experiment("SAM3_EXPERIMENT_MLP") == "exact", "default changed");
      setting("SAM3_EXPERIMENT_MLP", "int8");
      TORCH_CHECK(read_experiment("SAM3_EXPERIMENT_MLP") == "int8", "live setting cached");
      setting("SAM3_EXPERIMENT_MLP", "tanh");
      TORCH_CHECK(read_experiment("SAM3_EXPERIMENT_MLP") == "tanh", "live update ignored");
      rejects([] { check_experiment("", {"exact", "fused"}, "invalid mode: "); });
      setting("SAM3_EXPERIMENT_MLP_SCOPE", "global");
      TORCH_CHECK(!mlp_int8_layer(0) && mlp_int8_layer(7), "global layer selection changed");
      setting("SAM3_EXPERIMENT_MLP_SCOPE", "all");
      TORCH_CHECK(!mlp_int8_layer(0) && mlp_int8_layer(31), "cached scope changed");
      setting("SAM3_EXPERIMENT_INT4_FC2", "exact");
      TORCH_CHECK(!int4_fc2_layer(31), "default INT4 selection changed");
      setting("SAM3_EXPERIMENT_INT4_FC2", "typo");
      TORCH_CHECK(int4_fc2_mode() == "exact", "cached INT4 mode changed");
    }
    std::cout << "PASS experiment setting lifetime: " << mode << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
