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
    } else if (mode == "invalid_projection_scope") {
      setting("SAM3_EXPERIMENT_PROJECTION_SCOPE", "first33");
      rejects([] { projection_int8_layer(0); });
      setting("SAM3_EXPERIMENT_PROJECTION_SCOPE", "all");
      rejects([] { projection_int8_layer(0); });
    } else if (mode == "invalid_attention_scope") {
      setting("SAM3_EXPERIMENT_ATTENTION_SCOPE", "mask:0xz");
      rejects([] { attention_int8_layer(0); });
      setting("SAM3_EXPERIMENT_ATTENTION_SCOPE", "all");
      rejects([] { attention_int8_layer(0); });
    } else if (mode == "invalid_kitchen_center") {
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER", "typo");
      rejects([] { kitchen_center(); });
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER", "anchor");
      rejects([] { kitchen_center(); });
    } else if (mode == "center_scope") {
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER", "mean_half");
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER_SCOPE", "global");
      TORCH_CHECK(kitchen_center_for_length(5184)=="mean_half" && kitchen_center_for_length(576)=="anchor","centering scope mismatch");
      rejects([]{kitchen_center_for_length(129);});
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER_SCOPE", "local");
      TORCH_CHECK(kitchen_center_for_length(576)=="anchor","centering scope cache changed");
    } else if (mode == "invalid_center_scope") {
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER_SCOPE", "typo");
      rejects([] { kitchen_center_for_length(576); });
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER_SCOPE", "all");
      rejects([] { kitchen_center_for_length(576); });
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
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER", nullptr);
      TORCH_CHECK(kitchen_center()=="anchor","default kitchen centering changed");
      setting("SAM3_EXPERIMENT_KITCHEN_CENTER", "mean");
      TORCH_CHECK(kitchen_center()=="anchor","kitchen centering cache changed");
      for(int count=1;count<=32;++count) for(int layer=0;layer<32;++layer) {
        TORCH_CHECK(quant_mask_layer(quant_layer_mask("first"+std::to_string(count)),layer)==(layer<count),"first scope mismatch");
        TORCH_CHECK(quant_mask_layer(quant_layer_mask("last"+std::to_string(count)),layer)==(layer>=32-count),"last scope mismatch");
        TORCH_CHECK(quant_mask_layer(quant_layer_mask("global"),layer)==((layer+1)%8==0),"global mask mismatch");
        TORCH_CHECK(quant_mask_layer(quant_layer_mask("local"),layer)==((layer+1)%8!=0),"local mask mismatch");
      }
      TORCH_CHECK(quant_layer_mask("mask:0x0")==0 && quant_layer_mask("mask:0xFFFFFFFF")==0xffffffffu &&
          quant_layer_mask("mask:0x80000001")==0x80000001u,"explicit layer mask mismatch");
      for(const std::string bad:{"", "first", "first0", "first33", "last-1", "last1x", "mask:0x", "mask:0x100000000", "mask:0x1g"})
        rejects([&]{quant_layer_mask(bad);});
      rejects([]{quant_mask_layer(1,-1);});rejects([]{quant_mask_layer(1,32);});
      for(int layer=0;layer<32;++layer)
        TORCH_CHECK(attention_block_index("trunk.blocks."+std::to_string(layer)+".attn")==layer,"attention prefix mismatch");
      for(const std::string bad:{"", "trunk.blocks..attn", "trunk.blocks.-1.attn", "trunk.blocks.32.attn",
          "trunk.blocks.123.attn", "trunk.blocks.1x.attn", "trunk.blocks.1.attn.qkv", "blocks.1.attn"})
        rejects([&]{attention_block_index(bad);});
      setting("SAM3_EXPERIMENT_ATTENTION_SCOPE", "mask:0xffffffbf");
      for(int layer=0;layer<32;++layer) TORCH_CHECK(attention_int8_layer(layer)==(layer!=6),"attention scope mismatch");
      setting("SAM3_EXPERIMENT_ATTENTION_SCOPE", "all");
      TORCH_CHECK(!attention_int8_layer(6),"attention scope cache changed");
      setting("SAM3_EXPERIMENT_PROJECTION_SCOPE", "first24");
      TORCH_CHECK(projection_int8_layer(23) && !projection_int8_layer(24),"projection scope mismatch");
      setting("SAM3_EXPERIMENT_PROJECTION_SCOPE", "all");
      TORCH_CHECK(!projection_int8_layer(31),"projection scope cache changed");
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
