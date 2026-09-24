#pragma once
// Internal experiment selection shared by weight preparation and block execution.
// Preserve first-use environment caching and validation from the original encoder.
#include <c10/util/Exception.h>
#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <cstdlib>
#include <cstdint>
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

// Bit 0 is block 0. Explicit masks permit single-block counterfactuals without
// rebuilding or relying on an image-dependent precision policy.
inline uint32_t quant_layer_mask(const std::string& scope) {
  constexpr uint32_t all=0xffffffffu,global=0x80808080u;
  if(scope=="all")return all;
  if(scope=="global")return global;
  if(scope=="local")return all^global;
  const bool first=scope.rfind("first",0)==0,last=scope.rfind("last",0)==0;
  if(first || last) {
    const auto digits=scope.substr(first?5:4);
    TORCH_CHECK(!digits.empty() && digits.size()<=2 &&
        std::all_of(digits.begin(),digits.end(),[](char c){return c>='0' && c<='9';}),
        "invalid quantization layer scope: ",scope);
    const int count=std::stoi(digits);
    TORCH_CHECK(count>=1 && count<=32,"quantization layer count must be in [1,32]");
    return first?all>>(32-count):all<<(32-count);
  }
  if(scope.rfind("mask:0x",0)==0) {
    const auto digits=scope.substr(7);
    TORCH_CHECK(!digits.empty() && digits.size()<=8,"invalid 32-bit layer mask: ",scope);
    uint32_t mask=0;
    for(char c:digits) {
      const int digit=c>='0' && c<='9'?c-'0':c>='a' && c<='f'?c-'a'+10:c>='A' && c<='F'?c-'A'+10:-1;
      TORCH_CHECK(digit>=0,"invalid layer mask digit: ",scope);
      mask=(mask<<4)|uint32_t(digit);
    }
    return mask;
  }
  TORCH_CHECK(false,"invalid quantization layer scope: ",scope);
}
inline bool quant_mask_layer(uint32_t mask,int layer) {
  TORCH_CHECK(layer>=0 && layer<32,"quantization block index outside [0,31]");
  return (mask & (uint32_t(1)<<layer))!=0;
}
inline bool mlp_int8_layer(int layer) {
  static const std::string scope=read_experiment("SAM3_EXPERIMENT_MLP_SCOPE", "all");
  static const uint32_t mask=quant_layer_mask(scope);
  return quant_mask_layer(mask,layer);
}
inline bool projection_int8_layer(int layer) {
  static const std::string scope=read_experiment("SAM3_EXPERIMENT_PROJECTION_SCOPE", "all");
  static const uint32_t mask=quant_layer_mask(scope);
  return quant_mask_layer(mask,layer);
}
inline bool attention_int8_layer(int layer) {
  static const std::string scope=read_experiment("SAM3_EXPERIMENT_ATTENTION_SCOPE", "all");
  static const uint32_t mask=quant_layer_mask(scope);
  return quant_mask_layer(mask,layer);
}
// Attention keeps its existing private ABI; its internal weight prefix carries
// the block index. Reject malformed prefixes instead of partially parsing them.
inline int attention_block_index(const std::string& prefix) {
  constexpr std::string_view start="trunk.blocks.",end=".attn";
  TORCH_CHECK(prefix.size()>start.size()+end.size() && prefix.compare(0,start.size(),start)==0 &&
      prefix.compare(prefix.size()-end.size(),end.size(),end)==0,"invalid attention block prefix: ",prefix);
  const auto digits=prefix.substr(start.size(),prefix.size()-start.size()-end.size());
  TORCH_CHECK(digits.size()<=2 && std::all_of(digits.begin(),digits.end(),[](char c){return c>='0' && c<='9';}),
      "invalid attention block index: ",prefix);
  const int layer=std::stoi(digits);
  TORCH_CHECK(layer<32,"attention block index outside [0,31]");
  return layer;
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
