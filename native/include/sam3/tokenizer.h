#pragma once
#include "sam3_native_export.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace sam3 {
// Arbitrary UTF-8 input, source ftfy 6.1.1 cleaning and VE byte-level BPE.
// Unicode implementation details are hidden from callers. No Torch/Python API
// is needed to tokenize; output IDs can be supplied to TextEncoder.
class SAM3_NATIVE_EXPORT Tokenizer {
 public:
  explicit Tokenizer(const std::filesystem::path& gzip_vocabulary);
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  std::string clean(const std::string& utf8) const;
  std::vector<int64_t> encode(const std::string& utf8) const;
  std::vector<std::vector<int64_t>> tokenize(const std::vector<std::string>& texts,int64_t context=32) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
