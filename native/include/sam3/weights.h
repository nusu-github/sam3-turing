#pragma once
#include <ATen/ATen.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include "sam3_native_export.h"

namespace sam3 {
struct TensorRecord {
  std::string shard;
  uint64_t offset;
  uint64_t bytes;
  uint32_t crc32;
  at::ScalarType dtype;
  std::vector<int64_t> shape;
};

// Opens metadata only. Tensor storage is allocated on read(), so callers control
// module lifetime and device placement. No Python, mmap, or OS-specific loader.
class SAM3_NATIVE_EXPORT WeightStore {
 public:
  explicit WeightStore(const std::filesystem::path& directory);
  const std::map<std::string, TensorRecord>& records() const { return records_; }
  at::Tensor read(const std::string& name, at::Device device = at::kCPU) const;
  std::map<std::string, at::Tensor> read_prefix(
      const std::string& prefix, at::Device device = at::kCPU) const;
 private:
  std::filesystem::path directory_;
  std::map<std::string, TensorRecord> records_;
};
}
