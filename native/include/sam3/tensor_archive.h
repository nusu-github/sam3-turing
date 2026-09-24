#pragma once
#include "sam3/weights.h"
#include <memory>
namespace sam3 {
// Temporary lossless tensor paging. Metadata stays in the owning process;
// this is not a serialized session/checkpoint interchange format. Readers have
// independent streams, verify CRCs and preserve strides (including expansion).
// The last reference deletes only this archive's exclusively reserved files.
class SAM3_NATIVE_EXPORT TensorArchive {
 public:
  static std::shared_ptr<const TensorArchive> write(const std::filesystem::path&,const std::map<std::string,at::Tensor>&);
  ~TensorArchive();
  TensorArchive(const TensorArchive&)=delete;
  TensorArchive& operator=(const TensorArchive&)=delete;
  at::Tensor read(const std::string&) const; // original device, dtype and strides
  uint64_t bytes() const {return bytes_;}
  const std::filesystem::path& path() const {return path_;}
 private:
  explicit TensorArchive(std::filesystem::path);
  struct Record {uint64_t offset,bytes;uint32_t crc;at::ScalarType dtype;at::Device device;std::vector<int64_t> shape,strides,reduced;};
  std::filesystem::path directory_,path_;
  std::map<std::string,Record> records_;
  uint64_t bytes_=0;
};
}
