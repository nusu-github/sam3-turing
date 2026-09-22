#include "sam3/weights.h"
#include <array>
#include <fstream>
#include <limits>

namespace sam3 {
namespace {
void read_exact(std::istream& file, void* target, uint64_t size) {
  auto* data = static_cast<char*>(target);
  while (size) {
    const auto chunk = static_cast<std::streamsize>(std::min<uint64_t>(size, 1ULL << 30));
    TORCH_CHECK(file.read(data, chunk), "truncated weight store");
    data += chunk;
    size -= chunk;
  }
}
uint64_t read_uint(std::istream& file, int bytes) {
  uint8_t data[8]{};
  read_exact(file, data, bytes);
  uint64_t value = 0;
  for (int i = 0; i < bytes; ++i) value |= static_cast<uint64_t>(data[i]) << (8 * i);
  return value;
}
std::string read_string(std::istream& file) {
  const auto length = read_uint(file, 4);
  TORCH_CHECK(length > 0 && length <= 65536, "invalid weight index string length");
  std::string result(length, '\0');
  read_exact(file, result.data(), length);
  TORCH_CHECK(result.find('\0') == std::string::npos, "NUL in weight index string");
  return result;
}
at::ScalarType scalar_type(uint64_t code) {
  switch (code) {
    case 0: return at::kBool;
    case 1: return at::kByte;
    case 2: return at::kChar;
    case 3: return at::kShort;
    case 4: return at::kInt;
    case 5: return at::kLong;
    case 6: return at::kHalf;
    case 7: return at::kBFloat16;
    case 8: return at::kFloat;
    case 9: return at::kDouble;
    case 10: return at::kComplexFloat;
    case 11: return at::kComplexDouble;
    default: TORCH_CHECK(false, "unsupported weight dtype code: ", code);
  }
}
uint32_t checksum(const void* data, uint64_t length) {
  static const auto table = [] {
    std::array<uint32_t, 256> values{};
    for (uint32_t i = 0; i < 256; ++i) {
      auto c = i;
      for (int k = 0; k < 8; ++k) c = (c >> 1) ^ ((c & 1) ? 0xedb88320U : 0);
      values[i] = c;
    }
    return values;
  }();
  auto c = 0xffffffffU;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (uint64_t i = 0; i < length; ++i) c = table[(c ^ bytes[i]) & 255] ^ (c >> 8);
  return c ^ 0xffffffffU;
}
}

WeightStore::WeightStore(const std::filesystem::path& directory) : directory_(directory) {
  const uint16_t endian = 1;
  TORCH_CHECK(*reinterpret_cast<const uint8_t*>(&endian) == 1, "weight store requires a little-endian host");
  std::ifstream file(directory_ / "weights.s3i", std::ios::binary);
  TORCH_CHECK(file, "cannot open weight index");
  char magic[8];
  read_exact(file, magic, 8);
  TORCH_CHECK(std::string(magic, 8) == "SAM3WGT1", "invalid weight index magic/version");
  const auto count = read_uint(file, 8);
  const auto index_size = std::filesystem::file_size(directory_ / "weights.s3i");
  TORCH_CHECK(count <= (index_size - 16) / 38, "invalid weight index record count");
  for (uint64_t i = 0; i < count; ++i) {
    const auto name = read_string(file);
    TensorRecord record;
    record.shard = read_string(file);
    TORCH_CHECK(record.shard != "." && record.shard != ".." &&
      record.shard.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos,
      "invalid shard filename");
    record.offset = read_uint(file, 8);
    record.bytes = read_uint(file, 8);
    record.crc32 = static_cast<uint32_t>(read_uint(file, 4));
    record.dtype = scalar_type(read_uint(file, 4));
    const auto rank = read_uint(file, 4);
    TORCH_CHECK(rank <= 64, "invalid tensor rank");
    record.shape.reserve(rank);
    bool empty = false;
    for (uint64_t d = 0; d < rank; ++d) {
      const auto dim = read_uint(file, 8);
      TORCH_CHECK(dim <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "invalid tensor dimension");
      record.shape.push_back(static_cast<int64_t>(dim));
      empty |= dim == 0;
    }
    uint64_t expected = empty ? 0 : c10::elementSize(record.dtype);
    if (!empty) for (const auto dim : record.shape) {
      TORCH_CHECK(expected <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dim,
                  "tensor byte size overflow");
      expected *= dim;
    }
    TORCH_CHECK(expected == record.bytes, "tensor shape/byte count mismatch");
    TORCH_CHECK(record.offset % 64 == 0, "unaligned tensor offset");
    TORCH_CHECK(record.offset <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) &&
                record.bytes <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) - record.offset,
                "tensor offset overflow");
    TORCH_CHECK(records_.emplace(name, std::move(record)).second, "duplicate tensor name: ", name);
  }
  TORCH_CHECK(file.peek() == std::char_traits<char>::eof(), "trailing weight index data");
}

at::Tensor WeightStore::read(const std::string& name, at::Device device) const {
  const auto found = records_.find(name);
  TORCH_CHECK(found != records_.end(), "unknown tensor: ", name);
  const auto& record = found->second;
  const auto filename = directory_ / record.shard;
  const auto file_size = std::filesystem::file_size(filename);
  TORCH_CHECK(record.offset <= file_size && record.bytes <= file_size - record.offset,
              "truncated weight shard: ", record.shard);
  std::ifstream file(filename, std::ios::binary);
  TORCH_CHECK(file, "cannot open weight shard: ", record.shard);
  TORCH_CHECK(file.seekg(static_cast<std::streamoff>(record.offset)), "cannot seek weight shard");
  auto tensor = at::empty(record.shape, at::TensorOptions().dtype(record.dtype).device(at::kCPU));
  read_exact(file, tensor.mutable_data_ptr(), record.bytes);
  TORCH_CHECK(checksum(tensor.const_data_ptr(), record.bytes) == record.crc32,
              "weight checksum mismatch: ", name);
  return device.is_cpu() ? tensor : tensor.to(device);
}

std::map<std::string, at::Tensor> WeightStore::read_prefix(const std::string& prefix, at::Device device) const {
  std::map<std::string, at::Tensor> tensors;
  for (auto it = records_.lower_bound(prefix); it != records_.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it)
    tensors.emplace(it->first, read(it->first, device));
  TORCH_CHECK(!tensors.empty(), "no tensors under prefix: ", prefix);
  return tensors;
}
}
