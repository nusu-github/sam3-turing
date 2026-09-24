#include "sam3/weights.h"
#include <chrono>
#include <fstream>
#include <iostream>

namespace {
void uint_le(std::ostream& out, uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
}
void string_le(std::ostream& out, const std::string& value) {
  uint_le(out, value.size(), 4);
  out.write(value.data(), value.size());
}
void index(const std::filesystem::path& root, uint32_t crc = 0xb63cfbcdU, uint64_t bytes = 4, std::string shard = "test.s3w") {
  std::ofstream out(root / "weights.s3i", std::ios::binary);
  out.write("SAM3WGT1",8);
  uint_le(out, 1, 8);
  string_le(out, "test/model.weight");
  string_le(out, shard);
  uint_le(out, 0, 8);
  uint_le(out, bytes, 8);
  uint_le(out, crc, 4);
  uint_le(out, 1, 4);  // uint8
  uint_le(out, 2, 4);  // rank
  uint_le(out, 2, 8);
  uint_le(out, 2, 8);
}
template<class F> void must_fail(F function) {
  bool failed = false;
  try { function(); } catch (const std::exception&) { failed = true; }
  TORCH_CHECK(failed, "invalid weight store was accepted");
}
struct Temp {
  std::filesystem::path root;
  Temp() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int i = 0; i < 100; ++i) {
      root = std::filesystem::temp_directory_path() / ("sam3-weight-test-" + std::to_string(nonce) + "-" + std::to_string(i));
      if (std::filesystem::create_directory(root)) return;
    }
    throw std::runtime_error("cannot create test directory");
  }
  ~Temp() { std::error_code error; std::filesystem::remove_all(root, error); }
};
}
int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 2, "expected cpu|cuda");
    Temp temporary;
    const auto device = at::Device(argv[1]);
    const auto& root = temporary.root;
    {
      std::ofstream data(root / "test.s3w", std::ios::binary);
      data.write("\1\2\3\4",4);
    }
    index(root);  // CRC32(01 02 03 04) = b63cfbcd
    const sam3::WeightStore store(root);
    TORCH_CHECK(store.records().size() == 1, "index mismatch");
    const auto actual = store.read("test/model.weight",device).cpu();
    TORCH_CHECK(at::equal(actual, at::tensor({1,2,3,4}).to(at::kByte).reshape({2,2})), "weight values mismatch");
    TORCH_CHECK(store.read_prefix("test/",device).size() == 1, "prefix mismatch");
    must_fail([&] { store.read("missing"); });
    must_fail([&] { store.read_prefix("missing/"); });
    index(root,0);
    must_fail([&] { sam3::WeightStore(root).read("test/model.weight"); });
    index(root,0,3);
    must_fail([&] { sam3::WeightStore invalid(root); });
    index(root,0,4,"../test.s3w");
    must_fail([&] { sam3::WeightStore invalid(root); });
    index(root);
    std::filesystem::resize_file(root / "test.s3w", 3);
    must_fail([&] { sam3::WeightStore(root).read("test/model.weight"); });
    std::filesystem::resize_file(root / "weights.s3i", 20);
    must_fail([&] { sam3::WeightStore invalid(root); });
    std::cout << "PASS: native weight loading, device placement, checksum and malformed data checks on " << argv[1] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
