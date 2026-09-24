#include "sam3/weights.h"
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc >= 3 && argc <= 5, "usage: sam3_weights STORE list|verify [PREFIX] [cpu|cuda]");
    const std::string command = argv[2];
    TORCH_CHECK(command == "list" || command == "verify", "invalid command");
    const std::string prefix = argc > 3 ? argv[3] : "";
    const at::Device device(argc > 4 ? argv[4] : "cpu");
    const auto start = std::chrono::steady_clock::now();
    c10::InferenceMode guard;
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    uint64_t count = 0, bytes = 0;
    for (const auto& [name, record] : store.records()) {
      if (name.compare(0, prefix.size(), prefix) != 0) continue;
      ++count;
      bytes += record.bytes;
      if (command == "list") {
        std::cout << name << ' ' << c10::toString(record.dtype) << ' ' << record.bytes << ' ' << record.shard << '\n';
      } else {
        // Release each tensor before loading the next; verification never needs
        // all model weights resident at once. read() verifies payload CRC32.
        const auto tensor = store.read(name, device);
        TORCH_CHECK(static_cast<uint64_t>(tensor.nbytes()) == record.bytes, "unexpected loaded byte count");
      }
    }
    TORCH_CHECK(count, "no matching weights");
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "tensors=" << count << " logical_bytes=" << bytes << " seconds=" << seconds << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
