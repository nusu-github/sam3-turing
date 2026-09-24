#include "../src/rank_executor.h"
#include "sam3/autocast.h"
#include <ATen/Parallel.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    c10::InferenceMode inference;
    at::set_num_threads(1);
    const auto device = at::empty({0}, at::TensorOptions().device(at::Device(
                                           argc > 1 ? argv[1] : "cpu")))
                            .device();
    const c10::impl::VirtualGuardImpl impl(device.type());
    const auto stream = device.is_cuda() ? impl.getStreamFromGlobalPool(device)
                                         : impl.getStream(device);
    const c10::StreamGuard caller(stream);
    const sam3::AutocastGuard autocast(at::kCPU, true, at::kBFloat16);
    const auto input = at::arange(1024, at::TensorOptions().device(device));
    std::vector<at::Tensor> values(3);
    std::mutex mutex;
    std::condition_variable condition;
    int entered = 0;
    std::atomic<int> finished{0};
    bool failed = false;
    try {
      sam3::detail::run_tracking_ranks(
          {device, device, device}, sam3::VideoRankExecution::Parallel,
          [&](size_t rank) {
            TORCH_CHECK(c10::InferenceMode::is_enabled(), "inference TLS lost");
            TORCH_CHECK(at::autocast::is_autocast_enabled(at::kCPU) &&
                            at::autocast::get_autocast_dtype(at::kCPU) ==
                                at::kBFloat16,
                        "autocast TLS lost");
            TORCH_CHECK(impl.getStream(device) == stream, "caller stream lost");
            {
              std::unique_lock<std::mutex> lock(mutex);
              ++entered;
              condition.notify_all();
              TORCH_CHECK(condition.wait_for(lock, std::chrono::seconds(5),
                                             [&] { return entered == 3; }),
                          "rank workers did not overlap");
            }
            values[rank] = input + double(rank);
            ++finished;
            if (rank != 1)
              throw std::runtime_error("rank " + std::to_string(rank));
          });
    } catch (const std::runtime_error &e) {
      failed = std::string(e.what()) == "rank 0";
    }
    TORCH_CHECK(failed && finished == 3,
                "failure returned before draining all ranks");
    for (size_t rank = 0; rank < values.size(); ++rank)
      TORCH_CHECK(at::equal(values[rank], input + double(rank)),
                  "stream dependency lost");
    TORCH_CHECK(impl.getStream(device) == stream &&
                    c10::InferenceMode::is_enabled() &&
                    at::autocast::is_autocast_enabled(at::kCPU),
                "caller TLS changed");
    int order = 0;
    sam3::detail::run_tracking_ranks(
        {device, device, device}, sam3::VideoRankExecution::Serial,
        [&](size_t rank) {
          TORCH_CHECK(rank == size_t(order++), "serial order changed");
        });
    TORCH_CHECK(order == 3, "serial rank missing");
    std::cout << "parallel overlap, TLS, caller stream, error draining and "
                 "serial order passed on "
              << device << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
