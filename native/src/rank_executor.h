#pragma once
#include "sam3/video_collective.h"
#include <ATen/ThreadLocalState.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <future>

namespace sam3::detail {
// Keep the caller's stream on each device. Inputs prepared on those streams
// precede worker reads, and subsequent caller work follows worker submissions.
// This overlaps host dispatch/device execution across ranks, without creating
// independent streams or changing allocator stream ownership on one device.
template <class F>
void run_tracking_ranks(const std::vector<at::Device> &devices,
                        VideoRankExecution execution, F &&work) {
  TORCH_CHECK(execution == VideoRankExecution::Serial ||
                  execution == VideoRankExecution::Parallel,
              "invalid tracking execution policy");
  if (execution == VideoRankExecution::Serial || devices.size() <= 1) {
    for (size_t r = 0; r < devices.size(); ++r)
      work(r);
    return;
  }
  const at::ThreadLocalState state;
  std::vector<c10::Stream> streams;
  for (auto device : devices)
    streams.push_back(
        c10::impl::VirtualGuardImpl(device.type()).getStream(device));
  std::vector<std::future<void>> jobs;
  jobs.reserve(devices.size());
  for (size_t r = 0; r < devices.size(); ++r)
    jobs.push_back(std::async(std::launch::async, [&, r] {
      const at::ThreadLocalStateGuard tls(state);
      const c10::StreamGuard stream(streams[r]);
      work(r);
    }));
  // Drain every worker before caller-owned state goes away, including errors.
  // If multiple ranks fail, consistently report the lowest-index exception.
  std::exception_ptr failure;
  for (auto &job : jobs) {
    try {
      job.get();
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  }
  if (failure)
    std::rethrow_exception(failure);
}
} // namespace sam3::detail
