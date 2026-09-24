#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cub/device/device_transform.cuh>
namespace sam3 {
namespace {
using Index = unsigned long long;
constexpr int threads = 256;
__device__ Index read_atomic(Index* p) { return atomicCAS(p, 0ULL, 0ULL); }
__device__ Index root_atomic(Index* parents, Index i) {
  auto next = read_atomic(parents + i);
  while (next != i) {
    const auto grandparent = read_atomic(parents + next);
    atomicCAS(parents + i, next, grandparent);
    i = next;
    next = grandparent;
  }
  return i;
}
__device__ void merge(Index* parents, Index a, Index b) {
  while (true) {
    a = root_atomic(parents, a);
    b = root_atomic(parents, b);
    if (a == b) return;
    const auto low = a < b ? a : b, high = a < b ? b : a;
    if (atomicCAS(parents + high, high, low) == high) return;
  }
}
__global__ void initialize(const int64_t* input, Index* parents, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x)
    parents[i] = input[i] ? static_cast<Index>(i) : ~0ULL;
}
__global__ void union_neighbors(const int64_t* input, Index* parents, int64_t n, int64_t h, int64_t w) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const auto value = input[i];
    if (!value) continue;
    const auto x = i % w, y = (i / w) % h;
    if (x && input[i - 1] == value) merge(parents, i, i - 1);
    if (y) {
      if (x && input[i - w - 1] == value) merge(parents, i, i - w - 1);
      if (input[i - w] == value) merge(parents, i, i - w);
      if (x + 1 < w && input[i - w + 1] == value) merge(parents, i, i - w + 1);
    }
  }
}
__global__ void count_components(const Index* parents, int64_t* labels, Index* hist, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (parents[i] == ~0ULL) { labels[i] = 0; continue; }
    Index root = i;
    while (parents[root] != root) root = parents[root];
    labels[i] = static_cast<int64_t>(root) + 1;
    atomicAdd(hist + root, 1ULL);
  }
}
struct GatherSize {
  const int64_t* histogram;
  __device__ int64_t operator()(int64_t label) const {
    return label ? histogram[label - 1] : 0;
  }
};
}
std::tuple<at::Tensor, at::Tensor> components_cuda(const at::Tensor& values) {
  static_assert(sizeof(Index) == sizeof(int64_t), "64-bit atomic indices required");
  const c10::cuda::CUDAGuard guard(values.device());
  auto parents = at::empty_like(values);
  auto labels = at::empty_like(values);
  auto histogram = at::zeros_like(values);
  auto sizes = at::empty_like(values);
  const auto n = values.numel();
  if (!n) return {labels, sizes};
  const auto blocks = static_cast<int>(std::min<int64_t>((n - 1) / threads + 1, 4096));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  auto* p = reinterpret_cast<Index*>(parents.mutable_data_ptr<int64_t>());
  auto* hist = reinterpret_cast<Index*>(histogram.mutable_data_ptr<int64_t>());
  initialize<<<blocks, threads, 0, stream>>>(values.const_data_ptr<int64_t>(), p, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  union_neighbors<<<blocks, threads, 0, stream>>>(values.const_data_ptr<int64_t>(), p, n, values.size(-2), values.size(-1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  count_components<<<blocks, threads, 0, stream>>>(p, labels.mutable_data_ptr<int64_t>(), hist, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cub::DeviceTransform::Transform(labels.const_data_ptr<int64_t>(),
      sizes.mutable_data_ptr<int64_t>(), n, GatherSize{histogram.const_data_ptr<int64_t>()}, stream));
  return {labels, sizes};
}
}
