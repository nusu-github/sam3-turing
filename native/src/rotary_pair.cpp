#include "sam3/rotary_pair.h"
#include <ATen/TensorIterator.h>
#include <c10/core/InferenceMode.h>
namespace sam3 {
#ifdef SAM3_WITH_CUDA
void rotary_pair_cuda(const at::Tensor &, const at::Tensor &,
                      const at::Tensor &, at::Tensor &, at::Tensor &);
namespace {
at::Tensor allocate_output(const at::Tensor &value,
                           const at::Tensor &frequencies) {
  auto pairs = value
                   .reshape({value.size(0), value.size(1), value.size(2),
                             value.size(3) / 2, 2})
                   .select(4, 0);
  auto iterator = at::TensorIteratorConfig()
                      .check_all_same_dtype(false)
                      .declare_static_dtype_and_device(
                          value.element_size() == 4 ? at::kDouble : at::kInt,
                          value.device())
                      .add_owned_output(at::Tensor())
                      .add_const_input(pairs)
                      .add_owned_const_input(frequencies.view(
                          {1, 1, value.size(2), value.size(3) / 2}))
                      .build();
  return iterator.output().view(value.scalar_type());
}
} // namespace
#endif
std::tuple<at::Tensor, at::Tensor> rotary_embedding_pair(const at::Tensor &q,
                                                         const at::Tensor &k,
                                                         const at::Tensor &f) {
  c10::InferenceMode inference;
#ifdef SAM3_WITH_CUDA
  const bool type = q.scalar_type() == at::kFloat ||
                    q.scalar_type() == at::kHalf ||
                    q.scalar_type() == at::kBFloat16;
  if (q.is_cuda() && q.dim() == 4 && q.numel() > 0 && q.size(1) == 16 &&
      q.size(3) == 64 && type && k.sizes() == q.sizes() &&
      k.strides() == q.strides() && k.scalar_type() == q.scalar_type() &&
      k.device() == q.device() && f.device() == q.device() &&
      f.scalar_type() == at::kComplexFloat &&
      f.sizes() == at::IntArrayRef({q.size(2), 32}) && f.is_contiguous() &&
      !f.is_conj() && !f.is_neg() && !q.is_neg() && !k.is_neg() &&
      q.stride(3) == 1 && q.stride(1) == 64 && q.stride(2) == 3072 &&
      (q.size(0) == 1 || q.stride(0) == q.size(2) * 3072)) {
    auto a = allocate_output(q, f), b = allocate_output(k, f);
    const auto ordinary = [&](const at::Tensor &x) {
      return x.stride(3) == 1 && x.stride(1) == 64 && x.stride(2) == 1024 &&
             (x.size(0) == 1 || x.stride(0) == x.size(2) * 1024);
    };
    if (ordinary(a) && ordinary(b)) {
      rotary_pair_cuda(q, k, f, a, b);
      return {a, b};
    }
  }
#endif
  return {rotary_embedding(q, f), rotary_embedding(k, f)};
}
} // namespace sam3
