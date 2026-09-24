#include "sam3/rotary.h"
#include <c10/core/InferenceMode.h>
#include <ATen/TensorIterator.h>

namespace sam3 {
#ifdef SAM3_WITH_CUDA
void rotary_embedding_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&);
#endif
at::Tensor rotary_embedding(const at::Tensor& value, const at::Tensor& frequencies) {
  c10::InferenceMode inference;
  TORCH_CHECK(value.dim() == 4 && value.is_floating_point() &&
                  value.size(3) > 0 && value.size(3) % 2 == 0,
              "rotary values require floating [B,H,N,even D]");
  TORCH_CHECK(frequencies.dim() == 2 && frequencies.size(0) == value.size(2) &&
                  frequencies.size(1) == value.size(3) / 2 &&
                  frequencies.scalar_type() == at::kComplexFloat &&
                  frequencies.device() == value.device(),
              "rotary frequencies require complex64 [N,D/2] on the value device");
#ifdef SAM3_WITH_CUDA
  if (value.is_cuda() && value.numel() > 0 && value.stride(3) == 1 && frequencies.is_contiguous() &&
      !frequencies.is_conj() && !frequencies.is_neg() && !value.is_neg() &&
      (value.scalar_type() == at::kFloat || value.scalar_type() == at::kHalf ||
       value.scalar_type() == at::kBFloat16)) {
    // Reshaping the complex pairs normalizes singleton strides just as in the
    // original expression. TensorIterator then chooses the multiplication's
    // output layout without materializing FP32 values or a complex result.
    auto pairs = value.reshape({value.size(0), value.size(1), value.size(2),
                                value.size(3)/2, 2}).select(4, 0);
    auto iterator = at::TensorIteratorConfig()
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(value.element_size()==4 ? at::kDouble : at::kInt,
                                         value.device())
        .add_owned_output(at::Tensor())
        .add_const_input(pairs)
        .add_owned_const_input(frequencies.view({1,1,value.size(2),value.size(3)/2}))
        .build();
    auto output = iterator.output().view(value.scalar_type());
    rotary_embedding_cuda(value, frequencies, output);
    return output;
  }
#endif
  const auto complex = at::view_as_complex(value.to(at::kFloat).reshape(
      {value.size(0), value.size(1), value.size(2), value.size(3) / 2, 2}));
  return at::view_as_real(complex * frequencies.view(
      {1, 1, value.size(2), value.size(3) / 2})).flatten(3).to(value.scalar_type());
}
}
