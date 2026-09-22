#include "sam3/ops.h"
#include <torch/library.h>

// Dispatcher registration permits development-time parity tests via load_library.
// It does not link libtorch_python or embed a Python interpreter.
TORCH_LIBRARY(sam3_native, m) {
  m.def("pack_masks(Tensor masks) -> Tensor", &sam3::pack_masks);
  m.def("unpack_masks(Tensor packed, int height, int width) -> Tensor", &sam3::unpack_masks);
  m.def("resize_and_pack_masks(Tensor logits, int height, int width, int chunk_size=8) -> Tensor", &sam3::resize_and_pack_masks);
  m.def("generic_nms(Tensor ious, Tensor scores, float threshold) -> Tensor", &sam3::generic_nms);
}
