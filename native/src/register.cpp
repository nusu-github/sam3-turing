#include "sam3/ops.h"
#include "sam3/weights.h"
#include "sam3/text_encoder.h"
#include <torch/library.h>

// Dispatcher registration permits development-time parity tests via load_library.
// It does not link libtorch_python or embed a Python interpreter.
TORCH_LIBRARY(sam3_native, m) {
  m.def("text_encode(str directory, str model, Tensor tokens) -> (Tensor, Tensor, Tensor)",
        [](const std::string& directory, const std::string& model, const at::Tensor& tokens) {
          const sam3::WeightStore store(std::filesystem::u8path(directory));
          return sam3::TextEncoder(store, model, tokens.device()).forward(tokens);
        });
  m.def("read_weight(str directory, str name, str device=\"cpu\") -> Tensor",
        [](const std::string& directory, const std::string& name, const std::string& device) {
          return sam3::WeightStore(std::filesystem::u8path(directory)).read(name, at::Device(device));
        });
  m.def("connected_components(Tensor masks) -> (Tensor, Tensor)", &sam3::connected_components);
  m.def("euclidean_distance_transform(Tensor masks) -> Tensor", &sam3::euclidean_distance_transform);
  m.def("pack_masks(Tensor masks) -> Tensor", &sam3::pack_masks);
  m.def("unpack_masks(Tensor packed, int height, int width) -> Tensor", &sam3::unpack_masks);
  m.def("resize_and_pack_masks(Tensor logits, int height, int width, int chunk_size=8) -> Tensor", &sam3::resize_and_pack_masks);
  m.def("generic_nms(Tensor ious, Tensor scores, float threshold) -> Tensor", &sam3::generic_nms);
}
