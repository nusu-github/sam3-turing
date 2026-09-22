#include "sam3/ops.h"
#include "sam3/weights.h"
#include "sam3/text_encoder.h"
#include "sam3/geometry_encoder.h"
#include "sam3/detector.h"
#include "sam3/vision_encoder.h"
#include "sam3/preprocess.h"
#include <torch/library.h>

// Dispatcher registration permits development-time parity tests via load_library.
// It does not link libtorch_python or embed a Python interpreter.
TORCH_LIBRARY(sam3_native, m) {
  m.def("detector_decode_trace(str directory, str model, Tensor memory, Tensor positions, Tensor prompt, Tensor prompt_padding, Tensor spatial_shapes, Tensor valid_ratios, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const std::string& model,const at::Tensor& memory,const at::Tensor& positions,
         const at::Tensor& prompt,const at::Tensor& prompt_padding,const at::Tensor& spatial_shapes,const at::Tensor& valid_ratios,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        std::map<std::string,at::Tensor> trace;
        sam3::DetectorDecoder(store,model,memory.device()).forward(
            {memory,at::Tensor(),positions,prompt,at::Tensor(),spatial_shapes,valid_ratios},prompt_padding,mode,&trace);
        c10::Dict<std::string,at::Tensor> result;
        for (const auto& [name,value]:trace) result.insert(name,value);
        return result;
      });
  m.def("detector_decode(str directory, str model, Tensor memory, Tensor positions, Tensor prompt, Tensor prompt_padding, Tensor? image_padding, Tensor spatial_shapes, Tensor valid_ratios, str mode) -> (Tensor, Tensor, Tensor, Tensor)",
      [](const std::string& directory,const std::string& model,const at::Tensor& memory,const at::Tensor& positions,
         const at::Tensor& prompt,const at::Tensor& prompt_padding,const std::optional<at::Tensor>& image_padding,
         const at::Tensor& spatial_shapes,const at::Tensor& valid_ratios,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const auto out=sam3::DetectorDecoder(store,model,memory.device()).forward(
            {memory,image_padding.value_or(at::Tensor()),positions,prompt,at::Tensor(),spatial_shapes,valid_ratios},prompt_padding,mode);
        return std::make_tuple(out.hidden,out.references,out.presence_logits,out.presence);
      });
  m.def("detector_encode(str directory, str model, Tensor image, Tensor positions, Tensor prompt, Tensor prompt_padding, Tensor? image_padding, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const std::string& model,const at::Tensor& image,const at::Tensor& positions,
         const at::Tensor& prompt,const at::Tensor& prompt_padding,const std::optional<at::Tensor>& image_padding,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const auto out=sam3::DetectorEncoder(store,model,image.device()).forward(image,positions,prompt,prompt_padding,image_padding.value_or(at::Tensor()),mode);
        c10::Dict<std::string,at::Tensor> tensors;
        tensors.insert("memory",out.memory);tensors.insert("pos_embed",out.positions);tensors.insert("memory_text",out.prompt);
        tensors.insert("level_start_index",out.level_start);tensors.insert("spatial_shapes",out.spatial_shapes);tensors.insert("valid_ratios",out.valid_ratios);
        if (out.padding.defined()) tensors.insert("padding_mask",out.padding);
        return tensors;
      });
  m.def("geometry_encode(str directory, str model, Tensor image, Tensor positions, Tensor points, Tensor point_labels, Tensor point_padding, Tensor boxes, Tensor box_labels, Tensor box_padding, str mode) -> (Tensor, Tensor)",
        [](const std::string& directory,const std::string& model,const at::Tensor& image,const at::Tensor& positions,
           const at::Tensor& points,const at::Tensor& point_labels,const at::Tensor& point_padding,
           const at::Tensor& boxes,const at::Tensor& box_labels,const at::Tensor& box_padding,const std::string& mode) {
          const sam3::WeightStore store(std::filesystem::u8path(directory));
          return sam3::GeometryEncoder(store,model,image.device()).forward(image,positions,
              {points,point_labels,point_padding,boxes,box_labels,box_padding},mode);
        });
  m.def("roi_align(Tensor input, Tensor rois, float spatial_scale=1., int pooled_height=7, int pooled_width=7, int sampling_ratio=-1, bool aligned=False) -> Tensor", &sam3::roi_align);
  m.def("preprocess_rgb(Tensor pixels) -> Tensor", &sam3::preprocess_rgb);
  m.def("vision_encode(str directory, str model, Tensor image, str mode, str[] heads) -> Dict(str, Tensor)",
        [](const std::string& directory, const std::string& model, const at::Tensor& image,
           const std::string& mode, const std::vector<std::string>& heads) {
          const sam3::WeightStore store(std::filesystem::u8path(directory));
          const auto result = sam3::VisionEncoder(store, model, image.device()).forward(image,mode,heads);
          c10::Dict<std::string,at::Tensor> tensors;
          tensors.insert("trunk",result.trunk);
          for (const auto& [name,pyramid] : result.pyramid)
            for (size_t i = 0; i < pyramid.size(); ++i) tensors.insert(name + "." + std::to_string(i),pyramid[i]);
          for (size_t i = 0; i < result.positions.size(); ++i) tensors.insert("position." + std::to_string(i),result.positions[i]);
          return tensors;
        });
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
