#include "sam3/ops.h"
#include "sam3/weights.h"
#include "sam3/text_encoder.h"
#include "sam3/geometry_encoder.h"
#include "sam3/detector.h"
#include "sam3/detection_heads.h"
#include "sam3/grounding.h"
#include "sam3/image_results.h"
#include "sam3/interactive_prompt.h"
#include "sam3/interactive_decoder.h"
#include "sam3/interactive_image.h"
#include "sam3/video_heads.h"
#include "sam3/memory_encoder.h"
#include "sam3/memory_attention.h"
#include "sam3/vision_encoder.h"
#include "sam3/preprocess.h"
#include <torch/library.h>

namespace {
c10::Dict<std::string,at::Tensor> detection_dict(const sam3::DetectionOutput& out) {
  c10::Dict<std::string,at::Tensor> result;
  result.insert("pred_logits",out.logits);result.insert("pred_boxes",out.boxes);result.insert("pred_boxes_xyxy",out.boxes_xyxy);
  result.insert("pred_masks",out.masks);result.insert("semantic_seg",out.semantic);result.insert("queries",out.queries);
  result.insert("presence_logit_dec",out.presence_logits);result.insert("presence_feats",out.presence);
  return result;
}
}

// Dispatcher registration permits development-time parity tests via load_library.
// It does not link libtorch_python or embed a Python interpreter.
TORCH_LIBRARY(sam3_native, m) {
  m.def("memory_attention(str directory, str model, Tensor source, Tensor source_position, Tensor memory, Tensor memory_position, int pointers, str mode, Tensor? image, Tensor? memory_image, Tensor? memory_image_position) -> Tensor[]",
      [](const std::string& directory,const std::string& model,const at::Tensor& source,const at::Tensor& source_position,
         const at::Tensor& memory,const at::Tensor& memory_position,int64_t pointers,const std::string& mode,
         const std::optional<at::Tensor>& image,const std::optional<at::Tensor>& memory_image,const std::optional<at::Tensor>& memory_image_position) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const sam3::MemoryAttention module(store,model,source.device());std::vector<at::Tensor> trace;
        const auto out=module.forward(source,source_position,memory,memory_position,pointers,mode,image.value_or(at::Tensor()),
            memory_image.value_or(at::Tensor()),memory_image_position.value_or(at::Tensor()),&trace);
        trace.push_back(out);return trace;
      });
  m.def("memory_encode(str directory, str model, Tensor image, Tensor masks, bool skip_sigmoid, str mode) -> (Tensor, Tensor)",
      [](const std::string& directory,const std::string& model,const at::Tensor& image,const at::Tensor& masks,bool skip,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const auto out=sam3::MaskMemoryEncoder(store,model,masks.device()).forward(image,masks,skip,mode);
        return std::make_tuple(out.features,out.position);
      });
  m.def("memory_frame(str directory, str model, Tensor image, Tensor masks, Tensor scores, bool from_points, bool non_overlap, float threshold, Tensor? mux_matrix, Tensor? conditioning_objects, str mode) -> (Tensor, Tensor)",
      [](const std::string& directory,const std::string& model,const at::Tensor& image,const at::Tensor& masks,const at::Tensor& scores,
         bool points,bool non_overlap,double threshold,const std::optional<at::Tensor>& matrix,const std::optional<at::Tensor>& conditions,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const auto out=sam3::MaskMemoryEncoder(store,model,masks.device()).encode_frame(image,masks,scores,{points,non_overlap,threshold},
            matrix.value_or(at::Tensor()),conditions.value_or(at::Tensor()),mode);
        return std::make_tuple(out.features,out.position);
      });
  m.def("video_interactive_heads(str directory, str model, Tensor image, Tensor[] high, Tensor? points, Tensor? labels, Tensor? masks, bool multimask, bool direct_mask, str mode, float object_threshold=0.) -> Tensor[]",
      [](const std::string& directory,const std::string& model,const at::Tensor& image,const std::vector<at::Tensor>& high,
         const std::optional<at::Tensor>& points,const std::optional<at::Tensor>& labels,const std::optional<at::Tensor>& masks,
         bool multimask,bool direct_mask,const std::string& mode,double threshold) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const sam3::VideoInteractiveHeads heads(store,model,image.device());
        const auto out=direct_mask?heads.use_mask_as_output(image,high,masks.value(),mode,threshold):
          heads.forward(image,high,points.value_or(at::Tensor()),labels.value_or(at::Tensor()),masks.value_or(at::Tensor()),multimask,mode,threshold);
        return std::vector<at::Tensor>{out.low_res_multimasks,out.high_res_multimasks,out.iou,out.low_res_mask,out.high_res_mask,out.object_pointer,out.object_logits};
      });
  m.def("interactive_image_batch_pixels(str directory, str model, Tensor[] pixels, str mode) -> (Tensor[], Tensor[], Tensor[], Tensor)",
      [](const std::string& directory,const std::string& model,const std::vector<at::Tensor>& pixels,const std::string& mode) {
        TORCH_CHECK(!pixels.empty(),"empty image batch");
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        sam3::InteractiveImageSession session(store,model,pixels[0].device());
        { const sam3::VisionEncoder vision(store,model,pixels[0].device());session.set_images(pixels,vision,mode); }
        const auto results=session.predict_batch(std::vector<sam3::InteractiveImagePrompt>(pixels.size()));
        std::vector<at::Tensor> masks,iou,low;
        for (const auto& out:results) { masks.push_back(out.masks);iou.push_back(out.iou);low.push_back(out.low_res_logits); }
        return std::make_tuple(masks,iou,low,session.image_embedding());
      });
  m.def("interactive_image(str directory, str model, Tensor[] pyramid, int[] heights, int[] widths, int index, Tensor? points, Tensor? labels, Tensor? boxes, Tensor? masks, bool pixels, bool multimask, bool logits, float threshold, float holes, float sprinkles, str mode, bool projected_high=False) -> (Tensor, Tensor, Tensor, Tensor)",
      [](const std::string& directory,const std::string& model,const std::vector<at::Tensor>& pyramid,const std::vector<int64_t>& heights,const std::vector<int64_t>& widths,int64_t index,
         const std::optional<at::Tensor>& points,const std::optional<at::Tensor>& labels,const std::optional<at::Tensor>& boxes,const std::optional<at::Tensor>& masks,
         bool pixels,bool multimask,bool logits,double threshold,double holes,double sprinkles,const std::string& mode,bool projected_high) {
        TORCH_CHECK(!pyramid.empty(),"empty image pyramid");
        const sam3::WeightStore store(std::filesystem::u8path(directory));sam3::InteractiveImageSession session(store,model,pyramid[0].device());
        session.set_features(pyramid,heights,widths,mode,projected_high);
        const auto out=session.predict(index,{points.value_or(at::Tensor()),labels.value_or(at::Tensor()),boxes.value_or(at::Tensor()),masks.value_or(at::Tensor()),pixels},
            {multimask,logits,threshold,holes,sprinkles});
        return std::make_tuple(out.masks,out.iou,out.low_res_logits,session.image_embedding());
      });
  m.def("interactive_postprocess(Tensor masks, int height, int width, float threshold, float holes, float sprinkles) -> Tensor",
      [](const at::Tensor& masks,int64_t height,int64_t width,double threshold,double holes,double sprinkles) {
        return sam3::postprocess_interactive_masks(masks,height,width,{true,true,threshold,holes,sprinkles});
      });
  m.def("interactive_decode(str directory, str model, Tensor image, Tensor sparse, Tensor dense, Tensor position, Tensor[] high, bool project_high, bool multimask, bool repeat_image, str mode, bool dynamic_stability, float delta, float threshold) -> Tensor[]",
      [](const std::string& directory,const std::string& model,const at::Tensor& image,const at::Tensor& sparse,const at::Tensor& dense,const at::Tensor& position,
         const std::vector<at::Tensor>& high,bool project_high,bool multimask,bool repeat_image,const std::string& mode,bool dynamic_stability,double delta,double threshold) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const sam3::InteractiveMaskDecoder decoder(store,model,image.device());
        const auto out=decoder.forward(image,{sparse,dense,position},project_high?decoder.project_pyramid(high,mode):high,multimask,repeat_image,mode,dynamic_stability,delta,threshold);
        return std::vector<at::Tensor>{out.masks,out.iou,out.tokens,out.object_logits,out.all_masks,out.all_iou,out.all_tokens};
      });
  m.def("interactive_prompt(str directory, str model, str device, Tensor? points, Tensor? labels, Tensor? boxes, Tensor? masks, int[] grid, int[] input_size, str mode) -> (Tensor, Tensor, Tensor)",
      [](const std::string& directory,const std::string& model,const std::string& device,const std::optional<at::Tensor>& points,
         const std::optional<at::Tensor>& labels,const std::optional<at::Tensor>& boxes,const std::optional<at::Tensor>& masks,
         const std::vector<int64_t>& grid,const std::vector<int64_t>& input_size,const std::string& mode) {
        TORCH_CHECK(grid.size()==2 && input_size.size()==2,"expected two spatial dimensions");
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const auto out=sam3::InteractivePromptEncoder(store,model,at::Device(device),{grid[0],grid[1]},{input_size[0],input_size[1]}).forward(
            {points.value_or(at::Tensor()),labels.value_or(at::Tensor()),boxes.value_or(at::Tensor()),masks.value_or(at::Tensor())},mode);
        return std::make_tuple(out.sparse,out.dense,out.position);
      });
  m.def("postprocess_image(Tensor boxes, Tensor logits, Tensor masks, Tensor presence, int[] heights, int[] widths, float threshold, bool combine_presence, int chunk_size, str mode=\"fp32\") -> (Tensor[], Tensor[], Tensor[], Tensor[], Tensor[])",
      [](const at::Tensor& boxes,const at::Tensor& logits,const at::Tensor& masks,const at::Tensor& presence,
         const std::vector<int64_t>& heights,const std::vector<int64_t>& widths,double threshold,bool combine_presence,int64_t chunk_size,const std::string& mode) {
        sam3::DetectionOutput input;input.boxes=boxes;input.logits=logits;input.masks=masks;input.presence_logits=presence;
        const auto results=sam3::postprocess_image(input,heights,widths,threshold,combine_presence,chunk_size,mode);
        std::vector<at::Tensor> out_boxes,out_scores,out_probabilities,out_masks,out_indices;
        for (const auto& result:results) {
          out_boxes.push_back(result.boxes);out_scores.push_back(result.scores);out_probabilities.push_back(result.mask_probabilities);
          out_masks.push_back(result.masks);out_indices.push_back(result.query_indices);
        }
        return std::make_tuple(out_boxes,out_scores,out_probabilities,out_masks,out_indices);
      });
  m.def("grounding(str directory, str model, Tensor[] pyramid, Tensor positions, Tensor image_ids, Tensor text_ids, Tensor text_features, Tensor text_padding, Tensor points, Tensor point_labels, Tensor point_padding, Tensor boxes, Tensor box_labels, Tensor box_padding, Tensor? visual_features, Tensor? visual_padding, Tensor? previous_mask, bool use_text, bool joint_scores, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const std::string& model,const std::vector<at::Tensor>& pyramid,const at::Tensor& positions,
         const at::Tensor& image_ids,const at::Tensor& text_ids,const at::Tensor& text_features,const at::Tensor& text_padding,
         const at::Tensor& points,const at::Tensor& point_labels,const at::Tensor& point_padding,const at::Tensor& boxes,
         const at::Tensor& box_labels,const at::Tensor& box_padding,const std::optional<at::Tensor>& visual_features,
         const std::optional<at::Tensor>& visual_padding,const std::optional<at::Tensor>& previous_mask,bool use_text,bool joint_scores,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        sam3::GroundingPrompt prompt{image_ids,text_ids,text_features,text_padding,{points,point_labels,point_padding,boxes,box_labels,box_padding},
            visual_features.value_or(at::Tensor()),visual_padding.value_or(at::Tensor()),previous_mask.value_or(at::Tensor()),use_text};
        const auto out=sam3::GroundingDetector(store,model,positions.device()).forward(pyramid,positions,prompt,joint_scores,mode);
        auto result=detection_dict(out.detection);result.insert("encoder_hidden_states",out.encoded.memory);
        return result;
      });
  m.def("detection_heads(str directory, str model, Tensor[] pyramid, Tensor image_ids, Tensor memory, Tensor prompt, Tensor prompt_padding, Tensor hidden, Tensor references, Tensor presence_logits, Tensor presence, bool joint_scores, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const std::string& model,const std::vector<at::Tensor>& pyramid,const at::Tensor& image_ids,
         const at::Tensor& memory,const at::Tensor& prompt,const at::Tensor& prompt_padding,const at::Tensor& hidden,
         const at::Tensor& references,const at::Tensor& presence_logits,const at::Tensor& presence,bool joint_scores,const std::string& mode) {
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        sam3::FusionFeatures encoded;encoded.memory=memory;encoded.prompt=prompt;
        const auto out=sam3::DetectionHeads(store,model,memory.device()).forward(pyramid,image_ids,encoded,prompt_padding,
            {hidden,references,presence_logits,presence},joint_scores,mode);
        return detection_dict(out);
      });
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
  m.def("text_encode(str directory, str model, Tensor tokens, str mode=\"fp32\") -> (Tensor, Tensor, Tensor)",
        [](const std::string& directory, const std::string& model, const at::Tensor& tokens,const std::string& mode) {
          const sam3::WeightStore store(std::filesystem::u8path(directory));
          return sam3::TextEncoder(store, model, tokens.device()).forward(tokens,mode);
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
