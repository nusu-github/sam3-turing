#include "sam3/multiplex_history.h"
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(sam3_native,m) {
  m.def("multiplex_history_encode(str directory, Tensor image, Tensor masks, Tensor scores, int[][] assignments, int[] conditions, bool overlap, str device, str mode) -> Tensor[]",
      [](const std::string& directory,const at::Tensor& image,const at::Tensor& masks,const at::Tensor& scores,const std::vector<std::vector<int64_t>>& assignments,const std::vector<int64_t>& conditions,bool overlap,const std::string& target,const std::string& mode) {
        const at::Device device(target);const sam3::Sam31TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(directory)),device);
        sam3::MultiplexFrame frame;frame.image=image;frame.masks.high_res_mask=masks;frame.masks.object_logits=scores;frame.conditioning_objects=conditions;
        const sam3::MultiplexState state(assignments,device,at::kFloat,16);sam3::MultiplexFrameOptions options;options.non_overlap_memory=overlap;
        const auto output=core.encode_history(frame,state,options,mode);return std::vector<at::Tensor>{output.features,output.position};
      });
  m.def("multiplex_history(int[][] old_slots, int[] old_ids, int[][] new_slots, int[] new_ids, Dict(str, Tensor) tensors, int[] conditions, Tensor? rebuilt_memory, Tensor? rebuilt_position, str mode) -> Dict(str, Tensor)",
      [](const std::vector<std::vector<int64_t>>& old_slots,const std::vector<int64_t>& old_ids,const std::vector<std::vector<int64_t>>& new_slots,const std::vector<int64_t>& new_ids,
         const c10::Dict<std::string,at::Tensor>& tensors,const std::vector<int64_t>& conditions,const std::optional<at::Tensor>& rebuilt_memory,const std::optional<at::Tensor>& rebuilt_position,const std::string& mode) {
        const auto get=[&](const std::string& key){return tensors.contains(key)?tensors.at(key):at::Tensor();};const auto device=get("pointer").device();
        sam3::MultiplexState old_state(old_slots,device,at::kFloat,16,old_ids),new_state(new_slots,device,at::kFloat,16,new_ids);
        sam3::MultiplexFrame frame;frame.masks.low_res_mask=get("low");frame.masks.high_res_mask=get("high");frame.masks.object_logits=get("logits");frame.pointer=get("pointer");frame.iou=get("iou");frame.confidence=get("confidence");frame.image=get("image");frame.image_position=get("image_position");frame.memory=get("memory");frame.memory_position=get("position");frame.conditioning_objects=conditions;
        frame.masks.low_res_multimasks=get("candidates");sam3::MultiplexFrameHistory history;history.conditioning.push_back(frame);int64_t calls=0;
        sam3::MultiplexHistoryRebuilder rebuild;if(rebuilt_memory || rebuilt_position)rebuild=[&](const auto&,const auto&){++calls;return std::make_pair(rebuilt_memory.value_or(at::Tensor()),rebuilt_position.value_or(at::Tensor()));};
        sam3::remap_multiplex_history(history,old_state,new_state,rebuild,mode);const auto& out=history.conditioning.front();
        c10::Dict<std::string,at::Tensor> result;const auto save=[&](const std::string& key,const at::Tensor& value){if(value.defined())result.insert(key,value);};
        save("low",out.masks.low_res_mask);save("high",out.masks.high_res_mask);save("logits",out.masks.object_logits);save("pointer",out.pointer);save("iou",out.iou);save("confidence",out.confidence);save("image",out.image);save("image_position",out.image_position);save("memory",out.memory);save("position",out.memory_position);save("candidates",out.masks.low_res_multimasks);save("conditions",at::tensor(out.conditioning_objects,at::kLong));save("rebuild_calls",at::scalar_tensor(calls,at::kLong));return result;
      });
}
