#include "sam3/multiplex_frame.h"
#include <torch/library.h>
#include <set>
TORCH_LIBRARY_FRAGMENT(sam3_native,m) {
  m.def("multiplex_update(str directory, Tensor[] features, Tensor masks, int[] indices, int[]? object_ids, int[][] assignments, int[]? existing_ids, int capacity, Dict(str, Tensor) previous, int[] conditioning, bool[] flags, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const std::vector<at::Tensor>& features,const at::Tensor& masks,const std::vector<int64_t>& indices,
         const std::optional<std::vector<int64_t>>& object_ids,const std::vector<std::vector<int64_t>>& assignments,const std::optional<std::vector<int64_t>>& existing_ids,
         int64_t capacity,const c10::Dict<std::string,at::Tensor>& previous,const std::vector<int64_t>& conditioning,const c10::List<bool>& flags,const std::string& mode) {
        TORCH_CHECK(features.size()==4 && flags.size()==8,"invalid mask-update test arguments");
        const auto device=features[0].device();sam3::MultiplexState state(assignments,device,at::kFloat,capacity,existing_ids);
        sam3::MultiplexFrame frame;frame.conditioning_objects=conditioning;
        const auto get=[&](const std::string& key){return previous.contains(key)?previous.at(key):at::Tensor();};
        frame.masks.low_res_mask=get("pred_masks");frame.masks.high_res_mask=get("pred_masks_high_res");frame.masks.object_logits=get("object_score_logits");
        frame.pointer=get("obj_ptr");frame.memory=get("maskmem_features");frame.memory_position=get("maskmem_pos_enc");frame.iou=get("iou_score");frame.confidence=get("eff_iou_score");
        frame.image=get("image_features");frame.image_position=get("image_pos_enc");frame.input_masks=get("input_masks");frame.masks.low_res_multimasks=get("candidates");
        sam3::MultiplexMaskUpdate request;request.append=flags[0];request.encode_memory=flags[1];request.masks_from_points=flags[2];request.allow_new_buckets=flags[3];request.prefer_new_buckets=flags[4];
        sam3::MultiplexFrameOptions options;options.temporal.select_by_score=flags[5];options.non_overlap_memory=flags[6];options.save_image=flags[7];
        const sam3::Sam31TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(directory)),device);
        const auto affected=core.update_masks({features[0],{},{features[1],features[2]}},{features[3],{},{}},masks,indices,object_ids,frame,state,request,options,mode);
        c10::Dict<std::string,at::Tensor> out;const auto save=[&](const std::string& key,const at::Tensor& value){if(value.defined())out.insert(key,value);};
        save("pred_masks",frame.masks.low_res_mask);save("pred_masks_high_res",frame.masks.high_res_mask);save("object_score_logits",frame.masks.object_logits);
        save("obj_ptr",frame.pointer);save("maskmem_features",frame.memory);save("maskmem_pos_enc",frame.memory_position);save("iou_score",frame.iou);save("eff_iou_score",frame.confidence);
        save("image_features",frame.image);save("image_pos_enc",frame.image_position);save("input_masks",frame.input_masks);save("candidates",frame.masks.low_res_multimasks);
        const std::set<int64_t> sorted(frame.conditioning_objects.begin(),frame.conditioning_objects.end());save("conditioning_objects",at::tensor(std::vector<int64_t>(sorted.begin(),sorted.end()),at::kLong));
        std::vector<int64_t> slots;for(const auto& bucket:state.assignments())slots.insert(slots.end(),bucket.begin(),bucket.end());save("assignments",at::tensor(slots,at::kLong).reshape({-1,16}));
        if(state.object_ids())save("object_ids",at::tensor(*state.object_ids(),at::kLong));save("affected",at::tensor(affected,at::kLong));return out;
      });
}
