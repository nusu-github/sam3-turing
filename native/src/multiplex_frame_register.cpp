#include "sam3/multiplex_frame.h"
#include <torch/library.h>
#include <set>
TORCH_LIBRARY_FRAGMENT(sam3_native,m) {
  m.def("multiplex_frame(str directory, Tensor[] features, Tensor? points, Tensor? labels, Tensor? masks, Tensor? previous, int[]? objects, int[][] assignments, int[] ids, bool[] conditioning, Tensor[] memory, Tensor[] positions, Tensor[] pointers, Tensor[] scores, Tensor[] images, Tensor[] image_positions, int[] settings, bool[] flags, float threshold, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const std::vector<at::Tensor>& features,const std::optional<at::Tensor>& points,const std::optional<at::Tensor>& labels,
         const std::optional<at::Tensor>& masks,const std::optional<at::Tensor>& previous,const std::optional<std::vector<int64_t>>& objects,
         const std::vector<std::vector<int64_t>>& assignments,const std::vector<int64_t>& ids,const c10::List<bool>& conditioning,
         const std::vector<at::Tensor>& memory,const std::vector<at::Tensor>& positions,const std::vector<at::Tensor>& pointers,
         const std::vector<at::Tensor>& scores,const std::vector<at::Tensor>& images,const std::vector<at::Tensor>& image_positions,
         const std::vector<int64_t>& settings,const c10::List<bool>& flags,double threshold,const std::string& mode) {
        TORCH_CHECK(features.size()==8 && settings.size()==8 && flags.size()==11,"invalid multiplex-frame test arguments");
        TORCH_CHECK(ids.size()==conditioning.size() && ids.size()==memory.size() && ids.size()==positions.size() && ids.size()==pointers.size() && ids.size()==scores.size() && ids.size()==images.size() && ids.size()==image_positions.size(),"inconsistent history arrays");
        const auto device=features[0].device();const sam3::MultiplexState buckets(assignments,device,at::kFloat,16);
        sam3::MultiplexFrameHistory history;
        for(size_t i=0;i<ids.size();++i) {
          sam3::MultiplexFrame frame;frame.index=ids[i];frame.memory=memory[i];frame.memory_position=positions[i];frame.pointer=pointers[i];frame.confidence=scores[i];frame.image=images[i];frame.image_position=image_positions[i];
          frame.masks.low_res_mask=at::ones({buckets.object_count(),1,1,1});frame.masks.high_res_mask=at::ones_like(frame.masks.low_res_mask);frame.iou=at::ones({buckets.object_count()});frame.masks.low_res_multimasks=frame.masks.low_res_mask;
          (conditioning[i]?history.conditioning:history.tracked).push_back(std::move(frame));
        }
        sam3::MultiplexFrameRequest request;request.index=settings[0];request.frame_count=settings[1];request.initial=flags[0];request.reverse=flags[1];request.encode_memory=flags[2];request.points=points.value_or(at::Tensor());request.labels=labels.value_or(at::Tensor());request.mask=masks.value_or(at::Tensor());request.previous_logits=previous.value_or(at::Tensor());request.objects_to_interact=objects;
        sam3::MultiplexFrameOptions options;options.temporal.memory_slots=settings[2];options.temporal.max_conditioning_frames=settings[3];options.temporal.max_pointer_frames=settings[4];options.temporal.stride=settings[5];options.multimask_min_points=settings[6];options.multimask_max_points=settings[7];
        options.temporal.select_by_score=flags[3];options.non_overlap_memory=flags[4];options.offload_output=flags[5];options.trim_history=flags[6];options.save_image=flags[7];options.multimask=flags[8];options.multimask_tracking=flags[9];options.attenuate_iou_by_stability=flags[10];options.object_threshold=threshold;
        const sam3::Sam31TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(directory)),device);
        const auto frame=core.forward({features[0],features[1],{features[2],features[3]}},{features[4],features[5],{features[6],features[7]}},request,history,buckets,options,mode);
        c10::Dict<std::string,at::Tensor> out;const auto save=[&](const std::string& name,const at::Tensor& value){if(value.defined())out.insert(name,value);};
        save("pred_masks",frame.masks.low_res_mask);save("pred_masks_high_res",frame.masks.high_res_mask);save("object_score_logits",frame.masks.object_logits);save("obj_ptr",frame.pointer);
        save("maskmem_features",frame.memory);save("maskmem_pos_enc",frame.memory_position);save("image_features",frame.image);save("image_pos_enc",frame.image_position);save("iou_score",frame.iou);save("eff_iou_score",frame.confidence);
        save("multistep_pred_multimasks",frame.masks.low_res_multimasks);save("multistep_pred_multimasks_high_res",frame.masks.high_res_multimasks);save("multistep_pred_ious",frame.masks.iou);
        const std::set<int64_t> sorted(frame.conditioning_objects.begin(),frame.conditioning_objects.end());save("conditioning_objects",at::tensor(std::vector<int64_t>(sorted.begin(),sorted.end()),at::kLong));
        std::vector<int64_t> status;
        for(const auto* group:{&history.conditioning,&history.tracked})for(const auto& f:*group)status.insert(status.end(),{f.index,int64_t(f.memory.defined()),int64_t(f.memory_position.defined()),int64_t(f.image.defined()),int64_t(f.image_position.defined()),int64_t(f.masks.high_res_mask.defined()),int64_t(f.iou.defined()),int64_t(f.confidence.defined()),int64_t(f.masks.low_res_multimasks.defined())});
        save("history_status",at::tensor(status,at::kLong).reshape({-1,9}));return out;
      });
}
