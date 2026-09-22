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
#include "sam3/temporal_memory.h"
#include "sam3/multiplex.h"
#include "sam3/multiplex_decoder.h"
#include "sam3/multiplex_temporal.h"
#include "sam3/tracking_frame.h"
#include "sam3/autocast.h"
#include "sam3/vision_encoder.h"
#include "sam3/preprocess.h"
#include <torch/library.h>
#include <cmath>

namespace {
void multiplex_snapshot(c10::Dict<std::string,at::Tensor>& out,const std::string& prefix,
    const sam3::MultiplexState& state,const at::Tensor& probe) {
  const auto cpu=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  out.insert(prefix+"counts",at::tensor({int64_t(state.valid()),state.bucket_count(),state.width(),state.capacity(),state.object_count(),state.occupied_count(),state.valid()?state.available_slots():0},cpu));
  std::vector<int64_t> flat;for (const auto& bucket:state.assignments()) flat.insert(flat.end(),bucket.begin(),bucket.end());
  out.insert(prefix+"assignments",at::tensor(flat,cpu).reshape({state.bucket_count(),state.width()}));
  if (state.object_ids()) out.insert(prefix+"ids",at::tensor(*state.object_ids(),cpu));
  if (!state.valid()) return;
  out.insert(prefix+"mux_matrix",state.mux_matrix());out.insert(prefix+"demux_matrix",state.demux_matrix());out.insert(prefix+"valid_mask",state.valid_object_mask());
  if (state.object_count()>0) {
    TORCH_CHECK(probe.size(0)>=state.object_count(),"multiplex probe is too short");
    const auto muxed=state.mux(probe.slice(0,0,state.object_count()));
    out.insert(prefix+"mux",muxed);out.insert(prefix+"demux",state.demux(muxed));
  }
}
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
  m.def("tracking_frame(str directory, Tensor image, Tensor position, Tensor[] high, Tensor? points, Tensor? labels, Tensor? mask, Tensor? previous, int[] ids, bool[] conditioning, Tensor[] memory, Tensor[] memory_positions, Tensor[] pointers, Tensor[] scores, int[] settings, bool[] flags, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const at::Tensor& image,const at::Tensor& position,const std::vector<at::Tensor>& high,
         const std::optional<at::Tensor>& points,const std::optional<at::Tensor>& labels,const std::optional<at::Tensor>& mask,const std::optional<at::Tensor>& previous,
         const std::vector<int64_t>& ids,const c10::List<bool>& conditioning,const std::vector<at::Tensor>& memory,
         const std::vector<at::Tensor>& positions,const std::vector<at::Tensor>& pointers,const std::vector<at::Tensor>& scores,
         const std::vector<int64_t>& settings,const c10::List<bool>& flags,const std::string& mode) {
        TORCH_CHECK(settings.size()==8 && flags.size()==10,"invalid tracking-frame test options");
        TORCH_CHECK(ids.size()==conditioning.size() && ids.size()==memory.size() && ids.size()==positions.size() && ids.size()==pointers.size() && ids.size()==scores.size(),"inconsistent tracking history");
        sam3::TrackingHistory history;
        for (size_t i=0;i<ids.size();++i) {
          sam3::TrackingFrame frame;frame.index=ids[i];frame.memory=memory[i];frame.memory_position=positions[i];frame.pointer=pointers[i];
          frame.confidence=scores[i].numel()?scores[i]:at::Tensor();frame.iou=at::ones({image.size(0)});frame.high_mask=at::ones({image.size(0),1,1,1});
          (conditioning[i]?history.conditioning:history.tracked).push_back(std::move(frame));
        }
        sam3::TrackingFrameRequest request;request.index=settings[0];request.frame_count=settings[1];request.initial=flags[0];request.reverse=flags[1];request.use_previous=flags[2];request.encode_memory=flags[3];
        request.points=points.value_or(at::Tensor());request.labels=labels.value_or(at::Tensor());request.mask=mask.value_or(at::Tensor());request.previous_logits=previous.value_or(at::Tensor());
        sam3::TrackingFrameOptions options;options.temporal.memory_slots=settings[2];options.temporal.max_conditioning_frames=settings[3];options.temporal.max_pointer_frames=settings[4];options.temporal.stride=settings[5];
        options.multimask_min_points=settings[6];options.multimask_max_points=settings[7];options.temporal.select_by_score=flags[4];options.non_overlap_memory=flags[5];options.offload_output=flags[6];options.trim_history=flags[7];options.multimask=flags[8];options.multimask_tracking=flags[9];
        const sam3::Sam3TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(directory)),image.device());
        const auto frame=core.forward({image,position,high},request,history,options,mode);
        c10::Dict<std::string,at::Tensor> out;
        out.insert("pred_masks",frame.low_mask);out.insert("pred_masks_high_res",frame.high_mask);out.insert("obj_ptr",frame.pointer);out.insert("object_score_logits",frame.object_logits);
        if (frame.iou.defined()) {out.insert("iou_score",frame.iou);out.insert("eff_iou_score",frame.confidence);}
        if (frame.memory.defined()) {out.insert("maskmem_features",frame.memory);out.insert("maskmem_pos_enc",frame.memory_position);}
        std::vector<int64_t> status;
        for (const auto* frames:{&history.conditioning,&history.tracked}) for (const auto& past:*frames)
          status.insert(status.end(),{past.index,int64_t(past.memory.defined()),int64_t(past.memory_position.defined()),int64_t(past.high_mask.defined()),int64_t(past.iou.defined()),int64_t(past.confidence.defined())});
        out.insert("history_status",at::tensor(status,at::kLong).reshape({-1,6}));return out;
      });
  m.def("multiplex_temporal(str directory, Tensor source, Tensor source_position, int[][] assignments, int[] ids, bool[] conditioning, Tensor[] features, Tensor[] positions, Tensor[] pointers, Tensor[] scores, Tensor[] images, Tensor[] image_positions, int[] settings, bool[] flags, float threshold, str mode) -> Dict(str, Tensor)",
      [](const std::string& directory,const at::Tensor& source,const at::Tensor& source_position,
         const std::vector<std::vector<int64_t>>& assignments,const std::vector<int64_t>& ids,const c10::List<bool>& conditioning,
         const std::vector<at::Tensor>& features,const std::vector<at::Tensor>& positions,const std::vector<at::Tensor>& pointers,
         const std::vector<at::Tensor>& scores,const std::vector<at::Tensor>& images,const std::vector<at::Tensor>& image_positions,
         const std::vector<int64_t>& settings,const c10::List<bool>& flags,double threshold,const std::string& mode) {
        TORCH_CHECK(settings.size()==8 && flags.size()==10,"invalid multiplex temporal test options");
        TORCH_CHECK(ids.size()==conditioning.size() && ids.size()==features.size() && ids.size()==positions.size() && ids.size()==pointers.size() && ids.size()==scores.size() && ids.size()==images.size() && ids.size()==image_positions.size(),"inconsistent temporal frame arrays");
        const auto optional=[](const at::Tensor& t) {return t.dim()==1 && t.numel()==0?at::Tensor():t;};
        sam3::MultiplexTemporalState state;
        for (size_t i=0;i<ids.size();++i) (conditioning[i]?state.conditioning:state.tracked).push_back(
            {ids[i],optional(features[i]),optional(positions[i]),optional(pointers[i]),optional(scores[i]),optional(images[i]),optional(image_positions[i])});
        sam3::MultiplexTemporalOptions options;
        options.memory_slots=settings[4];options.max_conditioning_frames=settings[5];options.max_pointer_frames=settings[6];options.stride=settings[7];
        options.keep_first=flags[3];options.select_by_score=flags[4];options.only_past_pointers=flags[5];options.signed_pointer_time=flags[6];
        options.temporal_v2=flags[7];options.encode_pointer_time=flags[8];options.use_pointers=flags[9];options.score_threshold=threshold;
        const sam3::MultiplexState buckets(assignments,source.device(),at::kFloat,16);
        const sam3::MultiplexMemoryConditioner conditioner(sam3::WeightStore(std::filesystem::u8path(directory)),source.device());
        sam3::MultiplexTemporalAssembly trace;c10::Dict<std::string,at::Tensor> out;
        out.insert("features",conditioner.forward(source,source_position,settings[0],settings[1],settings[2],settings[3],flags[0],flags[1],flags[2],state,buckets,options,mode,&trace));
        out.insert("counts",at::tensor({trace.pointer_tokens,int64_t(trace.fuse)},at::kLong));
        if (trace.fuse) {out.insert("memory",trace.memory);out.insert("position",trace.position);out.insert("image",trace.image);out.insert("image_position",trace.image_position);}
        for (const auto* frames:{&state.conditioning,&state.tracked}) for (const auto& entry:*frames) {
          if (entry.features.defined()) out.insert("stored_features_"+std::to_string(entry.index),entry.features);
          if (entry.position.defined()) out.insert("stored_position_"+std::to_string(entry.index),entry.position);
        }
        return out;
      });
  m.def("multiplex_decode(str directory, Tensor image, Tensor position, Tensor[] high, Tensor? extra, str mode) -> Tensor[]",
      [](const std::string& directory,const at::Tensor& image,const at::Tensor& position,const std::vector<at::Tensor>& high,
         const std::optional<at::Tensor>& extra,const std::string& mode) {
        const sam3::MultiplexMaskDecoder decoder(sam3::WeightStore(std::filesystem::u8path(directory)),image.device());
        const auto out=decoder.forward(image,position,high,extra.value_or(at::Tensor()),mode);
        return std::vector<at::Tensor>{out.masks,out.iou,out.tokens,out.object_logits};
      });
  m.def("multiplex_propagation(str directory, Tensor image, Tensor[] high, int[][] assignments, str mode, float threshold, bool attenuate, bool project) -> Tensor[]",
      [](const std::string& directory,const at::Tensor& image,const std::vector<at::Tensor>& high,
         const std::vector<std::vector<int64_t>>& assignments,const std::string& mode,double threshold,bool attenuate,bool project) {
        const sam3::MultiplexPropagationHeads heads(sam3::WeightStore(std::filesystem::u8path(directory)),image.device());
        const sam3::MultiplexState state(assignments,image.device(),at::kFloat,16);
        const auto projected=project?heads.project_pyramid(high,mode):high;
        const auto out=heads.forward(state,image,projected,mode,threshold,attenuate);
        return std::vector<at::Tensor>{out.low_res_multimasks,out.high_res_multimasks,out.iou,out.low_res_mask,out.high_res_mask,out.object_pointer,out.object_logits,heads.dense_position(mode),projected[0],projected[1]};
      });
  m.def("multiplex_controller(Tensor probe, int objects, int width, int capacity, bool full_shuffle, bool random, int[]? ids, str mode) -> Dict(str, Tensor)",
      [](const at::Tensor& probe,int64_t objects,int64_t width,int64_t capacity,bool full,bool random,const std::optional<std::vector<int64_t>>& ids,const std::string& mode) {
        const auto dtype=mode=="fp16"?at::kHalf:mode=="bf16_reference"?at::kBFloat16:at::kFloat;
        sam3::AutocastGuard autocast(probe.device().type(),mode!="fp32",dtype);
        const auto state=sam3::MultiplexController(width,full,capacity).get_state(objects,probe.device(),dtype,random,ids);
        c10::Dict<std::string,at::Tensor> out;multiplex_snapshot(out,"",state,probe);return out;
      });
  m.def("multiplex_state(Tensor probe, int[][] assignments, int capacity, int[]? ids, str[] operations, int[][] indices, int[][] added_ids, bool[] allow_new, bool[] prefer_new, bool[] strict, str mode) -> Dict(str, Tensor)",
      [](const at::Tensor& probe,const std::vector<std::vector<int64_t>>& assignments,int64_t capacity,const std::optional<std::vector<int64_t>>& ids,
         const std::vector<std::string>& operations,const std::vector<std::vector<int64_t>>& indices,const std::vector<std::vector<int64_t>>& added_ids,
         const c10::List<bool>& allow,const c10::List<bool>& prefer,const c10::List<bool>& strict,const std::string& mode) {
        TORCH_CHECK(operations.size()==indices.size() && operations.size()==added_ids.size() && operations.size()==allow.size() && operations.size()==prefer.size() && operations.size()==strict.size(),"inconsistent multiplex operations");
        const auto dtype=mode=="fp16"?at::kHalf:mode=="bf16_reference"?at::kBFloat16:at::kFloat;
        sam3::AutocastGuard autocast(probe.device().type(),mode!="fp32",dtype);
        sam3::MultiplexState state(assignments,probe.device(),dtype,capacity,ids);c10::Dict<std::string,at::Tensor> out;
        multiplex_snapshot(out,"0.",state,probe);
        for (size_t i=0;i<operations.size();++i) {
          const auto prefix=std::to_string(i+1)+".";
          if (operations[i]=="add") state.add_objects(indices[i],ids?std::optional<std::vector<int64_t>>(added_ids[i]):std::nullopt,allow[i],prefer[i]);
          else if (operations[i]=="remove") out.insert(prefix+"kept",at::tensor(state.remove_objects(indices[i],strict[i]),at::kLong));
          else if (operations[i]=="next") {TORCH_CHECK(indices[i].size()==1,"next needs a count");out.insert(prefix+"next",at::tensor(state.next_indices(indices[i][0],allow[i],prefer[i]),at::kLong));}
          else TORCH_CHECK(false,"unknown multiplex operation");
          multiplex_snapshot(out,prefix,state,probe);
        }
        return out;
      });
  m.def("select_conditioning_frames(int current, int[] order, int limit, bool keep_first) -> (int[], int[])",
      [](int64_t current,const std::vector<int64_t>& order,int64_t limit,bool keep) {
        auto selected=sam3::select_conditioning_frames(current,order,limit,keep);
        return std::make_tuple(std::move(selected.first),std::move(selected.second));
      });
  m.def("memory_confidence(Tensor logits, Tensor iou) -> Tensor",&sam3::memory_confidence);
  m.def("sam3_temporal(str directory, Tensor source, Tensor source_position, int height, int width, int[] frame_ids, bool[] conditioning, Tensor[] features, Tensor[] positions, Tensor[] pointers, Tensor[] scores, int frame, int frame_count, bool initial, bool reverse, bool use_previous, int slots, int condition_limit, int pointer_limit, int stride, bool keep_first, bool filter_scores, float score_threshold, str mode) -> (Tensor, Tensor[], int[])",
      [](const std::string& directory,const at::Tensor& source,const at::Tensor& source_position,int64_t height,int64_t width,
         const std::vector<int64_t>& ids,const c10::List<bool>& conditioning,const std::vector<at::Tensor>& features,const std::vector<at::Tensor>& positions,
         const std::vector<at::Tensor>& pointers,const std::vector<at::Tensor>& scores,int64_t frame,int64_t frame_count,bool initial,bool reverse,bool previous,
         int64_t slots,int64_t condition_limit,int64_t pointer_limit,int64_t stride,bool keep_first,bool filter,double threshold,const std::string& mode) {
        TORCH_CHECK(ids.size()==conditioning.size() && ids.size()==features.size() && ids.size()==positions.size() && ids.size()==pointers.size() && ids.size()==scores.size(),"inconsistent temporal test frame arrays");
        sam3::TemporalState state;
        for (size_t i=0;i<ids.size();++i) {
          sam3::TemporalFrame entry{ids[i],features[i],positions[i],pointers[i],{}};
          if (scores[i].numel()) entry.effective_iou=scores[i];
          (conditioning[i]?state.conditioning:state.tracked).push_back(std::move(entry));
        }
        const sam3::WeightStore store(std::filesystem::u8path(directory));
        const sam3::Sam3MemoryConditioner module(store,source.device());sam3::TemporalAssembly trace;
        const auto out=module.forward(source,source_position,height,width,frame,frame_count,initial,reverse,previous,state,
            {slots,condition_limit,pointer_limit,stride,keep_first,filter,threshold},mode,&trace);
        std::vector<at::Tensor> tensors;
        if (trace.memory.defined()) tensors={trace.memory,trace.position};
        return std::make_tuple(out,tensors,std::vector<int64_t>{trace.pointer_tokens});
      });
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
