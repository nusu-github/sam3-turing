// Development-only dispatcher adapter for original-Python session comparisons.
// The production API and session implementation do not require Python.
#include "sam3/tracking_session.h"
#include "sam3/video_recondition.h"
#include "sam3/video_memory.h"
#include "sam3/preprocess.h"
#include "frame_order.h"
#include <torch/library.h>
namespace {
using Snapshot=c10::Dict<std::string,at::Tensor>;
void save(Snapshot& out,const std::string& key,const at::Tensor& tensor) {
  if(tensor.defined()) out.insert(key,tensor.cpu().clone());
}
void save_frame(Snapshot& out,const std::string& prefix,const sam3::TrackingFrame& frame) {
  save(out,prefix+"pred_masks",frame.low_mask);save(out,prefix+"obj_ptr",frame.pointer);
  save(out,prefix+"object_score_logits",frame.object_logits);save(out,prefix+"maskmem_features",frame.memory);
  save(out,prefix+"maskmem_pos_enc",frame.memory_position);save(out,prefix+"iou_score",frame.iou);save(out,prefix+"eff_iou_score",frame.confidence);
}
void save_history(Snapshot& out,const std::string& prefix,const sam3::TrackingHistory& history) {
  for(const bool cond:{true,false}) {
    const auto name=prefix+(cond?"cond/":"tracked/");std::vector<int64_t> ids;
    for(const auto& frame:cond?history.conditioning:history.tracked) {ids.push_back(frame.index);save_frame(out,name+std::to_string(frame.index)+"/",frame);}
    save(out,name+"order",at::tensor(ids,at::kLong));
  }
}
void save_state(Snapshot& out,const std::string& prefix,const sam3::Sam3TrackingSession& session) {
  const auto& state=session.state();save(out,prefix+"ids",at::tensor(session.object_ids(),at::kLong));
  save(out,prefix+"status",at::tensor({int64_t(state.started),state.first_annotation.value_or(-1)},at::kLong));
  std::vector<int64_t> tracked;for(const auto& [index,reverse]:state.tracked_direction) {tracked.push_back(index);tracked.push_back(reverse);}
  save(out,prefix+"direction",at::tensor(tracked,at::kLong).reshape({-1,2}));
  save_history(out,prefix+"global/",state.history);
  for(const bool cond:{true,false}) {
    const auto& indices=cond?state.consolidated_conditioning:state.consolidated_tracked;
    save(out,prefix+(cond?"consolidated_cond":"consolidated_tracked"),at::tensor(std::vector<int64_t>(indices.begin(),indices.end()),at::kLong));
  }
  for(size_t i=0;i<state.objects.size();++i) {
    const auto& object=state.objects[i];const auto root=prefix+"obj"+std::to_string(i)+"/";
    for(const auto& [index,point]:object.points) {const auto key=root+"points/"+std::to_string(index)+"/";save(out,key+"coords",point.points);save(out,key+"labels",point.labels);}
    for(const auto& [index,mask]:object.masks) save(out,root+"masks/"+std::to_string(index),mask);
    save_history(out,root,object.history);
    for(const bool cond:{true,false}) for(const auto& [index,edit]:cond?object.pending_conditioning:object.pending_tracked) {
      const auto key=root+(cond?"pending_cond/":"pending_tracked/")+std::to_string(index)+"/";
      save_frame(out,key,edit.frame);save(out,key+"pred_masks_video_res",edit.video_mask);
    }
  }
}
}
TORCH_LIBRARY_FRAGMENT(sam3_native,m) {
  m.def("resize_tracking_rgb(Tensor image, int height, int width) -> Tensor",&sam3::resize_tracking_rgb);
  m.def("preprocess_video_rgb(Tensor image) -> Tensor",&sam3::preprocess_video_rgb);
  m.def("preprocess_tracking_rgb(Tensor image) -> Tensor",&sam3::preprocess_tracking_rgb);
  m.def("tracking_frame_order(int[][] groups, bool dictionary) -> int[]",&sam3::detail::frame_set_order);
  m.def("tracking_postprocess(Tensor masks, int height, int width, bool non_overlap, int area) -> Tensor",&sam3::postprocess_tracking_masks);
  m.def("tracking_session(str directory, Tensor[] images, Tensor[] positions, Tensor[] high0, Tensor[] high1, int[][] operations, Tensor[] payloads, int[] settings, bool[] flags, str mode) -> Dict(str, Tensor)",
    [](const std::string& directory,const std::vector<at::Tensor>& images,const std::vector<at::Tensor>& positions,
       const std::vector<at::Tensor>& high0,const std::vector<at::Tensor>& high1,const std::vector<std::vector<int64_t>>& operations,
       const std::vector<at::Tensor>& payloads,const std::vector<int64_t>& settings,const c10::List<bool>& flags,const std::string& mode) {
      TORCH_CHECK(settings.size()==7 && flags.size()==8 && operations.size()==payloads.size(),"invalid session parity options");
      TORCH_CHECK(!images.empty() && images.size()==positions.size() && images.size()==high0.size() && images.size()==high1.size(),"invalid test features");
      sam3::TrackingSessionOptions options;options.offload_state=flags[0];options.non_overlap_output=flags[1];options.clear_near_input=flags[2];options.clear_near_multi_object=flags[3];
      options.all_edits_conditioning=flags[4];options.always_start_at_first_annotation=flags[5];options.frame.temporal.select_by_score=flags[6];options.frame.non_overlap_memory=flags[7];
      options.max_points=settings[3];options.fill_hole_area=settings[4];options.frame.temporal.memory_slots=settings[5];options.frame.temporal.max_conditioning_frames=2;options.frame.temporal.max_pointer_frames=settings[6];
      const auto core=std::make_shared<sam3::Sam3TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(directory)),images.front().device());
      sam3::Sam3TrackingSession session(core,[&](int64_t frame){const auto i=frame%images.size();return sam3::TrackingFeatures{images[i],positions[i],{high0[i],high1[i]}};},settings[0],settings[1],settings[2],images.front().device(),mode,options);
      Snapshot result;
      for(size_t i=0;i<operations.size();++i) {
        const auto& op=operations[i];TORCH_CHECK(op.size()==8,"operation must have eight integers");
        const auto prefix=std::to_string(i)+"/";int64_t outputs=0;
        const auto emit=[&](const sam3::TrackingSessionOutput& out,bool propagation=false) {
          const auto key=prefix+"out"+std::to_string(outputs++)+"/";
          save(result,key+"frame",at::tensor(out.index,at::kLong));save(result,key+"ids",at::tensor(out.object_ids,at::kLong));save(result,key+"masks",out.masks);
          if(propagation) {save(result,key+"low",out.low_masks);save(result,key+"logits",out.object_logits);}
        };
        if(op[0]==0 || op[0]==1) {
          sam3::TrackingPoints prompt;prompt.normalized=op[4];
          if(op[0]==0) {prompt.points=payloads[i].slice(1,0,2);prompt.labels=payloads[i].select(1,2);}else prompt.box=payloads[i];
          emit(session.add_points(op[1],op[2],prompt,op[3],op[5]));
        } else if(op[0]==2) emit(session.add_mask(op[1],op[2],payloads[i]));
        else if(op[0]==8){sam3::ReconditionMasks prepared;prepared.ids={op[2]};prepared.binary_masks=payloads[i].gt(0).unsqueeze(0);const auto executed=sam3::execute_reconditioning(op[1],prepared,std::vector<sam3::Sam3TrackingSession*>{&session});save(result,prefix+"affected",at::tensor(std::vector<int64_t>(executed.affected_ids.begin(),executed.affected_ids.end()),at::kLong));}
        else if(op[0]==9) sam3::update_video_memories(op[1],payloads[i].to(images.front().device()),session.object_ids(),std::vector<sam3::Sam3TrackingSession*>{&session},bool(op[3]));
        else if(op[0]==3) session.preflight(op[3]);
        else if(op[0]==4) {
          sam3::TrackingPropagation request;if(op[1]>=0)request.start=op[1];if(op[2]>=0)request.max_steps=op[2];request.reverse=op[3];request.encode_memory=op[4];request.preflight=op[5];
          session.propagate(request,[&](const auto& out){emit(out,true);if(op[6]>0 && outputs>=op[6]) {if(op[7])session.cancel();else return false;}return true;});
        } else if(op[0]==5) emit(session.clear_input(op[1],op[2]));
        else if(op[0]==6) for(const auto& out:session.remove_object(op[2],op[3]))emit(out);
        else if(op[0]==7) session.reset();
        else TORCH_CHECK(false,"unknown session test operation");
        save(result,prefix+"outputs",at::tensor(outputs,at::kLong));save_state(result,prefix+"state/",session);
      }
      return result;
    });
}
