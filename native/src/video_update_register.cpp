#include "sam3/video_update.h"
#include <torch/library.h>
namespace {
using Snapshot=c10::Dict<std::string,at::Tensor>;
void save(Snapshot& out,const std::string& name,const at::Tensor& x){if(x.defined())out.insert(name,x.cpu().clone());}
void ids(Snapshot& out,const std::string& name,const std::vector<int64_t>& x){save(out,name,at::tensor(x,at::kLong));}
void snapshot(Snapshot& out,const std::string& root,const sam3::VideoUpdatePlan& p){
  const auto& m=p.metadata;ids(out,root+"ids",m.object_ids());ids(out,root+"new_ids",p.new_ids);ids(out,root+"new_ranks",p.new_ranks);ids(out,root+"new_detections",p.association.new_detections);ids(out,root+"removed",{p.removed.begin(),p.removed.end()});ids(out,root+"unmatched",p.association.unmatched_tracks);ids(out,root+"empty",p.association.empty_tracks);ids(out,root+"corrections",p.corrections.ids);ids(out,root+"geometry",{p.correction_decision.geometry_ids.begin(),p.correction_decision.geometry_ids.end()});
  save(out,root+"max_id",at::scalar_tensor(m.max_id,at::kLong));save(out,root+"tracking_masks",p.tracking_masks);save(out,root+"correction_masks",p.corrections.binary_masks);
  for(size_t r=0;r<m.ids_per_rank.size();++r)ids(out,root+"rank/"+std::to_string(r),m.ids_per_rank[r]);
  for(const auto& [id,score]:m.object_scores)save(out,root+"score/"+std::to_string(id),at::scalar_tensor(score,at::kDouble));
  for(const auto& [frame,scores]:m.frame_scores)for(const auto& [id,score]:scores)save(out,root+"frame_score/"+std::to_string(frame)+"/"+std::to_string(id),score);
  for(const auto& [id,frame]:m.last_occluded)save(out,root+"occluded/"+std::to_string(id),frame);
  ids(out,root+"confirmation/status",m.confirmation.status);ids(out,root+"confirmation/count",m.confirmation.consecutive_detections);
  for(const auto& [id,frame]:m.host.first_frame)save(out,root+"host/first/"+std::to_string(id),at::scalar_tensor(frame,at::kLong));
  for(const auto& [id,count]:m.host.keep_alive)save(out,root+"host/alive/"+std::to_string(id),at::scalar_tensor(count,at::kLong));
  for(const auto& [id,frames]:m.host.unmatched_frames)ids(out,root+"host/unmatched/"+std::to_string(id),frames);
  for(const auto& [pair,frames]:m.host.overlap_frames)ids(out,root+"host/overlap/"+std::to_string(pair.first)+"/"+std::to_string(pair.second),frames);
  ids(out,root+"host/removed",{m.host.removed.begin(),m.host.removed.end()});
  for(const auto& [frame,values]:m.host.suppressed)ids(out,root+"host/suppressed/"+std::to_string(frame),{values.begin(),values.end()});
  const auto& s=m.device;save(out,root+"device/first",s.first_frame);save(out,root+"device/unmatched",s.unmatched_count);save(out,root+"device/alive",s.keep_alive);save(out,root+"device/removed",s.removed);save(out,root+"device/overlap",s.overlap_count);save(out,root+"device/occluded",s.last_occluded);
}
}
TORCH_LIBRARY_FRAGMENT(sam3_native,m){
  m.def("video_update_sequence(Tensor[][] inputs, int[] frames, bool multiplex, int ranks, int[] integers, float[] floats, bool[] flags, str mode='fp32') -> Dict(str, Tensor)",
    [](const std::vector<std::vector<at::Tensor>>& inputs,const std::vector<int64_t>& frames,bool mux,int64_t ranks,const std::vector<int64_t>& integers,const std::vector<double>& floats,const c10::List<bool>& flags,const std::string& mode){
      TORCH_CHECK(inputs.size()==frames.size() && !inputs.empty() && integers.size()==10 && floats.size()==6 && flags.size()==8,"invalid video planning test parameters");
      sam3::VideoUpdateOptions o;o.association.policy=o.recondition.policy=mux?sam3::AssociationPolicy::Sam31:sam3::AssociationPolicy::Sam3;
      o.hotstart={integers[0],integers[1],integers[2],integers[3],integers[4],integers[5],flags[4],flags[5]};o.confirmation_threshold=integers[6];o.recondition.period=integers[7];o.cleanup_area=integers[8];o.bucket_capacity=integers[9];
      o.association.new_detection_threshold=floats[0];o.association.track_match_threshold=floats[1];o.association.detection_match_threshold=floats[2];o.recondition.box_iou_threshold=floats[3];o.recondition.detection_score_threshold=floats[4];o.occlusion_threshold=floats[5];o.policy_mode=mode;
      o.warmup_complete=flags[1];o.confirmation_enabled=flags[2];o.boundary_filter=flags[3];o.association.use_iom=flags[6];o.allow_unoccluded_suppression=flags[7];
      auto metadata=sam3::initialize_video_metadata(ranks,inputs[0][0].device());Snapshot out;
      for(size_t i=0;i<inputs.size();++i){const auto& x=inputs[i];TORCH_CHECK(x.size()==6,"each frame requires detections/score/boxes/keep/tracks/logits");const auto root=std::to_string(i)+"/";sam3::VideoDetections d{x[0],x[1],x[2],x[3]};
        auto p=sam3::plan_video_update(frames[i],flags[0],d,x[4],x[5],metadata,o);snapshot(out,root+"planned/",p);
        for(const auto& [id,mask]:sam3::build_video_outputs(p,d,37,53,o))save(out,root+"output/"+std::to_string(id),mask);
        sam3::finalize_video_scores(p.metadata,frames[i],p.previous_ids,x[5]);snapshot(out,root+"final/",p);metadata=std::move(p.metadata);
      }return out;
    });
}
