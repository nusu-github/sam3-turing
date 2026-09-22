#include "sam3/association.h"
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(sam3_native,m){
  m.def("associate_tracking(Tensor detections, Tensor scores, Tensor tracks, Tensor? keep, int[] ids, bool multiplex, float new_threshold, float track_threshold, float detection_threshold, float high_confidence, bool iom, float iom_threshold, float iou_threshold, int padding, str mode) -> Dict(str,Tensor)",
    [](const at::Tensor& det,const at::Tensor& scores,const at::Tensor& tracks,const std::optional<at::Tensor>& keep,const std::vector<int64_t>& ids,bool multiplex,double nt,double tt,double dt,double hc,bool iom,double im,double iu,int64_t padding,const std::string& mode){
      sam3::AssociationOptions options;options.policy=multiplex?sam3::AssociationPolicy::Sam31:sam3::AssociationPolicy::Sam3;options.new_detection_threshold=nt;options.track_match_threshold=tt;options.detection_match_threshold=dt;options.high_confidence_threshold=hc;options.use_iom=iom;options.iom_recondition_threshold=im;options.iou_recondition_threshold=iu;options.pad_tracks_to=padding;
      const auto tensors=sam3::associate_tracking(det,scores,tracks,keep.value_or(at::Tensor()),options,mode);const auto metadata=sam3::realize_association(tensors,ids,options.policy);
      c10::Dict<std::string,at::Tensor> out;const auto values=tensors.tensors();const std::vector<std::string> names{"unmatched","nonempty","is_new","best_track","high_confidence","high_overlap","keep","matches"};for(size_t i=0;i<values.size();++i)out.insert(names[i],values[i]);
      const auto put=[&](const std::string& name,const std::vector<int64_t>& values){out.insert(name,at::tensor(values,at::kLong));};put("new_detections",metadata.new_detections);put("unmatched_tracks",metadata.unmatched_tracks);put("empty_tracks",metadata.empty_tracks);
      std::vector<int64_t> keys,offsets{0},matched,recondition_ids,recondition_dets;
      for(const auto& [id,tracks]:metadata.detection_to_tracks){keys.push_back(id);matched.insert(matched.end(),tracks.begin(),tracks.end());offsets.push_back(matched.size());}
      for(const auto& [id,det]:metadata.track_to_recondition_detection){recondition_ids.push_back(id);recondition_dets.push_back(det);}
      put("detection_keys",keys);put("match_offsets",offsets);put("matched_ids",matched);put("recondition_order",metadata.recondition_order);put("recondition_ids",recondition_ids);put("recondition_detections",recondition_dets);return out;
    });
  m.def("assign_detection_devices(int count, int[] previous, int capacity) -> int[]",&sam3::assign_detection_devices);
  m.def("detection_boundary_keep(Tensor boxes, float margin) -> Tensor",&sam3::detection_boundary_keep);
}
