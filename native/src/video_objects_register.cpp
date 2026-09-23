// Development comparison adapter; the exported C++ coordinator has no Python dependency.
#include "sam3/video_objects.h"
#include "sam3/multiplex_storage.h"
#include <torch/library.h>
namespace {
using Snapshot=c10::Dict<std::string,at::Tensor>;
void save(Snapshot& out,const std::string& key,const at::Tensor& tensor){if(tensor.defined())out.insert(key,tensor.cpu().clone());}
void snapshot(Snapshot& out,const std::string& key,const sam3::Sam3TrackingSession& session){
  for(bool cond:{true,false})for(const auto& f:cond?session.state().history.conditioning:session.state().history.tracked){
    const auto root=key+(cond?"cond/":"tracked/")+std::to_string(f.index)+"/";
    save(out,root+"low",f.low_mask);save(out,root+"memory",f.memory);save(out,root+"position",f.memory_position);save(out,root+"pointer",f.pointer);save(out,root+"logits",f.object_logits);
  }
}
void snapshot(Snapshot& out,const std::string& key,const sam3::Sam31TrackingSession& session){
  for(bool cond:{true,false})for(const auto& stored:cond?session.state().history.conditioning:session.state().history.tracked){
    const auto f=sam3::load_multiplex_frame(stored);const auto root=key+(cond?"cond/":"tracked/")+std::to_string(f.index)+"/";
    save(out,root+"low",f.masks.low_res_mask);save(out,root+"memory",f.memory);save(out,root+"position",f.memory_position);save(out,root+"pointer",f.pointer);save(out,root+"logits",f.masks.object_logits);
  }
  if(session.state().buckets){const auto& b=*session.state().buckets;std::vector<int64_t> flat;for(const auto& row:b.assignments())flat.insert(flat.end(),row.begin(),row.end());save(out,key+"layout",at::tensor(flat,at::kLong).reshape({b.bucket_count(),b.width()}));}
}
template<class Sessions,class Factory> Snapshot execute(Sessions& sessions,const Factory& factory,const std::vector<std::vector<int64_t>>& operations,const std::vector<at::Tensor>& payloads,sam3::VideoObjectPlacement policy){
  Snapshot out;TORCH_CHECK(operations.size()==payloads.size(),"operation/payload mismatch");
  for(size_t step=0;step<operations.size();++step){const auto& op=operations[step];TORCH_CHECK(!op.empty(),"empty operation");const auto root=std::to_string(step)+"/";
    if(op[0]==0){TORCH_CHECK(op.size()>=2,"addition needs a frame");const std::vector<int64_t> ids(op.begin()+2,op.end());int64_t index;
      if constexpr(std::is_same_v<Sessions,sam3::Sam31VideoSessions>)index=sam3::add_video_objects(op[1],ids,payloads[step],sessions,factory,policy);
      else index=sam3::add_video_objects(op[1],ids,payloads[step],sessions,factory);
      save(out,root+"destination",at::scalar_tensor(index,at::kLong));
    }else if(op[0]==1)sam3::remove_video_objects(std::vector<int64_t>(op.begin()+1,op.end()),sessions);
    else if(op[0]==2){TORCH_CHECK(op.size()==4,"propagation needs start/steps/reverse");sam3::TrackingPropagation request;request.start=op[1];request.max_steps=op[2];request.reverse=op[3];
      for(size_t i=0;i<sessions.size();++i){int64_t count=0;sessions[i]->propagate(request,[&](const auto& value){const auto key=root+"state"+std::to_string(i)+"/out"+std::to_string(count++)+"/";save(out,key+"frame",at::scalar_tensor(value.index,at::kLong));save(out,key+"masks",value.masks);save(out,key+"low",value.low_masks);save(out,key+"ids",at::tensor(value.object_ids,at::kLong));return true;});}
    }else if(op[0]==3)sessions.clear();else TORCH_CHECK(false,"unknown pool operation");
    save(out,root+"states",at::scalar_tensor(int64_t(sessions.size()),at::kLong));
    for(size_t i=0;i<sessions.size();++i){const auto key=root+"state"+std::to_string(i)+"/";save(out,key+"ids",at::tensor(sessions[i]->object_ids(),at::kLong));snapshot(out,key,*sessions[i]);}
  }return out;
}
}
TORCH_LIBRARY_FRAGMENT(sam3_native,m){
  m.def("video_object_destination(int[] slots, int count, int policy) -> int",[](const std::vector<int64_t>& slots,int64_t count,int64_t policy){return sam3::video_object_destination(slots,count,static_cast<sam3::VideoObjectPlacement>(policy));});
  m.def("prepare_video_object_masks(Tensor logits) -> Tensor",&sam3::prepare_video_object_masks);
  m.def("video_objects(str directory, bool multiplex, Tensor[][] arrays, int[][] operations, Tensor[] payloads, int[] settings, bool offload, str mode, str history_directory='', int policy=0) -> Dict(str, Tensor)",
    [](const std::string& directory,bool mux,const std::vector<std::vector<at::Tensor>>& arrays,const std::vector<std::vector<int64_t>>& operations,const std::vector<at::Tensor>& payloads,const std::vector<int64_t>& settings,bool offload,const std::string& mode,const std::string& history,int64_t policy){
      TORCH_CHECK(!arrays.empty() && settings.size()==3,"invalid pool test inputs");const auto device=arrays[0][0].device();
      const auto feature=[&](int64_t frame,int offset){const auto& a=arrays.at(frame%arrays.size());return sam3::TrackingFeatures{a.at(offset),a.at(offset+1),{a.at(offset+2),a.at(offset+3)}};};
      if(mux){const auto core=std::make_shared<sam3::Sam31TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(directory)),device);sam3::Sam31VideoSessions sessions;
        sam3::MultiplexSessionOptions options;options.offload_state=offload;options.non_overlap_output=false;options.history_directory=std::filesystem::u8path(history);options.frame.temporal.memory_slots=3;options.frame.temporal.max_pointer_frames=4;options.frame.temporal.max_conditioning_frames=2;
        sam3::Sam31SessionFactory factory=[&]{return std::make_unique<sam3::Sam31TrackingSession>(core,[&](int64_t frame){return sam3::MultiplexTrackingFeatures{feature(frame,0),feature(frame,4)};},settings[0],settings[1],settings[2],device,mode,options);};return execute(sessions,factory,operations,payloads,static_cast<sam3::VideoObjectPlacement>(policy));
      }
      const auto core=std::make_shared<sam3::Sam3TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(directory)),device);sam3::Sam3VideoSessions sessions;
      sam3::TrackingSessionOptions options;options.offload_state=offload;options.non_overlap_output=false;options.clear_near_input=false;options.frame.temporal.memory_slots=3;options.frame.temporal.max_pointer_frames=4;options.frame.temporal.max_conditioning_frames=2;
      sam3::Sam3SessionFactory factory=[&]{return std::make_unique<sam3::Sam3TrackingSession>(core,[&](int64_t frame){return feature(frame,0);},settings[0],settings[1],settings[2],device,mode,options);};return execute(sessions,factory,operations,payloads,static_cast<sam3::VideoObjectPlacement>(policy));
    });
}
