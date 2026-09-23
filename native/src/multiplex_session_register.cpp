#include "sam3/multiplex_session.h"
#include "sam3/video_recondition.h"
#include "sam3/video_memory.h"
#include "sam3/multiplex_storage.h"
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(sam3_native,m) {
  m.def("multiplex_session(str directory, Tensor[][] features, int[][] operations, Tensor[] payloads, int[] settings, bool[] flags, str mode, str history_directory=\"\") -> Dict(str, Tensor)",
    [](const std::string& directory,const std::vector<std::vector<at::Tensor>>& features,const std::vector<std::vector<int64_t>>& operations,const std::vector<at::Tensor>& payloads,const std::vector<int64_t>& settings,const c10::List<bool>& flags,const std::string& mode,const std::string& history_directory) {
      TORCH_CHECK(!features.empty() && settings.size()==3 && flags.size()==5 && operations.size()==payloads.size(),"invalid session test arguments");const auto device=features[0][0].device();
      const auto core=std::make_shared<sam3::Sam31TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(directory)),device);sam3::MultiplexSessionOptions options;options.history_directory=std::filesystem::u8path(history_directory);options.offload_state=flags[0];options.non_overlap_output=flags[1];options.all_edits_conditioning=flags[2];options.frame.temporal.select_by_score=flags[3];options.frame.non_overlap_memory=flags[4];options.frame.temporal.memory_slots=3;options.frame.temporal.max_pointer_frames=4;options.frame.temporal.max_conditioning_frames=2;
      sam3::Sam31TrackingSession session(core,[&](int64_t index){const auto& x=features[index%features.size()];TORCH_CHECK(x.size()==8,"two feature pyramids required");return sam3::MultiplexTrackingFeatures{{x[0],x[1],{x[2],x[3]}},{x[4],x[5],{x[6],x[7]}}};},settings[0],settings[1],settings[2],device,mode,options);
      c10::Dict<std::string,at::Tensor> result;const auto save=[&](const std::string& key,const at::Tensor& value){if(value.defined())result.insert(key,value.cpu().clone());};
      for(size_t i=0;i<operations.size();++i){const auto& op=operations[i];TORCH_CHECK(op.size()>=2 && ((op[0]==8 || op[0]==9 || op[0]==10)?op.size()>=3:op.size()==8),"invalid operation fields");const auto prefix=std::to_string(i)+"/";int64_t count=0;
        const auto emit=[&](const sam3::TrackingSessionOutput& value){const auto key=prefix+"out"+std::to_string(count++)+"/";save(key+"frame",at::scalar_tensor(value.index,at::kLong));save(key+"ids",at::tensor(value.object_ids,at::kLong));save(key+"masks",value.masks);save(key+"low",value.low_masks);save(key+"logits",value.object_logits);};
        if(op[0]==0 || op[0]==1){sam3::TrackingPoints prompt;prompt.normalized=op[4];if(op[0]==0){prompt.points=payloads[i].slice(1,0,2);prompt.labels=payloads[i].select(1,2);}else prompt.box=payloads[i];emit(session.add_points(op[1],op[2],prompt,op[3],op[5]));}
        else if(op[0]==2)emit(session.add_mask(op[1],op[2],payloads[i]));
        else if(op[0]==8)emit(session.add_masks(op[1],std::vector<int64_t>(op.begin()+2,op.end()),payloads[i]));
        else if(op[0]==9)emit(session.recondition_masks(op[1],std::vector<int64_t>(op.begin()+2,op.end()),payloads[i]));
        else if(op[0]==10){sam3::ReconditionMasks prepared;prepared.ids={op.begin()+2,op.end()};prepared.binary_masks=payloads[i].gt(0);const auto executed=sam3::execute_reconditioning(op[1],prepared,std::vector<sam3::Sam31TrackingSession*>{&session});save(prefix+"affected",at::tensor(std::vector<int64_t>(executed.affected_ids.begin(),executed.affected_ids.end()),at::kLong));}
        else if(op[0]==11)sam3::update_video_memories(op[1],payloads[i].to(device),session.object_ids(),std::vector<sam3::Sam31TrackingSession*>{&session},bool(op[3]));
        else if(op[0]==3)session.preflight(op[3]);
        else if(op[0]==4){sam3::TrackingPropagation request;if(op[1]>=0)request.start=op[1];if(op[2]>=0)request.max_steps=op[2];request.reverse=op[3];request.encode_memory=op[4];request.preflight=op[5];session.propagate(request,[&](const auto& value){emit(value);if(op[6]>0 && count>=op[6]){if(op[7])session.cancel();else return false;}return true;});}
        else if(op[0]==5)emit(session.clear_input(op[1],op[2]));else if(op[0]==6)session.remove_object(op[2],op[3]);else if(op[0]==7)session.reset();else TORCH_CHECK(false,"unknown operation");
        save(prefix+"outputs",at::scalar_tensor(count,at::kLong));save(prefix+"ids",at::tensor(session.object_ids(),at::kLong));
        const auto& state=session.state();save(prefix+"status",at::tensor({int64_t(state.started),state.first_annotation.value_or(-1)},at::kLong));
        for(const bool cond:{true,false})for(const auto& stored:cond?state.history.conditioning:state.history.tracked){const auto frame=sam3::load_multiplex_frame(stored);const auto key=prefix+(cond?"cond/":"tracked/")+std::to_string(frame.index)+"/";save(key+"low",frame.masks.low_res_mask);save(key+"memory_masks",frame.memory_masks);save(key+"memory_scores",frame.memory_object_logits);save(key+"memory",frame.memory);save(key+"pointer",frame.pointer);save(key+"logits",frame.masks.object_logits);save(key+"position",frame.memory_position);save(key+"conditions",at::tensor(frame.conditioning_objects,at::kLong));}
        int64_t resident=0,archived=0;
        for(const auto* group:{&state.history.conditioning,&state.history.tracked})for(const auto& frame:*group){
          for(const auto& value:{frame.memory,frame.memory_position,frame.image,frame.image_position,frame.pointer,frame.masks.low_res_mask,frame.masks.high_res_mask,frame.masks.object_logits,frame.iou,frame.memory_masks,frame.memory_object_logits})if(value.defined())resident+=value.nbytes();
          if(frame.archive)archived+=frame.archive->bytes();
        }
        save(prefix+"resident_history_bytes",at::scalar_tensor(resident,at::kLong));save(prefix+"archived_history_bytes",at::scalar_tensor(archived,at::kLong));
        for(const auto& object:state.objects){for(const auto& [index,point]:object.points)save(prefix+"points/"+std::to_string(object.id)+"/"+std::to_string(index),point.points);for(const auto& [index,video]:object.video_edits){const auto key=prefix+"video/"+std::to_string(object.id)+"/"+std::to_string(index);save(key,video);save(key+"/strides",at::tensor(video.strides().vec(),at::kLong));}}
      }return result;
    });
}
