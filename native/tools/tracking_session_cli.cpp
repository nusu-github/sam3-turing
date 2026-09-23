#include "sam3/tracking_session.h"
#include "sam3/video_recondition.h"
#include "sam3/video_memory.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==4,"usage: sam3_tracking_session STORE cpu|cuda fp32|fp16|bf16_reference");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[2]);const std::string mode=argv[3];
    const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
    const auto core=std::make_shared<sam3::Sam3TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(argv[1])),device);
    int64_t reads=0;sam3::TrackingSessionOptions policy;policy.offload_state=true;
    policy.frame.temporal.memory_slots=3;policy.frame.temporal.max_pointer_frames=4;
    sam3::Sam3TrackingSession session(core,[&](int64_t index){
      ++reads;return sam3::TrackingFeatures{at::full({1,256,72,72},index*.01,options),at::zeros({1,256,72,72},options),
          {at::zeros({1,32,288,288},options),at::zeros({1,64,144,144},options)}};
    },4,73,91,device,mode,policy);
    sam3::TrackingPoints points;points.points=at::full({17,2},.5,options);points.labels=at::ones({17},options.dtype(at::kInt));
    session.add_points(0,101,points);
    TORCH_CHECK(session.state().objects[0].points.at(0).points.size(1)==17,"point input was truncated");
    auto mask=at::zeros({37,53},options);mask.slice(0,4,25).slice(1,8,42).fill_(1);
    session.add_mask(1,202,mask);session.preflight();
    TORCH_CHECK(session.state().history.conditioning.size()==2,"missing consolidated input");
    for(const auto& frame:session.state().history.conditioning)
      TORCH_CHECK(frame.memory.device().is_cpu() && frame.memory.scalar_type()==at::kBFloat16,"memory offload/compression failed");
    int64_t count=0;
    const auto emit=[&](const sam3::TrackingSessionOutput& out) {
      TORCH_CHECK(out.masks.sizes()==at::IntArrayRef({int64_t(out.object_ids.size()),1,73,91}) && at::isfinite(out.masks).all().item<bool>(),"invalid output masks");
      ++count;std::cout<<"frame="<<out.index<<" objects="<<out.object_ids.size()<<" masks="<<out.masks.sizes()<<'\n';return true;
    };
    sam3::TrackingPropagation request;request.start=0;request.preflight=false;
    session.propagate(request,[&](const auto& out){emit(out);if(count==2)session.cancel();return true;});
    TORCH_CHECK(count==2 && session.state().tracked_direction.size()==2,"cancellation did not preserve completed frames");
    request.start=2;session.propagate(request,emit);TORCH_CHECK(count==4,"resume missed frames");
    sam3::ReconditionMasks prepared;prepared.ids={202,101};prepared.binary_masks=at::stack({mask.gt(0),mask.flip({0}).gt(0)});
    const auto executed=sam3::execute_reconditioning(2,prepared,std::vector<sam3::Sam3TrackingSession*>{&session});
    TORCH_CHECK(executed.affected_ids==std::set<int64_t>({101,202}) && executed.preflight_states==std::vector<int64_t>({0,0}),"SAM3 did not preflight after each candidate");
    const auto old=session.state().history.conditioning.back();
    sam3::update_video_memories(2,at::ones({2,5,7},options),{202,101},std::vector<sam3::Sam3TrackingSession*>{&session});
    for(const auto& frame:session.state().history.conditioning)if(frame.index==2)TORCH_CHECK(at::equal(old.low_mask,frame.low_mask) && at::equal(old.object_logits,frame.object_logits),"memory rewrite changed predicted masks/logits");
    session.add_points(2,101,points,true,true);request.start=3;request.reverse=true;request.preflight=true;
    session.propagate(request,emit);TORCH_CHECK(count==8,"reverse propagation missed frames");
    const auto updates=session.remove_object(101,true);
    TORCH_CHECK(session.object_ids()==std::vector<int64_t>{202} && updates.size()==2,"object removal/remapping failed");
    session.clear_input(2,202);session.clear_input(1,202);
    TORCH_CHECK(!session.state().started && session.state().history.conditioning.empty() && session.state().history.tracked.empty(),"last annotation clear did not reset tracking");
    session.add_points(0,202,points);session.reset();TORCH_CHECK(session.object_ids().empty(),"reset retained objects");
    bool failed=false;try {session.remove_object(404,true);}catch(const c10::Error&){failed=true;}
    TORCH_CHECK(failed,"strict removal accepted an unknown ID");
    std::cout<<"native session passed: callbacks="<<count<<" provider_reads="<<reads<<"; synthetic features, no Python\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
