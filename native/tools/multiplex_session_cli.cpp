#include "sam3/multiplex_session.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){
  try{
    TORCH_CHECK(argc==4,"usage: sam3_multiplex_session STORE cpu|cuda fp32|fp16|bf16_reference");at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[2]);const std::string mode=argv[3];const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
    const auto core=std::make_shared<sam3::Sam31TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(argv[1])),device);
    int64_t loads=0;bool fail=false;
    const auto provider=[&](int64_t index){TORCH_CHECK(!fail || index!=4,"injected provider failure");++loads;const auto neck=[&]{return sam3::TrackingFeatures{at::randn({1,256,72,72},opts),at::randn({1,256,72,72},opts),{at::randn({1,32,288,288},opts),at::randn({1,64,144,144},opts)}};};return sam3::MultiplexTrackingFeatures{neck(),neck()};};
    sam3::MultiplexSessionOptions options;options.offload_state=true;options.frame.temporal.memory_slots=3;options.frame.temporal.max_pointer_frames=4;options.frame.temporal.select_by_score=true;
    sam3::Sam31TrackingSession session(core,provider,5,37,53,device,mode,options);
    sam3::TrackingPoints points{at::rand({17,2},opts),at::ones({17},opts.dtype(at::kInt)),{},true};
    auto first=session.add_points(0,101,points);TORCH_CHECK(first.object_ids==std::vector<int64_t>{101},"initial object ID incorrect");
    auto brush=at::zeros({37,53},opts);brush.slice(0,5,30).slice(1,9,44).fill_(1);session.add_mask(0,202,brush);
    TORCH_CHECK(loads==1 && session.state().objects[0].points.at(0).points.size(1)==17,"cache or full point retention failed");
    int64_t callbacks=0;const auto emit=[&](const sam3::TrackingSessionOutput& out){++callbacks;TORCH_CHECK(out.masks.sizes()==at::IntArrayRef({int64_t(out.object_ids.size()),1,37,53}) && at::isfinite(out.masks).all().item<bool>(),"invalid session output");return true;};
    sam3::TrackingPropagation request;request.start=0;request.max_steps=2;session.propagate(request,emit);
    sam3::TrackingPoints one{at::tensor({.4f,.6f},opts).reshape({1,2}),at::ones({1},opts.dtype(at::kInt)),{},true};
    session.add_points(2,303,one);TORCH_CHECK(session.object_ids()==std::vector<int64_t>({101,202,303}) && session.state().buckets->bucket_count()==2,"midstream insertion failed");
    session.add_points(1,101,one);session.add_points(1,101,points,false);TORCH_CHECK(session.state().objects[0].points.at(1).points.size(1)==18,"refinement dropped points");
    request.start=3;request.max_steps=3;request.reverse=true;session.propagate(request,emit);
    session.remove_object(202,true);TORCH_CHECK(session.object_ids()==std::vector<int64_t>({101,303}),"removal did not remap IDs");session.clear_input(1,101);
    request.start=0;request.max_steps=3;request.reverse=false;int64_t before=callbacks;session.propagate(request,[&](const auto& out){emit(out);session.cancel();return true;});TORCH_CHECK(callbacks==before+1,"cancellation failed");
    request.start=1;request.max_steps=2;session.propagate(request,emit);
    fail=true;const auto ids=session.object_ids();bool rejected=false;try{session.add_points(4,999,one);}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected && session.object_ids()==ids,"failed new-object edit changed session IDs");fail=false;
    session.reset();TORCH_CHECK(session.object_ids().empty() && !session.state().buckets && session.state().history.conditioning.empty(),"reset retained state");
    session.add_points(4,909,{{},{},at::tensor({.1f,.2f,.7f,.8f},opts),true});request.start=4;request.max_steps=0;session.propagate(request,emit);session.remove_object(909,true);TORCH_CHECK(session.object_ids().empty(),"last-object removal failed");
    const auto batched=session.add_masks(0,{700,800},at::stack({brush,brush}));TORCH_CHECK((batched.masks.select(2,10).select(2,15)<0).all().item<bool>(),"simultaneous brushes did not mutually suppress overlap");session.preflight();session.reset();
    std::cout<<"native SAM3.1 session passed: "<<callbacks<<" callbacks; 18 accumulated points, masks, box, midstream add, refinement, reverse, removal/clear, cancel/resume, rollback/reset; feature loads="<<loads<<"; no Python\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
