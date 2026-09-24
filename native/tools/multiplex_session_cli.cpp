#include "sam3/multiplex_session.h"
#include "sam3/video_recondition.h"
#include "sam3/video_memory.h"
#include "sam3/video_objects.h"
#include "video_update_probe.h"
#include "sam3/multiplex_storage.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){
  try{
    TORCH_CHECK(argc==4 || argc==5,"usage: sam3_multiplex_session STORE cpu|cuda fp32|fp16|bf16_reference [HISTORY_DIRECTORY]");at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[2]);const std::string mode=argv[3];const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
    const auto core=std::make_shared<sam3::Sam31TrackingFrame>(sam3::WeightStore(std::filesystem::u8path(argv[1])),device);
    int64_t loads=0;bool fail=false;
    const auto provider=[&](int64_t index){TORCH_CHECK(!fail || index!=4,"injected provider failure");++loads;const auto neck=[&]{return sam3::TrackingFeatures{at::randn({1,256,72,72},opts),at::randn({1,256,72,72},opts),{at::randn({1,32,288,288},opts),at::randn({1,64,144,144},opts)}};};return sam3::MultiplexTrackingFeatures{neck(),neck()};};
    sam3::MultiplexSessionOptions options;options.offload_state=true;if(argc==5)options.history_directory=std::filesystem::u8path(argv[4]);options.frame.temporal.memory_slots=3;options.frame.temporal.max_pointer_frames=4;options.frame.temporal.select_by_score=true;
    sam3::Sam31TrackingSession session(core,provider,5,37,53,device,mode,options);
    sam3::TrackingPoints points{at::rand({17,2},opts),at::ones({17},opts.dtype(at::kInt)),{},true};
    auto first=session.add_points(0,101,points);TORCH_CHECK(first.object_ids==std::vector<int64_t>{101},"initial object ID incorrect");
    auto brush=at::zeros({37,53},opts);brush.slice(0,5,30).slice(1,9,44).fill_(1);session.add_mask(0,202,brush);
    TORCH_CHECK(loads==1 && session.state().objects[0].points.at(0).points.size(1)==17,"cache or full point retention failed");
    int64_t callbacks=0;const auto emit=[&](const sam3::TrackingSessionOutput& out){++callbacks;TORCH_CHECK(out.masks.sizes()==at::IntArrayRef({int64_t(out.object_ids.size()),1,37,53}) && at::isfinite(out.masks).all().item<bool>(),"invalid session output");return true;};
    sam3::TrackingPropagation request;request.start=0;request.max_steps=2;session.propagate(request,emit);
    sam3::TrackingPoints one{at::tensor({.4f,.6f},opts).reshape({1,2}),at::ones({1},opts.dtype(at::kInt)),{},true};
    if(!options.history_directory.empty()){
      const auto snapshot=session.state();const auto source=snapshot.history.tracked.back().archive;TORCH_CHECK(source,"paged history missing archive");
      const auto path=source->path();auto held=path;held+=".held";std::filesystem::rename(path,held);
      bool rejected=false;try{session.add_points(4,999,one);}catch(const std::exception&){rejected=true;}
      bool clear_rejected=false,remove_rejected=false;
      try{session.clear_input(snapshot.history.tracked.back().index,101);}catch(const std::exception&){clear_rejected=true;}
      try{session.remove_object(202,true);}catch(const std::exception&){remove_rejected=true;}
      std::filesystem::rename(held,path);
      TORCH_CHECK(clear_rejected && remove_rejected && session.state().dirty==snapshot.dirty,"failed clear/removal changed edit state");
      TORCH_CHECK(rejected && session.object_ids()==std::vector<int64_t>({101,202}) && session.state().history.tracked.back().archive==source,"failed history read committed partial edit");
      for(size_t i=0;i<snapshot.history.conditioning.size();++i)TORCH_CHECK(session.state().history.conditioning[i].archive==snapshot.history.conditioning[i].archive,"rollback replaced earlier archive");
      size_t directories=0;for(const auto& entry:std::filesystem::directory_iterator(options.history_directory))if(entry.is_directory())++directories;
      TORCH_CHECK(directories==snapshot.history.conditioning.size()+snapshot.history.tracked.size(),"failed transaction leaked temporary archives");
    }
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
    const auto batched=session.add_masks(0,{700,800},at::stack({brush,brush}));TORCH_CHECK((batched.masks.select(2,10).select(2,15)<0).all().item<bool>(),"simultaneous brushes did not mutually suppress overlap");session.preflight();
    request.start=1;request.max_steps=1;request.reverse=false;request.preflight=false;session.propagate(request,emit);
    const auto layout=session.state().buckets->assignments();const auto ids_before=session.object_ids();const auto dirty_before=session.state().dirty;
    bool unknown=false,duplicate=false,missing_frame=false;
    try{session.recondition_masks(1,{999},brush.unsqueeze(0));}catch(const c10::Error&){unknown=true;}
    try{session.recondition_masks(1,{700,700},at::stack({brush,brush}));}catch(const c10::Error&){duplicate=true;}
    try{session.recondition_masks(4,{700},brush.unsqueeze(0));}catch(const c10::Error&){missing_frame=true;}
    TORCH_CHECK(unknown && duplicate && missing_frame && session.object_ids()==ids_before && session.state().dirty==dirty_before && session.state().buckets->assignments()==layout,"invalid reconditioning changed session metadata");
    sam3::ReconditionMasks prepared;prepared.ids={800,700};prepared.binary_masks=at::stack({brush.gt(0),brush.flip({0}).gt(0)});
    sam3::Sam31TrackingSession partner(core,provider,5,37,53,device,mode,options);
    partner.add_masks(0,{700,900},at::stack({brush,brush}));partner.preflight();
    const auto partner_layout=partner.state().buckets->assignments();
    const auto executed=sam3::execute_reconditioning(1,prepared,std::vector<sam3::Sam31TrackingSession*>{&session,&partner});
    TORCH_CHECK(executed.affected_ids==std::set<int64_t>({700,800}) && executed.preflight_states==std::vector<int64_t>({0,1}) && executed.edited_states==std::vector<int64_t>{0} && session.state().dirty.empty(),"reconditioning did not preflight affected state");
    TORCH_CHECK(session.state().buckets->assignments()==layout && session.object_ids()==ids_before,"reconditioning changed object layout");
    TORCH_CHECK(partner.state().buckets->assignments()==partner_layout && partner.object_ids()==std::vector<int64_t>({700,900}),"shared-ID preflight modified partner layout");
    auto before_memory=sam3::load_multiplex_frame(session.state().history.conditioning.back());
    sam3::update_video_memories(1,at::ones({2,5,7},opts),{800,700},std::vector<sam3::Sam31TrackingSession*>{&session},true);
    for(const auto& stored:session.state().history.conditioning)if(stored.index==1){
      const auto frame=sam3::load_multiplex_frame(stored);
      TORCH_CHECK(at::equal(before_memory.masks.low_res_mask,frame.masks.low_res_mask) && at::equal(before_memory.masks.object_logits,frame.masks.object_logits),"memory rewrite changed predictions");
      TORCH_CHECK(frame.memory_object_logits[0].item<float>()==-10 && frame.memory_object_logits[1].item<float>()==10 && frame.memory_masks.size(-1)==1152,"memory rows ignored global IDs");
    }
    request.start=2;request.max_steps=2;request.reverse=true;session.propagate(request,emit);session.reset();partner.reset();
    sam3::Sam31VideoSessions pool;
    sam3::Sam31SessionFactory factory=[&]{return std::make_unique<sam3::Sam31TrackingSession>(core,provider,5,37,53,device,mode,options);};
    const auto logits=brush*2-1;
    sam3::add_video_objects(0,{10,20},at::stack({logits,logits}),pool,factory);
    TORCH_CHECK(sam3::add_video_objects(1,{30},logits.unsqueeze(0),pool,factory)==0 && pool.size()==1,"best-fit did not reuse free slots");
    std::vector<int64_t> many_ids;for(int64_t i=100;i<117;++i)many_ids.push_back(i);
    TORCH_CHECK(sam3::add_video_objects(1,many_ids,logits.unsqueeze(0).expand({17,37,53}),pool,factory)==1 && pool[1]->object_ids().size()==17,"larger than one bucket group was capped");
    const auto before_ids=pool[0]->object_ids();bool strict_rejected=false;
    try{pool[0]->remove_objects({20,999},true);}catch(const c10::Error&){strict_rejected=true;}
    TORCH_CHECK(strict_rejected && pool[0]->object_ids()==before_ids,"strict batch removal partially committed");
    sam3::remove_video_objects({10,30,999,10},pool);TORCH_CHECK(pool[0]->object_ids()==std::vector<int64_t>{20},"batch removal lost surviving ID");
    sam3::remove_video_objects(many_ids,pool);TORCH_CHECK(pool.size()==1,"empty pooled state retained");sam3::remove_video_objects({20},pool);TORCH_CHECK(pool.empty(),"pool did not empty");
    check_video_update(pool,factory,device);
    std::cout<<"native SAM3.1 session passed: "<<callbacks<<" callbacks; 18 accumulated points, masks, box, midstream add, refinement, reverse, removal/clear, cancel/resume, rollback/reset, reconditioning/preflight/global memory; feature loads="<<loads<<"; no Python\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
