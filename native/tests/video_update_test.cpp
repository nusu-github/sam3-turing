#include "sam3/video_update.h"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
  at::set_num_threads(2);const at::Device device(argc>1?argv[1]:"cpu");const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
  sam3::VideoUpdateOptions options;options.association.policy=options.recondition.policy=sam3::AssociationPolicy::Sam31;options.confirmation_enabled=true;options.confirmation_threshold=2;
  auto previous=sam3::initialize_video_metadata(2,device);sam3::VideoDetections detections{at::ones({17,2,3},opts),at::ones({17},opts),at::zeros({17,4},opts),at::ones({17},opts.dtype(at::kBool))};
  auto first=sam3::plan_video_update(0,false,detections,at::empty({0,2,3},opts),at::empty({0},opts),previous,options);
  TORCH_CHECK(first.new_ids.size()==17 && first.new_ranks[15]==0 && first.new_ranks[16]==1,"multiplex group placement changed");
  TORCH_CHECK(previous.object_ids().empty() && previous.device.size()==0 && previous.frame_scores.empty(),"planning modified previous metadata");
  previous=first.metadata;previous.buckets_per_rank={1,1};const auto old_alive=previous.device.keep_alive.clone();
  detections={-at::ones({1,2,3},opts),at::ones({1},opts),at::zeros({1,4},opts),at::ones({1},opts.dtype(at::kBool))};
  auto second=sam3::plan_video_update(1,false,detections,at::ones({17,2,3},opts),at::zeros({17},opts),previous,options);
  const auto ids=second.metadata.object_ids();TORCH_CHECK(ids[16]==17 && ids[17]==16,"rank-concatenated ID order changed");
  TORCH_CHECK(second.metadata.device.first_frame[16].item<int64_t>()==1 && second.metadata.device.first_frame[17].item<int64_t>()==0,"device metadata rows lost ID alignment");
  TORCH_CHECK(at::equal(previous.device.keep_alive,old_alive) && previous.max_id==16 && previous.frame_scores.size()==1,"planning mutated a shared tensor/map");
  options.hotstart.delay=5;options.hotstart.unmatched_threshold=1;
  detections={at::empty({0,2,3},opts),at::empty({0},opts),at::empty({0,4},opts),at::empty({0},opts.dtype(at::kBool))};
  auto removed=sam3::plan_video_update(2,false,detections,at::ones({18,2,3},opts),at::zeros({18},opts),second.metadata,options);
  TORCH_CHECK(removed.metadata.object_ids().empty() && removed.removed.size()==18 && removed.metadata.device.size()==0,"hotstart removal did not compact metadata");
  const auto outputs=sam3::build_video_outputs(removed,detections,3,5,options);TORCH_CHECK(outputs.size()==18,"raw outputs prematurely filtered removed IDs");
  sam3::finalize_video_scores(removed.metadata,2,removed.previous_ids,at::zeros({18},opts));
  for(auto id:ids)TORCH_CHECK(removed.metadata.object_scores.at(id)==-1e4 && removed.metadata.frame_scores.at(2).at(id).item<float>()==.5,"score write order changed");
  auto overflow=sam3::initialize_video_metadata(1,device);overflow.max_id=INT64_MAX;bool rejected=false;
  detections={at::ones({1,2,3},opts),at::ones({1},opts),at::zeros({1,4},opts),at::ones({1},opts.dtype(at::kBool))};
  try{sam3::plan_video_update(0,false,detections,at::empty({0,2,3},opts),at::empty({0},opts),overflow,options);}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected && overflow.max_id==INT64_MAX,"ID overflow was not rejected without mutation");
  std::cout<<"video planning/rank alignment/score ordering passed on "<<device<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
