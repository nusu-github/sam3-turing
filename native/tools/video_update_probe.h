#pragma once
#include "sam3/video_update.h"
#include <type_traits>
// Actual-weight integration probe with synthetic projected features and controlled
// detector/global-mask inputs. Not a coherent visual-model accuracy benchmark.
template<class Sessions,class Factory>
void check_video_update(Sessions& sessions,const Factory& factory,at::Device device){
  constexpr bool mux=std::is_same_v<Sessions,sam3::Sam31VideoSessions>;
  sam3::VideoUpdateOptions options;options.association.policy=options.recondition.policy=mux?sam3::AssociationPolicy::Sam31:sam3::AssociationPolicy::Sam3;
  options.confirmation_enabled=true;options.confirmation_threshold=2;options.recondition.period=1;options.hotstart.delay=10;options.hotstart.unmatched_threshold=1;
  auto meta=sam3::initialize_video_metadata(1,device);const auto tensor_options=at::TensorOptions().device(device).dtype(at::kFloat);
  auto masks=at::full({2,288,288},-2.,tensor_options);masks[0].slice(0,10,100).slice(1,20,120).fill_(2);masks[1].slice(0,140,260).slice(1,150,270).fill_(2);
  sam3::VideoDetections detections{masks,at::full({2},.95,tensor_options),at::tensor({.1f,.1f,.4f,.4f,.5f,.5f,.9f,.9f},tensor_options).reshape({2,4}),at::ones({2},tensor_options.dtype(at::kBool))};
  auto empty=at::empty({0,288,288},tensor_options),empty_scores=at::empty({0},tensor_options);
  auto plan=sam3::plan_video_update(0,false,detections,empty,empty_scores,meta,options);
  sam3::execute_video_update(0,0,plan,detections,sessions,factory,options);meta=plan.metadata;
  TORCH_CHECK(meta.object_ids()==std::vector<int64_t>({0,1}) && sessions.size()==1,"frame update lost initial births");
  if constexpr(mux)TORCH_CHECK(meta.buckets_per_rank[0]==sessions[0]->state().buckets->bucket_count(),"bucket workload was not refreshed");
  sam3::TrackingPropagation request;request.start=1;request.max_steps=0;request.encode_memory=false;
  std::vector<at::Tensor> low,scores;std::vector<int64_t> ids;
  for(auto& session:sessions)session->propagate(request,[&](const auto& value){low.push_back(value.low_masks.squeeze(1).to(at::kFloat));scores.push_back(value.object_logits.reshape({-1}));ids.insert(ids.end(),value.object_ids.begin(),value.object_ids.end());return true;});
  const auto rows=sam3::video_memory_rows(ids,{meta.object_ids()})[0];const auto indices=at::tensor(rows,tensor_options.dtype(at::kLong));
  auto global=at::cat(low).index_select(0,indices),logits=at::cat(scores).index_select(0,indices);
  detections.masks=global.clone();
  plan=sam3::plan_video_update(1,false,detections,global,logits,meta,options);
  sam3::execute_video_update(1,0,plan,detections,sessions,factory,options);
  const auto outputs=sam3::build_video_outputs(plan,detections,37,53,options);TORCH_CHECK(outputs.size()>=2,"frame output assembly lost IDs");
  sam3::finalize_video_scores(plan.metadata,1,plan.previous_ids,logits);meta=plan.metadata;
  // Force a known unmatched/removal decision while executing the real neural
  // memory encoder on the stored current frame before deleting all states.
  detections={empty,empty_scores,at::empty({0,4},tensor_options),at::empty({0},tensor_options.dtype(at::kBool))};
  global=at::ones({int64_t(meta.object_ids().size()),288,288},tensor_options);logits=at::zeros({global.size(0)},tensor_options);
  plan=sam3::plan_video_update(1,false,detections,global,logits,meta,options);
  sam3::execute_video_update(1,0,plan,detections,sessions,factory,options);sam3::finalize_video_scores(plan.metadata,1,plan.previous_ids,logits);
  TORCH_CHECK(sessions.empty() && plan.metadata.object_ids().empty(),"frame update did not remove unmatched states");
  for(auto id:plan.previous_ids)TORCH_CHECK(plan.metadata.object_scores.at(id)==-1e4 && plan.metadata.frame_scores.at(1).at(id).template item<float>()==.5f,"final score write order changed");
}
