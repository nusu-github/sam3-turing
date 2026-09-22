#include "sam3/association.h"
#include "sam3/autocast.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
  const at::Device device(argc>1?argv[1]:"cpu");const std::string mode=argc>2?argv[2]:"fp32";at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);
  const auto det=at::tensor({1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,1.f}).view({2,2,2}).to(device);
  const auto trk=at::tensor({1.f,-1.f,-1.f,-1.f,-1.f,-1.f,1.f,-1.f,-1.f,-1.f,-1.f,-1.f}).view({3,2,2}).to(device);
  const auto scores=at::full({2},.9,det.options());
  for(auto policy:{sam3::AssociationPolicy::Sam3,sam3::AssociationPolicy::Sam31}){
    sam3::AssociationOptions options;options.policy=policy;if(policy==sam3::AssociationPolicy::Sam31)options.pad_tracks_to=16;
    const auto value=sam3::associate_tracking(det,scores,trk,{},options,mode);const auto out=sam3::realize_association(value,{5,9,12},policy);
    TORCH_CHECK(out.new_detections==std::vector<int64_t>{1} && out.unmatched_tracks==std::vector<int64_t>{9} && out.empty_tracks==std::vector<int64_t>{12},"incorrect new/unmatched/empty result");
    TORCH_CHECK(out.detection_to_tracks.at(0)==std::vector<int64_t>{5} && out.detection_to_tracks.at(1).empty() && out.track_to_recondition_detection.at(5)==0,"incorrect matching/reconditioning");
    const auto empty=sam3::realize_association(sam3::associate_tracking(det.slice(0,0,0),scores.slice(0,0,0),trk,{},options,mode),{5,9,12},policy);
    TORCH_CHECK(empty.unmatched_tracks.size()==(policy==sam3::AssociationPolicy::Sam3?2:3),"model-specific empty-detection policy lost");
    const auto low=at::zeros_like(scores);const auto initial=sam3::realize_association(sam3::associate_tracking(det,low,trk.slice(0,0,0),{},options,mode),{},policy);
    TORCH_CHECK(initial.new_detections.size()==(policy==sam3::AssociationPolicy::Sam3?2:0),"model-specific no-track policy lost");
  }
  // Association count precision is independent of neural autocast precision.
  {sam3::AutocastGuard neural_mode(device.type(),true,at::kHalf);
    sam3::AssociationOptions options;options.use_iom=true;
    const auto full=at::ones({1,288,288},det.options());
    const auto dense=sam3::associate_tracking(full,scores.slice(0,0,1),full,{},options);
    TORCH_CHECK(dense.matches.item<bool>() && !dense.is_new.item<bool>(),"default FP32 association overflowed under outer FP16 autocast");
  }
  TORCH_CHECK(sam3::assign_detection_devices(8,{1,0},3)==std::vector<int64_t>({1,1,1,0,0,0,1,1}),"multiplex device placement changed");
  TORCH_CHECK(sam3::assign_detection_devices(8,{1,0})==std::vector<int64_t>({1,0,1,0,1,0,1,0}),"single-object device placement changed");
  const auto boxes=at::tensor({.0f,.4f,.05f,.6f,.4f,.4f,.6f,.6f}).view({2,4}).to(device);const auto keep=sam3::detection_boundary_keep(boxes).cpu();TORCH_CHECK(!keep[0].item<bool>() && keep[1].item<bool>(),"strict boundary threshold changed");
  std::cout<<"native association/matching/empty-policy/placement passed "<<device<<" "<<mode<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
