#include "sam3/occlusion.h"
#include "sam3/autocast.h"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
  at::set_num_threads(4);const at::Device device(argc>1?argv[1]:"cpu");const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);const auto masks=at::ones({2,3,3},opts),last=at::tensor({2,4},at::kLong).to(device),removed=at::tensor({0,1},at::kLong).to(device).to(at::kBool);
  const auto forward=sam3::update_occlusion(masks,last,removed,5);TORCH_CHECK(forward.suppressed[1].item<bool>() && !forward.suppressed[0].item<bool>() && forward.masks[1].eq(-10).all().item<bool>(),"forward suppression failed");
  // FP16 mask-IoU intersection overflows at 288*288 pixels in the source.
  // Default policy arithmetic must stay FP32 under neural autocast.
  {
    sam3::AutocastGuard outer(device.type(),true,at::kHalf);
    const auto dense=sam3::update_occlusion(at::ones({2,288,288},opts),last,removed,5);
    TORCH_CHECK(dense.suppressed[1].item<bool>() && !dense.suppressed[0].item<bool>(),"default occlusion count inherited neural autocast");
  }
  const auto reverse=sam3::update_occlusion(masks,last,removed,5,true);TORCH_CHECK(reverse.suppressed[0].item<bool>() && !reverse.suppressed[1].item<bool>(),"reverse suppression failed");
  const auto no_history=at::zeros({2},removed.options());const auto absent=sam3::update_occlusion(masks,last,removed,5,false,{},no_history);TORCH_CHECK(!absent.suppressed.any().item<bool>(),"never-occluded object suppressed another");
  const auto allow=sam3::update_occlusion(masks,at::full_like(last,-1),removed,5,false,{sam3::AssociationPolicy::Sam31,.5,true});TORCH_CHECK(allow.suppressed[1].item<bool>(),"SAM3.1 allow-unoccluded/removal policy failed");
  auto state=sam3::extend_device_hotstart(sam3::empty_device_hotstart(device),2,0);const auto occluded=sam3::update_device_occlusion(state,masks,removed,5,false,.5,true);TORCH_CHECK(occluded.state.last_occluded[1].item<int64_t>()==5 && state.last_occluded[1].item<int64_t>()==-1,"hotstart state hook modified input");
  const auto boxes=sam3::tracking_mask_boxes(at::stack({at::zeros({1,3,3},opts.dtype(at::kBool)),at::ones({1,3,3},opts.dtype(at::kBool))}));TORCH_CHECK(boxes.scalar_type()==at::kInt && boxes[0].eq(0).all().item<bool>() && boxes[1][0][2].item<int>()==2,"mask boxes changed inclusive/empty semantics");
  sam3::AssociationMetadata association;association.track_to_recondition_detection={{3,1},{9,0}};association.recondition_order={9,3};const auto candidates=sam3::recondition_candidates(association);TORCH_CHECK(candidates[0].first==9,"candidate insertion order lost");
  const auto geometry=at::tensor({0.f,0.f,2.f/3,2.f/3,0.f,0.f,.1f,.1f}).view({2,4}).to(device),scores=at::ones({2},opts);
  const auto gate=sam3::recondition_decision(candidates,{9,3},geometry,scores,masks,1,{sam3::AssociationPolicy::Sam31,-1,.5,.5});TORCH_CHECK(!gate.geometry,"SAM3.1 gate must inspect first pair only");
  const auto reordered=sam3::recondition_decision({{3,1},{9,0}},{9,3},geometry,scores,masks,1,{sam3::AssociationPolicy::Sam31,-1,.5,.5});TORCH_CHECK(reordered.geometry,"SAM3.1 first-pair order not honored");
  const auto raw=sam3::prepare_recondition_masks(candidates,{9,3},masks,scores,masks,6,6,sam3::AssociationPolicy::Sam3,0);const auto sigmoid=sam3::prepare_recondition_masks(candidates,{9,3},masks,scores,masks,6,6,sam3::AssociationPolicy::Sam31,0);TORCH_CHECK(raw.ids.size()==2 && sigmoid.ids.empty(),"source raw/sigmoid score policies merged");
  TORCH_CHECK(masks.eq(1).all().item<bool>(),"input masks mutated");std::cout<<"occlusion/reconditioning policy/order/state hook passed "<<device<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
