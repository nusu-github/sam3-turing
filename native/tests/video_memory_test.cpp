#include "sam3/video_memory.h"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
  at::set_num_threads(4);const at::Device device(argc>1?argv[1]:"cpu");const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
  const auto low=at::ones({2,3,5},opts);const auto input=sam3::prepare_video_memory(low,sam3::AssociationPolicy::Sam3);
  TORCH_CHECK(input.high_masks.sizes()==at::IntArrayRef({2,1,1152,1152}) && input.object_logits.scalar_type()==at::kFloat,"invalid memory input layout/dtype");
  TORCH_CHECK(input.high_masks[0].eq(1).all().item<bool>() && input.high_masks[1].eq(-10).all().item<bool>() && input.object_logits[0].item<float>()==10 && input.object_logits[1].item<float>()==-10,"global tie/shrink suppression mismatch");
  const auto negative=at::full({1,3,5},-.2,opts);
  const auto sam3=sam3::prepare_video_memory(negative,sam3::AssociationPolicy::Sam3),sam31=sam3::prepare_video_memory(negative,sam3::AssociationPolicy::Sam31),warmup=sam3::prepare_video_memory(negative,sam3::AssociationPolicy::Sam3,false);
  TORCH_CHECK(sam3.high_masks.eq(-10).all().item<bool>() && at::equal(sam31.high_masks,warmup.high_masks) && sam31.high_masks.gt(-1).all().item<bool>(),"singleton/warmup source policies merged");
  const auto empty=sam3::prepare_video_memory(at::empty({0,3,5},opts),sam3::AssociationPolicy::Sam31);TORCH_CHECK(empty.high_masks.size(0)==0 && empty.object_logits.sizes()==at::IntArrayRef({0,1}),"empty memory input invalid");
  const auto rows=sam3::video_memory_rows({30,10,50,20},{{20,30},{50},{},{10}});TORCH_CHECK(rows==std::vector<std::vector<int64_t>>({{3,0},{2},{},{1}}),"ID-based memory assignment reordered local rows");
  bool rejected=false;try{sam3::video_memory_rows({1},{{2}});}catch(const c10::Error&){rejected=true;}TORCH_CHECK(rejected,"missing global memory ID accepted");
  TORCH_CHECK(low.eq(1).all().item<bool>(),"preparation mutated input");std::cout<<"global memory preparation/ID assignment passed "<<device<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
