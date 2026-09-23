#include "sam3/video_objects.h"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
  at::set_num_threads(2);const at::Device device(argc>1?argv[1]:"cpu");
  using sam3::VideoObjectPlacement;
  TORCH_CHECK(sam3::video_object_destination({12,3,3,0},3)==1,"best fit/tie changed");
  TORCH_CHECK(sam3::video_object_destination({12,3},13)==-1,"large group was limited");
  TORCH_CHECK(sam3::video_object_destination({0,10},20,VideoObjectPlacement::FirstState)==0,"first state policy changed");
  TORCH_CHECK(sam3::video_object_destination({100},1,VideoObjectPlacement::NewState)==-1,"new state policy changed");
  auto raw=at::tensor({-1.,1.},at::TensorOptions().device(device)).view({2,1,1});const auto mask=sam3::prepare_video_object_masks(raw);
  TORCH_CHECK(mask.sizes()==at::IntArrayRef({2,1152,1152}) && mask.scalar_type()==at::kBool && !mask[0].any().item<bool>() && mask[1].all().item<bool>(),"mask preparation changed");
  sam3::Sam3VideoSessions a;sam3::Sam31VideoSessions b;sam3::Sam3SessionFactory fa;sam3::Sam31SessionFactory fb;
  TORCH_CHECK(sam3::add_video_objects(0,{},raw.slice(0,0,0),a,fa)==-1 && sam3::add_video_objects(0,{},raw.slice(0,0,0),b,fb)==-1,"empty addition invoked factory");
  sam3::remove_video_objects({123},a);sam3::remove_video_objects({123},b);
  bool rejected=false;try{sam3::add_video_objects(0,{7,7},raw,b,fb);}catch(const c10::Error&){rejected=true;}TORCH_CHECK(rejected && b.empty(),"invalid addition changed collection");
  std::cout<<"video object coordinator passed on "<<device<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
