#include "sam3/hotstart.h"
#include "sam3/autocast.h"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
  at::set_num_threads(4);const at::Device device(argc>1?argv[1]:"cpu");const auto boolean=at::TensorOptions().device(device).dtype(at::kBool);
  sam3::HotstartOptions o;o.delay=3;o.unmatched_threshold=1;o.duplicate_threshold=1;o.suppress_only_within_hotstart=false;
  sam3::AssociationMetadata host_association;host_association.detection_to_tracks[0]={9,3};
  const auto host=sam3::update_host_hotstart({},host_association,{9,3},0,false,o);TORCH_CHECK(host.newly_removed==std::set<int64_t>{3},"host tie must retain first ID in match order");
  auto state=sam3::extend_device_hotstart(sam3::empty_device_hotstart(device),2,0);sam3::AssociationTensors a;a.unmatched=at::zeros({2},boolean);a.nonempty=at::ones({2},boolean);a.matches=at::ones({1,2},boolean);
  const auto tied=sam3::update_device_hotstart(state,a,0,false,o);TORCH_CHECK(!tied.remove.any().item<bool>(),"device policy must not remove equal-first-frame overlap ties");TORCH_CHECK(state.keep_alive.sum().item<int64_t>()==0 && state.overlap_count.sum().item<int64_t>()==0,"input state was mutated");
  state=sam3::extend_device_hotstart(sam3::extend_device_hotstart(sam3::empty_device_hotstart(device),1,0),1,1);
  const auto result=sam3::update_device_hotstart(state,a,1,false,o);TORCH_CHECK(result.remove[1].item<bool>() && !result.remove[0].item<bool>(),"later object not removed");
  const auto compact=sam3::compact_device_hotstart(result.state);TORCH_CHECK(compact.first.size()==1 && compact.second.item<int64_t>()==0,"invalid compaction");
  const auto grown=sam3::extend_device_hotstart(compact.first,2,2,3);TORCH_CHECK(grown.size()==3 && grown.keep_alive[2].item<int64_t>()==3 && grown.last_occluded[2].item<int64_t>()==-1,"invalid extension");
  const auto selected=sam3::select_device_hotstart(grown,at::tensor({2,0},at::kLong).to(device));TORCH_CHECK(selected.first_frame[0].item<int64_t>()==2 && selected.first_frame[1].item<int64_t>()==0,"invalid reorder");
  {sam3::AutocastGuard neural(device.type(),true,at::kBFloat16);a.matches=at::ones({301,2},boolean);const auto counts=sam3::update_device_hotstart(state,a,1,false,o);TORCH_CHECK(counts.state.overlap_count[0][1].item<int64_t>()==301,"overlap counts rounded under neural autocast");}
  auto confirmed=sam3::update_confirmation({}, {},{9,3},{},{9,3},1);confirmed=sam3::update_confirmation(confirmed,{9,3},{3,9},{},{},1);TORCH_CHECK(confirmed.status==std::vector<int64_t>({2,2}) && confirmed.consecutive_detections==std::vector<int64_t>({0,0}),"confirmed status lost on missing detections");
  bool rejected=false;try{sam3::update_host_hotstart(host.state,{}, {9},1,false,o);}catch(const std::exception&){rejected=true;}TORCH_CHECK(rejected && host.state.keep_alive.at(9)==1,"invalid new ID changed prior state");
  a.unmatched=at::empty({0},boolean);a.nonempty=at::empty({0},boolean);a.matches=at::empty({4,0},boolean);TORCH_CHECK(sam3::update_device_hotstart(sam3::empty_device_hotstart(device),a,0).remove.numel()==0,"empty state failed");
  std::cout<<"native hotstart/compaction/extension/confirmation/precision passed "<<device<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
