#include "sam3/tensor_archive.h"
#include "sam3/multiplex_storage.h"
#include <ATen/Parallel.h>
#include <fstream>
#include <iostream>
namespace {
template<class F> void fails(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}TORCH_CHECK(failed,"expected archive failure");}
}
int main(int argc,char** argv){try{
  at::set_num_threads(2);const at::Device device(argc>1?argv[1]:"cpu");
  const auto owner=sam3::TensorArchive::write(std::filesystem::temp_directory_path(),{});
  const auto parent=owner->path().parent_path();
  std::map<std::string,at::Tensor> values;
  for(const auto type:{at::kFloat,at::kHalf,at::kBFloat16,at::kLong,at::kByte,at::kBool}){
    const auto key=std::to_string(int(type));auto x=at::arange(48).reshape({2,3,8}).to(device,type);
    values[key]=x;values[key+"transpose"]=x.transpose(0,2);values[key+"gap"]=x.slice(2,0,8,2);values[key+"expand"]=x.slice(0,0,1).expand({5,3,8});values[key+"empty"]=x.slice(0,0,0);
  }
  auto special=at::empty_strided({1,3,2,2},{3,1,6,3},at::TensorOptions().device(device));special.fill_(2.5);values["singleton-strides"]=special;
  values["scalar"]=at::scalar_tensor(1.25,at::TensorOptions().device(device));values["missing"]=at::Tensor();
  const auto archive=sam3::TensorArchive::write(parent,values);const auto path=archive->path();
  for(const auto& [key,expected]:values){auto actual=archive->read(key);if(!expected.defined()){TORCH_CHECK(!actual.defined(),"undefined tensor changed");continue;}
    TORCH_CHECK(actual.device()==expected.device() && actual.scalar_type()==expected.scalar_type() && actual.sizes()==expected.sizes() && actual.strides()==expected.strides() && at::equal(actual,expected),"archive changed tensor: ",key);
  }
  TORCH_CHECK(!archive->read("unknown").defined(),"missing entry is defined");
  auto second=sam3::TensorArchive::write(parent,values);auto old=second->path();auto retained=second;second.reset();TORCH_CHECK(std::filesystem::exists(old),"released live archive");retained.reset();TORCH_CHECK(!std::filesystem::exists(old),"archive was not reclaimed");
  {std::fstream corrupt(path,std::ios::in|std::ios::out|std::ios::binary);char c;corrupt.read(&c,1);c^=1;corrupt.seekp(0);corrupt.write(&c,1);}
  fails([&]{archive->read(values.begin()->first);});
  std::filesystem::resize_file(path,0);fails([&]{archive->read("scalar");});
  auto restored=sam3::TensorArchive::write(parent,values);std::filesystem::remove(restored->path());fails([&]{restored->read("scalar");});
  const auto file=parent/"not-a-directory";{std::ofstream out(file);out<<"keep";}fails([&]{sam3::TensorArchive::write(file,values);});TORCH_CHECK(std::filesystem::file_size(file)==4,"modified caller file");std::filesystem::remove(file);
  sam3::MultiplexFrameHistory history;
  for(int64_t i=0;i<31;++i){sam3::MultiplexFrame frame;frame.index=i;
    frame.memory=at::full({1,2,2,2},double(i),special.options());frame.memory_position=at::ones_like(frame.memory);
    frame.image=at::ones({4,1,2},special.options());frame.image_position=frame.image;
    frame.pointer=at::full({1,16,2},double(i),special.options());frame.masks.low_res_mask=at::ones({1,1,2,2},special.options());
    sam3::archive_multiplex_frame(frame,parent);(i==0?history.conditioning:history.tracked).push_back(std::move(frame));
  }
  const auto unselected=history.tracked.front().archive->path(),selected=history.tracked.back().archive->path();auto held=unselected;held+=".held";std::filesystem::rename(unselected,held);
  sam3::MultiplexTemporalOptions options;options.memory_slots=3;options.max_pointer_frames=4;
  const auto loaded=sam3::load_selected_multiplex_history(history,31,32,false,options);
  TORCH_CHECK(!loaded.tracked.front().memory.defined() && loaded.tracked.back().memory.defined() && loaded.tracked.back().pointer.defined(),"incorrect selected history loading");
  TORCH_CHECK(!history.tracked.back().memory.defined() && history.tracked.back().archive,"materialization changed stored history");
  for(const auto& frame:loaded.tracked)TORCH_CHECK(!frame.masks.low_res_mask.defined(),"temporal loading read masks");
  std::filesystem::rename(held,unselected);held=selected;held+=".held";std::filesystem::rename(selected,held);
  fails([&]{sam3::load_selected_multiplex_history(history,31,32,false,options);});std::filesystem::rename(held,selected);
  std::cout<<"archive tensors="<<values.size()<<" device="<<device<<" exact values/strides/dtypes, CRC/truncation/missing-file failures, shared lifetime; selective spatial/pointer paging across 31 frames\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
