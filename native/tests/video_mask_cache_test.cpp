#include "sam3/video_mask_cache.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {try {
  TORCH_CHECK(argc==2,"expected cpu|cuda");
  c10::InferenceMode inference;at::set_num_threads(2);const at::Device device(argv[1]);
  const auto root=std::filesystem::temp_directory_path()/
      ("sam3-mask-cache-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);std::ofstream(root/"caller-owned.txt")<<"keep";
  int cases=0;
  for(auto storage:{sam3::VideoMaskStorage::PackedCPU,sam3::VideoMaskStorage::PackedDisk}) {
    const auto path=root/(storage==sam3::VideoMaskStorage::PackedCPU?"cpu":"disk");
    sam3::VideoMaskCache cache(3,5,storage,path);
    auto base=at::arange(15,at::TensorOptions().dtype(at::kLong).device(device)).remainder(3).eq(0).view({1,5,3}).transpose(1,2);
    auto expected=base.clone();std::map<int64_t,at::Tensor> input;
    for(int64_t id=-1;id<256;++id)input.emplace(id,base);
    cache.store(0,input);cache.store(1,{{9,base}});cache.store(2,{});
    base.zero_();const auto stats=cache.stats();
    TORCH_CHECK(stats.frames==3 && stats.masks==258 && stats.logical_bytes==258*15 && stats.packed_bytes==258*2,"cache accounting/cap");
    TORCH_CHECK(cache.contains(2) && cache.read(2).empty(),"empty frame lost");
    auto got=cache.read(0);TORCH_CHECK(got.size()==257,"object count changed");
    for(const auto& [id,x]:got)TORCH_CHECK(x.device()==expected.device() && x.strides()==expected.strides() && at::equal(x,expected),"mask round trip");
    got.begin()->second.zero_();TORCH_CHECK(at::equal(cache.read(0).begin()->second,expected),"read aliases cache");
    bool rejected=false;try{cache.store(0,{{9,base.to(at::kFloat)}});}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected && cache.read(0).size()==257,"failed replacement changed cache");
    cache.forget_object(9);TORCH_CHECK(cache.read(1).empty() && !cache.read(0).count(9),"object removal");
    cache.store(0,{{999,base}});TORCH_CHECK(cache.read(0).size()==1 && !cache.read(0).at(999).any().item<bool>(),"replacement");
    cache.erase(0);TORCH_CHECK(cache.frames()==std::vector<int64_t>({1,2}),"frame index");
    sam3::VideoMaskCache moved(std::move(cache));moved.clear();TORCH_CHECK(moved.stats().frames==0 && moved.stats().disk_bytes==0,"clear/move");
    if(std::filesystem::exists(path))TORCH_CHECK(std::filesystem::is_empty(path),"archive files leaked");
    ++cases;
  }
  // CRC and truncation errors do not silently turn into missing/empty masks.
  {
    const auto path=root/"rollback",parked=root/"parked";
    sam3::VideoMaskCache first(3,5,sam3::VideoMaskStorage::PackedDisk,path);
    sam3::VideoMaskCache second(3,5,sam3::VideoMaskStorage::PackedDisk,path);
    auto value=at::ones({1,3,5},at::TensorOptions().dtype(at::kBool).device(device));
    first.store(0,{{1,value}});second.store(0,{{2,value}});
    std::filesystem::rename(path,parked);std::ofstream(path)<<"blocks directory creation";
    bool rejected=false;try{first.store(0,{{3,value}});}catch(const std::exception&){rejected=true;}
    std::filesystem::remove(path);std::filesystem::rename(parked,path);
    TORCH_CHECK(rejected && first.read(0).count(1) && !first.read(0).count(3),"I/O replacement lost prior frame");
    first.clear();TORCH_CHECK(second.read(0).at(2).all().item<bool>(),"cleanup removed another cache's files");
    second.clear();TORCH_CHECK(std::filesystem::is_empty(path),"shared-parent cleanup");
  }
  const auto path=root/"corruption";
  sam3::VideoMaskCache cache(3,5,sam3::VideoMaskStorage::PackedDisk,path);
  cache.store(4,{{7,at::ones({1,3,5},at::TensorOptions().dtype(at::kBool).device(device))}});
  std::filesystem::path file;
  for(const auto& entry:std::filesystem::recursive_directory_iterator(path))if(entry.path().filename()=="tensors.bin")file=entry.path();
  TORCH_CHECK(!file.empty(),"missing archive");
  {std::fstream io(file,std::ios::in|std::ios::out|std::ios::binary);char x;io.read(&x,1);x^=1;io.seekp(0);io.write(&x,1);}
  int rejected=0;try{cache.read(4);}catch(const c10::Error&){++rejected;}
  std::filesystem::resize_file(file,0);try{cache.read(4);}catch(const c10::Error&){++rejected;}
  TORCH_CHECK(rejected==2 && cache.contains(4),"corruption not reported");cache.clear();
  TORCH_CHECK(std::filesystem::exists(root/"caller-owned.txt") && std::filesystem::is_empty(path),"cleanup affected caller data");
  std::filesystem::remove_all(root);
  std::cout<<"PASS lossless mask cache "<<device<<" policies="<<cases<<" corruption="<<rejected<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
