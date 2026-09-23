#include "sam3/weights.h"
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
void integer(std::ostream& out,uint64_t value,int bytes){for(int i=0;i<bytes;++i)out.put(char(value>>(8*i)));}
void string(std::ostream& out,const std::string& value){integer(out,value.size(),4);out.write(value.data(),value.size());}
void index(const std::filesystem::path& root,uint64_t length,uint32_t crc){
  std::ofstream file(root/"weights.s3i",std::ios::binary);file.write("SAM3WGT1",8);integer(file,1,8);
  string(file,"large");string(file,"large.s3w");integer(file,64,8);integer(file,length,8);integer(file,crc,4);integer(file,1,4);integer(file,1,4);integer(file,length,8);
  file.close();TORCH_CHECK(file,"cannot write index");
}
struct Temporary {
  std::filesystem::path root;
  Temporary(){
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    for(int i=0;i<100;++i){root=std::filesystem::temp_directory_path()/("sam3-large-crc-"+std::to_string(stamp)+"-"+std::to_string(i));if(std::filesystem::create_directory(root))return;}
    TORCH_CHECK(false,"cannot create CRC fixture directory");
  }
  ~Temporary(){std::error_code error;std::filesystem::remove_all(root,error);}
};
}
int main(){try{
  c10::InferenceMode inference;Temporary temporary;const auto root=temporary.root;
  const uint64_t length=(1ULL<<30)+17;
  // IEEE CRC32 of 1GiB zeros followed by bytes 01..11. A nonzero tail makes
  // truncating the length or failing to carry CRC state across chunks visible.
  const uint32_t crc=0xa42dcf87U;
  {
    index(root,length,crc);
    std::ofstream data(root/"large.s3w",std::ios::binary);const std::vector<char> zeros(1<<20,0);
    data.write(zeros.data(),64);for(int i=0;i<1024;++i)data.write(zeros.data(),zeros.size());
    for(int i=1;i<=17;++i)data.put(char(i));data.close();TORCH_CHECK(data,"cannot write large payload");
  }
  sam3::WeightStore store(root);
  {
    const auto value=store.read("large");TORCH_CHECK(value.numel()==int64_t(length) && value[0].item<uint8_t>()==0,"large tensor shape/head");
    for(int i=0;i<17;++i)TORCH_CHECK(value[int64_t(length)-17+i].item<uint8_t>()==i+1,"large tensor tail");
  }
  {std::fstream data(root/"large.s3w",std::ios::in|std::ios::out|std::ios::binary);data.seekp(64+length-1);data.put(char(99));data.close();TORCH_CHECK(data,"cannot corrupt tail");}
  bool rejected=false;try{store.read("large");}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"corruption after the CRC chunk boundary was accepted");
  index(root,0,0);TORCH_CHECK(sam3::WeightStore(root).read("large").numel()==0,"empty tensor CRC");
  index(root,0,1);rejected=false;try{sam3::WeightStore(root).read("large");}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"invalid empty tensor CRC was accepted");
  std::cout<<"PASS CRC chunk boundary, tensor tail, empty tensors and corruption rejection\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
