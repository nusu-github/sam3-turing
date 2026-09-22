#include "sam3/tensor_archive.h"
#include <c10/core/InferenceMode.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <random>
#include <zlib.h>
namespace sam3 {
namespace {
uint32_t checksum(const at::Tensor& value){
  auto crc=crc32(0L,Z_NULL,0);auto* bytes=static_cast<const Bytef*>(value.const_data_ptr());uint64_t left=value.nbytes();
  while(left){const auto count=static_cast<uInt>(std::min<uint64_t>(left,1ULL<<30));crc=crc32(crc,bytes,count);bytes+=count;left-=count;}return crc;
}
void write_bytes(std::ostream& out,const at::Tensor& value){
  auto* bytes=static_cast<const char*>(value.const_data_ptr());uint64_t left=value.nbytes();
  while(left){const auto count=static_cast<std::streamsize>(std::min<uint64_t>(left,1ULL<<30));out.write(bytes,count);TORCH_CHECK(out,"cannot write tensor history archive");bytes+=count;left-=count;}
}
}
TensorArchive::TensorArchive(std::filesystem::path parent){
  TORCH_CHECK(!parent.empty(),"history directory is required");std::filesystem::create_directories(parent);
  static std::atomic<uint64_t> sequence{0};std::random_device random;
  for(int attempt=0;attempt<64;++attempt){
    const auto name="sam3-history-"+std::to_string(random())+"-"+std::to_string(sequence.fetch_add(1));auto candidate=parent/name;
    if(std::filesystem::create_directory(candidate)){directory_=std::move(candidate);path_=directory_/"tensors.bin";return;}
  }
  TORCH_CHECK(false,"could not reserve a unique history archive directory");
}
TensorArchive::~TensorArchive(){std::error_code error;std::filesystem::remove(path_,error);std::filesystem::remove(directory_,error);}
std::shared_ptr<const TensorArchive> TensorArchive::write(const std::filesystem::path& parent,const std::map<std::string,at::Tensor>& values){
  c10::InferenceMode inference;auto archive=std::shared_ptr<TensorArchive>(new TensorArchive(parent));std::ofstream out(archive->path_,std::ios::binary);TORCH_CHECK(out,"cannot open tensor history archive");
  for(const auto& [name,value]:values){
    if(!value.defined())continue;TORCH_CHECK(value.layout()==at::kStrided && !value.is_quantized(),"history requires ordinary strided tensors");
    auto reduced=value.sizes().vec();for(int64_t i=0;i<value.dim();++i){TORCH_CHECK(value.stride(i)>=0,"negative history stride");if(value.stride(i)==0 && reduced[i]>1)reduced[i]=1;}
    auto data=value.as_strided(reduced,value.strides()).cpu().contiguous();
    Record record{archive->bytes_,data.nbytes(),checksum(data),value.scalar_type(),value.device(),value.sizes().vec(),value.strides().vec(),reduced};
    write_bytes(out,data);archive->bytes_+=data.nbytes();archive->records_.emplace(name,std::move(record));
  }
  out.flush();TORCH_CHECK(out,"cannot flush tensor history archive");out.close();TORCH_CHECK(out,"cannot close tensor history archive");return archive;
}
at::Tensor TensorArchive::read(const std::string& name) const{
  c10::InferenceMode inference;const auto found=records_.find(name);if(found==records_.end())return at::Tensor();const auto& r=found->second;
  std::ifstream input(path_,std::ios::binary);TORCH_CHECK(input,"cannot read tensor history archive: ",path_.u8string());input.seekg(r.offset);TORCH_CHECK(input,"cannot seek tensor history archive");
  auto packed=at::empty(r.reduced,at::TensorOptions().dtype(r.dtype));auto* bytes=static_cast<char*>(packed.mutable_data_ptr());uint64_t left=r.bytes;
  while(left){const auto count=static_cast<std::streamsize>(std::min<uint64_t>(left,1ULL<<30));TORCH_CHECK(input.read(bytes,count),"truncated tensor history archive");bytes+=count;left-=count;}
  TORCH_CHECK(checksum(packed)==r.crc,"tensor history checksum mismatch: ",name);
  auto result=at::empty_strided(r.reduced,r.strides,packed.options().device(r.device));result.copy_(packed);
  return result.as_strided(r.shape,r.strides);
}
}
