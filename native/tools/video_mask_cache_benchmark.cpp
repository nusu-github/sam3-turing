#include "sam3/video_mask_cache.h"
#include "sam3/video_interaction.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_BENCH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace {
void sync(at::Device d) {
#ifdef SAM3_BENCH_CUDA
 if(d.is_cuda())c10::cuda::device_synchronize();
#endif
}
int64_t allocated(at::Device d) {
#ifdef SAM3_BENCH_CUDA
 if(d.is_cuda())return c10::cuda::CUDACachingAllocator::getDeviceStats(d.index()).allocated_bytes[0].current;
#endif
 return 0;
}
}
int main(int argc,char** argv){try{
 TORCH_CHECK(argc==8,"DEVICE FRAMES OBJECTS HEIGHT WIDTH DIRECTORY REPORT.json");
 c10::InferenceMode inference;at::set_num_threads(4);
 const auto device=at::empty({0},at::TensorOptions().device(at::Device(argv[1]))).device();
#ifndef SAM3_BENCH_CUDA
 TORCH_CHECK(device.is_cpu(),"build with CUDA statistics support");
#endif
 const int64_t frames=std::stoll(argv[2]),objects=std::stoll(argv[3]),h=std::stoll(argv[4]),w=std::stoll(argv[5]);
 TORCH_CHECK(frames>0 && objects>0 && h>0 && w>0,"positive dimensions required");
 const auto plane=at::arange(h*w,at::TensorOptions().dtype(at::kLong).device(device)).view({1,h,w});
 std::map<int64_t,at::Tensor> masks;for(int64_t id=0;id<objects;++id)masks[id]=plane.remainder(11+id).lt(3+id%5);
 sam3::VideoOutput input;input.cached_masks=masks;
 std::ofstream report(std::filesystem::u8path(argv[7]));TORCH_CHECK(report,"cannot open report");
 report<<"{\"device\":\""<<device<<"\",\"frames\":"<<frames<<",\"objects\":"<<objects<<",\"height\":"<<h<<",\"width\":"<<w<<",\"cases\":[";
 for(int mode=0;mode<3;++mode){
  sync(device);const auto before=allocated(device);
  std::unique_ptr<sam3::VideoInteractionState> resident;
  std::unique_ptr<sam3::VideoMaskCache> packed;
  if(mode==0)resident=std::make_unique<sam3::VideoInteractionState>(sam3::AssociationPolicy::Sam3,frames,h,w);
  else packed=std::make_unique<sam3::VideoMaskCache>(h,w,mode==1?sam3::VideoMaskStorage::PackedCPU:sam3::VideoMaskStorage::PackedDisk,std::filesystem::u8path(argv[6]));
  const auto start=std::chrono::steady_clock::now();
  for(int64_t frame=0;frame<frames;++frame){if(resident)resident->record(frame,input);else packed->store(frame,masks);}
  sync(device);const auto stored=allocated(device)-before;
  const auto store_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  double read_ms=0;
  for(const auto frame:{int64_t(0),frames/2,frames-1}){
   sync(device);const auto t=std::chrono::steady_clock::now();
   auto output=resident?resident->cached_frames().at(frame):packed->read(frame);
   sync(device);read_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
   TORCH_CHECK(output.size()==masks.size(),"object count changed");
   for(const auto& [id,mask]:output)TORCH_CHECK(mask.device()==masks.at(id).device() && mask.strides()==masks.at(id).strides() && at::equal(mask,masks.at(id)),"restored mask differs");
  }
  sam3::VideoMaskCacheStats stats;if(packed)stats=packed->stats();
  if(mode)report<<',';
  report<<"{\"storage\":"<<mode<<",\"gpu_bytes_measured\":"<<(device.is_cuda()?"true":"false")<<",\"retained_gpu_bytes\":"<<stored<<",\"logical_mask_bytes\":"<<frames*objects*h*w<<",\"packed_bytes\":"<<stats.packed_bytes<<",\"disk_bytes\":"<<stats.disk_bytes<<",\"store_ms\":"<<store_ms<<",\"mean_read_ms\":"<<read_ms/3<<",\"exact\":true}";
  resident.reset();packed.reset();sync(device);TORCH_CHECK(allocated(device)==before,"cache retained GPU tensors after destruction");
 }
 report<<"]}\n";std::cout<<"cache round trips, accounting and cleanup passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
