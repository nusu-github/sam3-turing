#include "sam3/video_predictor.h"
#include "sam3/ops.h"
#include "ppm.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_BENCH_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <tuple>

namespace {
void sync(at::Device device) {
#ifdef SAM3_BENCH_CUDA
  if(device.is_cuda())c10::cuda::device_synchronize();
#endif
}
void equal(const sam3::VideoOutput& a,const sam3::VideoOutput& b) {
  auto tensor=[](const at::Tensor& x,const at::Tensor& y){
    TORCH_CHECK(x.device()==y.device() && x.scalar_type()==y.scalar_type() &&
                    x.strides()==y.strides() && at::equal(x,y),"fetch tensor changed");
  };
  tensor(a.ids,b.ids);tensor(a.probabilities,b.probabilities);
  tensor(a.boxes_xywh,b.boxes_xywh);tensor(a.masks,b.masks);tensor(a.centers,b.centers);
  TORCH_CHECK(a.frame_stats==b.frame_stats && a.cached_masks.size()==b.cached_masks.size(),"fetch metadata changed");
  for(const auto& [id,x]:a.cached_masks)tensor(x,b.cached_masks.at(id));
}
using Snapshot=std::map<std::filesystem::path,std::pair<uintmax_t,std::filesystem::file_time_type>>;
Snapshot snapshot(const std::filesystem::path& path) {
  Snapshot out;
  if(std::filesystem::exists(path))for(const auto& e:std::filesystem::recursive_directory_iterator(path))
    if(e.is_regular_file())out.emplace(e.path(),std::make_pair(e.file_size(),e.last_write_time()));
  return out;
}
void dump(const std::filesystem::path& dir,int64_t frame,const sam3::VideoOutput& out) {
  std::map<std::string,at::Tensor> fields{{"ids",out.ids},{"probabilities",out.probabilities},
      {"boxes",out.boxes_xywh},{"masks",sam3::pack_masks(out.masks)},{"centers",out.centers}};
  for(const auto& [id,mask]:out.cached_masks)fields.emplace("cached-"+std::to_string(id),sam3::pack_masks(mask));
  std::ofstream metadata(dir/(std::to_string(frame)+"-metadata.txt"));
  for(const auto& [name,tensor]:fields) {
    metadata<<name<<' '<<tensor.scalar_type()<<' '<<tensor.device()<<' '<<tensor.sizes()<<' '<<tensor.strides()<<'\n';
    auto cpu=tensor.cpu().contiguous();std::ofstream file(dir/(std::to_string(frame)+"-"+name+".bin"),std::ios::binary);
    file.write(static_cast<const char*>(cpu.const_data_ptr()),cpu.nbytes());TORCH_CHECK(file,"cannot write output");
  }
  for(const auto& [id,mask]:out.cached_masks)metadata<<"dense-"<<id<<' '<<mask.device()<<' '<<mask.sizes()<<' '<<mask.strides()<<'\n';
  for(const auto& [name,value]:out.frame_stats)metadata<<"stat-"<<name<<' '<<value<<'\n';
  metadata.close();TORCH_CHECK(metadata,"cannot write metadata");
}
}
int main(int argc,char** argv){try {
  TORCH_CHECK(argc==11,"STORE sam3|sam3.1 DEVICE MODE FRAMES.txt BPE.gz OUTPUT STORAGE EXPECT_READ_ONLY SAMPLES");
  c10::InferenceMode inference;at::set_num_threads(4);
  at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  const std::string model=argv[2];TORCH_CHECK(model=="sam3" || model=="sam3.1","invalid model");
  const auto device=at::empty({0},at::TensorOptions().device(at::Device(argv[3]))).device();
  c10::DeviceGuard device_guard(device);
  const auto manifest=std::filesystem::u8path(argv[5]),root=std::filesystem::u8path(argv[7]);
  const int storage=std::stoi(argv[8]),expect_read_only=std::stoi(argv[9]),samples=std::stoi(argv[10]);
  TORCH_CHECK((storage==1 || storage==2) && (expect_read_only==0 || expect_read_only==1) && samples>0,"invalid probe options");
  std::ifstream input(manifest);TORCH_CHECK(input,"cannot read manifest");std::string line;std::vector<std::filesystem::path> files;
  while(std::getline(input,line)){if(!line.empty() && line.back()=='\r')line.pop_back();if(line.empty() || line[0]=='#')continue;auto p=std::filesystem::u8path(line);files.push_back(p.is_absolute()?p:manifest.parent_path()/p);}
  TORCH_CHECK(files.size()>=4,"fixture requires four frames");std::filesystem::create_directories(root/"outputs");
  const auto rgb=sam3::cli::read_ppm(files.front());
  auto options=sam3::video_predictor_defaults(model=="sam3"?sam3::AssociationPolicy::Sam3:sam3::AssociationPolicy::Sam31);
  options.mode=argv[4];options.centers=true;options.sam31_session.history_directory=root/"history";
  sam3::VideoPredictor predictor(sam3::WeightStore(std::filesystem::u8path(argv[1])),std::filesystem::u8path(argv[6]),
      [&](int64_t f){return sam3::cli::read_ppm(files.at(f));},files.size(),rgb.size(1),rgb.size(2),device,options);
  predictor.set_output_cache(static_cast<sam3::VideoOutputCacheStorage>(storage),root/"cache");
  sam3::VideoSemanticPrompt prompt;prompt.text="person";predictor.add_prompt(0,prompt);
  sam3::VideoPredictorPropagation request;request.start=0;request.max_steps=3;
  predictor.propagate(request,[](int64_t,const auto&){return true;});
  std::map<int64_t,sam3::VideoOutput> expected;
  for(int64_t frame=0;frame<4;++frame){expected.emplace(frame,predictor.fetch(frame));dump(root/"outputs",frame,expected.at(frame));}
  const auto indices=predictor.cached_frame_indices();const auto actions=predictor.action_count(),encodes=predictor.visual_encodes();
  for(int i=0;i<3;++i)predictor.fetch(i%4);
  const auto before=snapshot(root/"cache");TORCH_CHECK(storage!=2 || !before.empty(),"fixture lacks disk masks");
  sync(device);int64_t peak_extra=0;
#ifdef SAM3_BENCH_CUDA
  int64_t allocated=0;if(device.is_cuda()){
    const int index=c10::cuda::current_device();allocated=c10::cuda::CUDACachingAllocator::getDeviceStats(index).allocated_bytes[0].current;
    c10::cuda::CUDACachingAllocator::resetPeakStats(index);
  }
#endif
  std::vector<double> ms;
  for(int i=0;i<samples;++i){sync(device);const auto start=std::chrono::steady_clock::now();
    const auto output=predictor.fetch(i%4);sync(device);
    ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
#ifdef SAM3_BENCH_CUDA
    if(device.is_cuda())peak_extra=std::max(peak_extra,c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device()).allocated_bytes[0].peak-allocated);
#endif
    equal(output,expected.at(i%4));
  }
  const auto after=snapshot(root/"cache");const bool unchanged=before==after;
  TORCH_CHECK(storage!=2 || unchanged==bool(expect_read_only),"unexpected archive rewrite behavior");
  TORCH_CHECK(predictor.cached_frame_indices()==indices && predictor.action_count()==actions && predictor.visual_encodes()==encodes,"fetch changed owner observations");
  {
    c10::InferenceMode ordinary(false);
    auto output=predictor.fetch(0);equal(output,expected.at(0));
    for(auto& [id,mask]:output.cached_masks){TORCH_CHECK(!mask.is_inference(),"ordinary C++ fetch became an inference-only tensor");mask.logical_not_();}
    equal(predictor.fetch(0),expected.at(0));
  }
  auto modified=predictor.fetch(0);modified.masks.logical_not_();for(auto& [id,mask]:modified.cached_masks)mask.logical_not_();
  equal(predictor.fetch(0),expected.at(0));
  int64_t callbacks=0;const auto before_replay=snapshot(root/"cache");
  predictor.propagate(request,[&](int64_t frame,const auto& output){equal(output,expected.at(frame));++callbacks;return true;});
  TORCH_CHECK(callbacks==4 && predictor.visual_encodes()==encodes,"cached propagation encoded frames");
  if(expect_read_only)TORCH_CHECK(before_replay==snapshot(root/"cache"),"fetch propagation rewrote cache");
  if(storage==2){
    const auto file=snapshot(root/"cache").begin()->first;char original;
    {std::fstream io(file,std::ios::in|std::ios::out|std::ios::binary);io.read(&original,1);TORCH_CHECK(io,"read corruption fixture");const char bad=original^1;io.seekp(0);io.write(&bad,1);}
    int failures=0;for(int64_t frame=0;frame<4;++frame)try{predictor.fetch(frame);}catch(const c10::Error&){++failures;}
    TORCH_CHECK(failures==1 && predictor.cached_frame_indices()==indices,"corrupt read lost cached frames");
    {std::fstream io(file,std::ios::in|std::ios::out|std::ios::binary);io.write(&original,1);TORCH_CHECK(io,"restore corruption fixture");}
    for(int64_t frame=0;frame<4;++frame)equal(predictor.fetch(frame),expected.at(frame));
  }
  const auto stats=predictor.output_cache_stats();TORCH_CHECK(!stats.inspection_pinned && stats.resident_bytes==0 && stats.packed_bytes>0,"fetch retained resident masks");
  const auto ids=predictor.metadata().object_ids();for(auto id:ids)predictor.remove_object(id);
  for(int64_t frame=0;frame<4;++frame){const auto empty=predictor.fetch(frame);TORCH_CHECK(empty.ids.numel()==0 && empty.masks.numel()==0 && empty.cached_masks.empty() && empty.masks.device().is_cpu(),"empty packed frame changed");}
  TORCH_CHECK(predictor.cached_frame_indices()==indices,"empty packed frames lost their index");
  const auto& legacy=predictor.interaction();TORCH_CHECK(predictor.output_cache_stats().inspection_pinned && legacy.cached_frames().size()==indices.size(),"legacy inspection changed");
  TORCH_CHECK(predictor.fetch(0).cached_masks.empty(),"legacy fetch restored removed masks");
  predictor.reset();TORCH_CHECK(predictor.cached_frame_indices().empty() && !predictor.output_cache_stats().inspection_pinned && int(predictor.output_cache_stats().storage)==storage && snapshot(root/"cache").empty(),"reset leaked cache state");
  std::ofstream report(root/"report.json");report<<"{\"storage\":"<<storage<<",\"archive_unchanged\":"<<(unchanged?"true":"false")<<",\"peak_extra_cuda_bytes\":"<<peak_extra<<",\"samples_ms\":[";
  for(size_t i=0;i<ms.size();++i)report<<(i?",":"")<<ms[i];
  report<<"],\"exact\":true,\"isolated\":true,\"cached_propagation\":true,\"corruption_recovery\":"<<(storage==2?"true":"null")<<",\"legacy_pin_reset\":true}\n";report.close();TORCH_CHECK(report,"cannot write report");
  std::cout<<"fetch outputs, isolation, archive identity and lifecycle passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
