#include "sam3/turing_attention.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>
double measure(const std::function<at::Tensor()>& fn) {
  for(int i=0;i<5;++i)fn();
  std::vector<float> times;
  for(int i=0;i<20;++i) {
    at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);
    a.record();auto out=fn();b.record();b.synchronize();times.push_back(a.elapsed_time(b));
  }
  std::sort(times.begin(),times.end());return .5*(times[9]+times[10]);
}
int main(int argc,char** argv) {try {
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: sam3_turing_attention_bench OUTPUT.json [--check-only]");
  c10::InferenceMode inference;c10::cuda::set_device(0);at::manual_seed(7429);
  std::ofstream file(argv[1]);file<<"{\"tolerance_max_abs\":0.02,\"tolerance_rmse\":0.002,\"resources\":[";
  for(int i=0;i<2;++i) {
    if(i)file<<',';auto r=sam3::turing_attention_resources(i!=0);
    file<<"{\"registers\":"<<r[0]<<",\"local_bytes\":"<<r[1]<<",\"static_shared\":"<<r[2]
        <<",\"dynamic_shared\":"<<r[3]<<",\"active_blocks_per_sm\":"<<r[4]<<"}";
  }
  file<<"],\"cases\":[";bool first=true;
  // All work is produced on a non-default stream, including Q/K/V writes.
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  for(const auto& shape:std::vector<std::pair<int,int>>{{1,1},{1,64},{1,128},{2,129},{9,576},{1,5184}}) {
    const int batch=shape.first,length=shape.second,heads=16;
    auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
    auto packed=at::randn({batch,length,3,heads,64},options);
    auto q=packed.select(2,0).transpose(1,2),k=packed.select(2,1).transpose(1,2),v=packed.select(2,2).transpose(1,2);
    // Match production RoPE layout for Q/K, keep packed non-contiguous V.
    q=q.transpose(1,2).contiguous().transpose(1,2);
    k=k.transpose(1,2).contiguous().transpose(1,2);
    auto baseline=[&] {return at::scaled_dot_product_attention(q,k,v);};
    auto small=[&] {return sam3::turing_attention(q,k,v,false);};
    auto large=[&] {return sam3::turing_attention(q,k,v,true);};
    auto ref=baseline();auto a=small(),b=large();
    stream.synchronize();
    TORCH_CHECK(at::equal(a,b),"shared-memory reservation changes output");
    if(length<=129)
      ref=at::matmul(at::softmax(at::matmul(q.to(at::kFloat),k.to(at::kFloat).transpose(-1,-2))*.125,-1),v.to(at::kFloat)).to(at::kHalf);
    auto diff=a.to(at::kFloat)-ref.to(at::kFloat);
    const float maxabs=diff.abs().max().item<float>(),rmse=diff.square().mean().sqrt().item<float>();
    TORCH_CHECK(at::isfinite(a).all().item<bool>() && maxabs<=.02 && rmse<=.002,
                "attention error exceeds preset tolerance at N=",length," max=",maxabs," rmse=",rmse);
    std::vector<double> ts,ta,tb;
    if(length>=576 && argc==2) {
      for(int pass=0;pass<4;++pass) {
        if(pass%2) {tb.push_back(measure(large));ta.push_back(measure(small));ts.push_back(measure(baseline));}
        else {ts.push_back(measure(baseline));ta.push_back(measure(small));tb.push_back(measure(large));}
      }
    }
    if(!first)file<<',';first=false;
    file<<"{\"batch\":"<<batch<<",\"length\":"<<length<<",\"max_abs\":"<<maxabs<<",\"rmse\":"<<rmse;
    auto write=[&](const char* name,const auto& values) {file<<",\""<<name<<"\":[";for(size_t i=0;i<values.size();++i){if(i)file<<',';file<<values[i];}file<<']';};
    write("sdpa_ms",ts);write("donor32_ms",ta);write("donor64_ms",tb);file<<'}';file.flush();
    std::cout<<batch<<" x "<<length<<" max="<<maxabs<<" rmse="<<rmse<<std::endl;
  }
  file<<"]}\n";TORCH_CHECK(file,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
