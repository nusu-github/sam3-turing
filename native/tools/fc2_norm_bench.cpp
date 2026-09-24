#include "../src/approx_kernels.h"
#include "sam3/vision_fusion.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
double measure(const std::function<void()>& fn) {
  for(int i=0;i<20;++i)fn();
  std::vector<float> t;
  for(int i=0;i<30;++i) {at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);a.record();fn();b.record();b.synchronize();t.push_back(a.elapsed_time(b));}
  std::sort(t.begin(),t.end());return (t[14]+t[15])*.5;
}
int main(int argc,char** argv){try {
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: fc2_norm_bench OUTPUT [--check-only]");
  c10::InferenceMode inf;c10::cuda::CUDAGuard dev(0);at::manual_seed(8923);
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  std::ofstream file(argv[1]);file<<"{\"cases\":[";bool first=true;
  for(auto shape:std::vector<std::vector<int64_t>>{{1,1,1,1024},{1,72,72,1024},{2,48,48,1024}})for(bool partition:{false,true}) {
    if(shape[1]==1 && partition)continue;
    auto x=at::randn(shape,opt);const auto rows=x.numel()/1024;
    auto acc=at::randint(-100000,100001,{rows,1024},opt.dtype(at::kInt));
    auto xs=at::rand({rows},opt)*.002,ws=at::rand({1024},opt)*.005;
    auto bias=(at::randn({1024},opt)*.02).to(at::kHalf),gamma=at::randn({1024},opt),beta=at::randn({1024},opt);
    at::Tensor projection,a,b,c,d;
    auto separate=[&]{projection=at::empty({rows,1024},opt.dtype(at::kHalf));approx_restore(acc,xs,ws,bias,projection,false);std::tie(a,b)=sam3::vision_residual_norm_projection(x,projection.view(shape),gamma,beta,at::kHalf,partition);};
    auto fused=[&]{std::tie(c,d)=sam3::approx_fc2_norm(x,acc,xs,ws,bias,gamma,beta,partition);};
    for(int kind=0;kind<3;++kind) {
      if(kind==1){x.zero_();acc.zero_();bias.zero_();}
      if(kind==2){x.normal_();acc.random_(-100000,100001);bias.normal_(0,.02);}
      separate();fused();
      TORCH_CHECK(at::equal(a.view(at::kInt),c.view(at::kInt)),"residual mismatch");
      TORCH_CHECK(at::equal(b.contiguous().view(at::kShort),d.contiguous().view(at::kShort)),"norm mismatch");
    }
    std::vector<double> ta,tb;
    if(rows==5184 && argc==2)for(int pass=0;pass<4;++pass) {
      if(pass%2){tb.push_back(measure(fused));ta.push_back(measure(separate));}
      else {ta.push_back(measure(separate));tb.push_back(measure(fused));}
    }
    if(!first)file<<',';first=false;
    file<<"{\"batch\":"<<shape[0]<<",\"height\":"<<shape[1]<<",\"partition\":"<<(partition?"true":"false")<<",\"bit_equal\":true";
    auto write=[&](const char* name,const auto& values){file<<",\""<<name<<"\":[";for(size_t i=0;i<values.size();++i){if(i)file<<',';file<<values[i];}file<<']';};
    write("separate_ms",ta);write("fused_ms",tb);file<<'}';file.flush();
    std::cout<<rows<<" partition="<<partition<<" passed"<<std::endl;
  }
  file<<"]}\n";TORCH_CHECK(file,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
