#include "sam3/int8_gemm_experiment.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
double measure(const std::function<at::Tensor()>& fn) {
  for(int i=0;i<10;++i)fn();
  std::vector<float> t;
  for(int i=0;i<20;++i){at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);a.record();auto out=fn();b.record();b.synchronize();t.push_back(a.elapsed_time(b));}
  std::sort(t.begin(),t.end());return (t[9]+t[10])*.5;
}
int main(int argc,char** argv){try {
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: int8_gemm_bench OUTPUT [--check-only]");
  c10::InferenceMode inf;c10::cuda::CUDAGuard dev(0);at::manual_seed(72441);
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kChar);
  std::ofstream f(argv[1]);f<<"{\"resources\":[";
  for(int tile=0;tile<6;++tile){if(tile)f<<',';auto r=sam3::int8_gemm_resources(tile);f<<"{\"tile\":"<<tile<<",\"registers\":"<<r[0]<<",\"local_bytes\":"<<r[1]<<",\"shared_bytes\":"<<r[2]<<",\"threads\":"<<r[3]<<",\"blocks_per_sm\":"<<r[4]<<'}';}
  f<<"],\"cases\":[";bool first=true;
  for(auto shape:std::vector<std::vector<int64_t>>{{80,80,128},{5184,1024,4736},{5184,4736,1024},{5184,3072,1024}}) {
    auto m=shape[0],n=shape[1],k=shape[2];auto a=at::randint(-127,128,{m,k},opt),w=at::randint(-127,128,{n,k},opt);
    for(int kind=0;kind<3;++kind){
      if(kind==1){a.fill_(127);w.fill_(127);}
      if(kind==2){a.random_(-128,128);w.random_(-128,128);}
      auto ref=at::_int_mm(a,w.t());
      if(kind==1)TORCH_CHECK((ref==k*127*127).all().item<bool>(),"reference extreme mismatch");
      for(int tile=0;tile<6;++tile)TORCH_CHECK(at::equal(ref,sam3::int8_gemm_experiment(a,w,tile)),"INT32 mismatch tile=",tile," shape=",shape," kind=",kind);
    }
    std::vector<std::vector<double>> times(7);
    if(m==5184 && argc==2){
      for(int i=0;i<100;++i)at::_int_mm(a,w.t());
      for(int pass=0;pass<4;++pass)for(int j=0;j<7;++j){int mode=pass%2?6-j:j;
        times[mode].push_back(measure([&]{return mode==0?at::_int_mm(a,w.t()):sam3::int8_gemm_experiment(a,w,mode-1);}));
      }
    }
    if(!first)f<<',';first=false;
    f<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"bit_equal\":true,\"timings_ms\":[";
    for(int mode=0;mode<7;++mode){if(mode)f<<',';f<<'[';for(size_t j=0;j<times[mode].size();++j){if(j)f<<',';f<<times[mode][j];}f<<']';}f<<"]}";f.flush();
    std::cout<<m<<'x'<<n<<'x'<<k<<" passed"<<std::endl;
  }
  f<<"]}\n";TORCH_CHECK(f,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
