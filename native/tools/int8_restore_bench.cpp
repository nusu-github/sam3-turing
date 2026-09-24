#include "../src/approx_kernels.h"
#include "sam3/int8_gemm_experiment.h"
#include "sam3/vision_fusion.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
double measure(const std::function<void()>& fn){
  for(int i=0;i<10;++i)fn();std::vector<float> t;
  for(int i=0;i<20;++i){at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);a.record();fn();b.record();b.synchronize();t.push_back(a.elapsed_time(b));}
  std::sort(t.begin(),t.end());return (t[9]+t[10])*.5;
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: int8_restore_bench OUTPUT [--check-only]");
  c10::InferenceMode inf;c10::cuda::CUDAGuard dev(0);at::manual_seed(20260924);
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  std::ofstream f(argv[1]);f<<"{\"cases\":[";bool first=true;
  for(auto shape:std::vector<std::vector<int64_t>>{{80,80,128},{5184,1024,4736}}){
    auto m=shape[0],n=shape[1],k=shape[2];auto a=at::randint(-128,128,{m,k},opt.dtype(at::kChar)),w=at::randint(-128,128,{n,k},opt.dtype(at::kChar));
    auto xs=at::rand({m},opt)*.002,ws=at::rand({n},opt)*.005,bias=(at::randn({n},opt)*.02).to(at::kHalf);
    auto residual=at::randn({1,72,72,1024},opt),gamma=at::randn({1024},opt),beta=at::randn({1024},opt);
    at::Tensor ref,out,acc,r1,r2,y1,y2;
    auto baseline=[&]{acc=at::_int_mm(a,w.t());ref=at::empty({m,n},opt.dtype(at::kHalf));approx_restore(acc,xs,ws,bias,ref,false);};
    auto fused=[&]{out=sam3::int8_gemm_restore_experiment(a,w,xs,ws,bias);};
    auto baseline_norm=[&]{acc=at::_int_mm(a,w.t());std::tie(r1,y1)=sam3::approx_fc2_norm(residual,acc,xs,ws,bias,gamma,beta,true);};
    auto fused_norm=[&]{fused();std::tie(r2,y2)=sam3::vision_residual_norm_projection(residual,out.view({1,72,72,1024}),gamma,beta,at::kHalf,true);};
    for(int kind=0;kind<4;++kind){
      if(kind==1){a.fill_(127);w.fill_(127);}
      if(kind==2){a.zero_();w.zero_();}
      if(kind==3){a.random_(-128,128);w.random_(-128,128);}
      baseline();fused();TORCH_CHECK(at::equal(ref.view(at::kShort),out.view(at::kShort)),"restored output differs: ",m," kind=",kind," max=",(ref-out).abs().max().item<float>());
      if(m==5184){baseline_norm();fused_norm();TORCH_CHECK(at::equal(r1.view(at::kInt),r2.view(at::kInt)) && at::equal(y1.contiguous().view(at::kShort),y2.contiguous().view(at::kShort)),"norm output differs");}
    }
    std::vector<std::vector<double>> ts(4);std::vector<std::function<void()>> fns={baseline,fused,baseline_norm,fused_norm};
    if(m==5184 && argc==2){for(int i=0;i<100;++i)baseline_norm();for(int pass=0;pass<4;++pass)for(int j=0;j<4;++j){int mode=pass%2?3-j:j;ts[mode].push_back(measure(fns[mode]));}}
    if(!first)f<<',';first=false;f<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"bit_equal\":true,\"timings_ms\":[";
    for(int i=0;i<4;++i){if(i)f<<',';f<<'[';for(size_t j=0;j<ts[i].size();++j){if(j)f<<',';f<<ts[i][j];}f<<']';}f<<"]}";f.flush();std::cout<<m<<" passed"<<std::endl;
  }
  f<<"]}\n";TORCH_CHECK(f,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
