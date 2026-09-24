#include "../src/approx_kernels.h"
#include "sam3/rotary_pair.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
double measure(const std::function<void()>& fn) {
  for(int i=0;i<20;++i)fn();
  std::vector<float> times;
  for(int i=0;i<30;++i) {
    at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);
    a.record();fn();b.record();b.synchronize();times.push_back(a.elapsed_time(b));
  }
  std::sort(times.begin(),times.end());return (times[14]+times[15])*.5;
}
int main(int argc,char** argv) {try {
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: qkv_rope_bench OUTPUT.json [--check-only]");
  c10::InferenceMode inference;at::manual_seed(9143);
  c10::cuda::CUDAGuard device(0);
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  std::ofstream file(argv[1]);file<<"{\"cases\":[";bool first=true;
  for(auto [batch,length]:std::vector<std::pair<int,int>>{{1,1},{2,129},{9,576},{1,5184}}) {
    const int rows=batch*length;
    auto accum=at::randint(-100000,100001,{rows,3072},opt.dtype(at::kInt));
    auto xs=at::rand({rows},opt)*.002,ws=at::rand({3072},opt)*.005;
    auto bias=(at::randn({3072},opt)*.02).to(at::kHalf);
    auto phase=at::randn({length,32},opt)*5;
    auto freq=at::complex(at::cos(phase),at::sin(phase));
    at::Tensor restored,q,k,v,packed;
    auto separate=[&]{
      restored=at::empty({rows,3072},opt.dtype(at::kHalf));
      approx_restore(accum,xs,ws,bias,restored,false);
      auto x=restored.view({batch,length,3,16,64}).permute({2,0,3,1,4});
      std::tie(q,k)=sam3::rotary_embedding_pair(x[0],x[1],freq);v=x[2];
    };
    auto fused=[&]{packed=approx_restore_rope(accum,xs,ws,bias,freq,batch);};
    for(int kind=0;kind<3;++kind) {
      if(kind==1){accum.zero_();bias.zero_();}
      if(kind==2){accum.random_(-100000,100001);bias.normal_(0,.1);xs.uniform_(0,.002);ws.uniform_(0,.005);}
      separate();fused();
      for(int part=0;part<3;++part) {
        const auto ref=part==0?q:part==1?k:v;
        TORCH_CHECK(at::equal(ref.view(at::kShort),packed[part].view(at::kShort)),"not bit equal B=",batch," N=",length," part=",part," kind=",kind,
                    " max=",(ref.to(at::kFloat)-packed[part].to(at::kFloat)).abs().max().item<float>());
      }
      // Contiguous V must not silently alter the downstream SDPA result.
      auto a=at::scaled_dot_product_attention(q,k,v),b=at::scaled_dot_product_attention(packed[0],packed[1],packed[2]);
      TORCH_CHECK(at::equal(a.view(at::kShort),b.view(at::kShort)),"V layout changes SDPA output B=",batch," N=",length);
    }
    std::vector<double> a,b;
    if(length>=576 && argc==2)for(int pass=0;pass<4;++pass) {
      if(pass%2){b.push_back(measure(fused));a.push_back(measure(separate));}
      else {a.push_back(measure(separate));b.push_back(measure(fused));}
    }
    if(!first)file<<',';first=false;
    file<<"{\"batch\":"<<batch<<",\"length\":"<<length<<",\"bit_equal\":true";
    auto write=[&](const char* name,const auto& values){file<<",\""<<name<<"\":[";for(size_t i=0;i<values.size();++i){if(i)file<<',';file<<values[i];}file<<']';};
    write("separate_ms",a);write("fused_ms",b);file<<'}';file.flush();
    std::cout<<batch<<" x "<<length<<" passed"<<std::endl;
  }
  file<<"]}\n";TORCH_CHECK(file,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
