#include "sam3/pixel_transform.h"
#include "sam3/autocast.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <fstream>
#include <iostream>
#include <functional>
#include <algorithm>
double measure(const std::function<at::Tensor()>& fn) {
  for(int i=0;i<20;++i)fn();
  std::vector<float> times;
  for(int i=0;i<20;++i){at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);a.record();auto y=fn();b.record();b.synchronize();times.push_back(a.elapsed_time(b));}
  std::sort(times.begin(),times.end());return .5*(times[9]+times[10]);
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: sam3_pixel_bench OUTPUT.json [--check-only]");
  c10::InferenceMode inference;at::manual_seed(921);
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  sam3::AutocastGuard autocast(at::kCUDA,true,at::kHalf);
  std::ofstream file(argv[1]);file<<"{\"cases\":[";bool first=true;
  for(const auto& dims:std::vector<std::vector<int64_t>>{{2,40,33,35},{1,256,144,144},{1,256,288,288}}){
    auto x=at::randn(dims,at::TensorOptions().device(at::kCUDA).dtype(at::kHalf)).contiguous(at::MemoryFormat::ChannelsLast);
    auto gamma=at::randn({dims[1]},x.options().dtype(at::kFloat)),beta=at::randn({dims[1]},gamma.options());
    auto converted=sam3::pixel_nchw_float(x);
    TORCH_CHECK(converted.is_contiguous() && at::equal(converted,x.to(at::kFloat).contiguous()),"pixel transpose mismatch");
    auto norm=[&](const at::Tensor& input){return at::group_norm(input,8,gamma,beta,1e-5);};
    auto base=[&]{return norm(x);};
    auto plain=[&]{return norm(x.contiguous().to(at::kFloat));};
    auto fused=[&]{return norm(sam3::pixel_nchw_float(x));};
    auto reference=base();
    TORCH_CHECK(reference.scalar_type()==at::kFloat && at::equal(reference,plain()) && at::equal(reference,fused()),"group norm mismatch after transpose");
    if(!first)file<<',';first=false;
    file<<"{\"channels\":"<<dims[1]<<",\"height\":"<<dims[2]<<",\"width\":"<<dims[3]<<",\"exact\":true,\"timing\":[";
    if(argc==2 && dims[1]==256)for(int p=0;p<4;++p){double b,n,f;if(p%2){f=measure(fused);n=measure(plain);b=measure(base);}else{b=measure(base);n=measure(plain);f=measure(fused);}if(p)file<<',';file<<"["<<b<<','<<n<<','<<f<<']';}
    file<<"]}";file.flush();std::cout<<dims[2]<<" x "<<dims[3]<<" exact"<<std::endl;
  }
  file<<"]}\n";TORCH_CHECK(file,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
