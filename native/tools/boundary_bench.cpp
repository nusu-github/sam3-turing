#include "../src/approx_kernels.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
// Independent scalar reference: do not let two implementations share a reduction bug.
void check_quant(const at::Tensor& input,const at::Tensor& q,const at::Tensor& scales) {
  auto x=input.to(at::kCPU).to(at::kFloat), actual=q.cpu(), s=scales.cpu();
  const auto* values=x.const_data_ptr<float>();
  const auto* bytes=actual.const_data_ptr<int8_t>();
  for(int64_t row=0;row<x.size(0);++row) {
    float maximum=0;
    for(int64_t col=0;col<x.size(1);++col)
      maximum=std::fmax(maximum,std::fabs(values[row*x.size(1)+col]));
    const float scale=std::fmax(maximum/127.f,1e-12f);
    TORCH_CHECK(scale==s.const_data_ptr<float>()[row],"scalar scale mismatch");
    for(int64_t col=0;col<x.size(1);++col) {
      const auto i=row*x.size(1)+col;
      const auto expected=static_cast<int8_t>(std::nearbyint(std::fmin(127.f,std::fmax(-127.f,values[i]/scale))));
      TORCH_CHECK(bytes[i]==expected,"scalar quantization mismatch");
    }
  }
}
double timing(const std::function<void()>& fn) {
  std::vector<float> times;
  for(int i=0;i<20;++i) {
    at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);
    a.record();fn();b.record();b.synchronize();times.push_back(a.elapsed_time(b));
  }
  std::sort(times.begin(),times.end());return .5*(times[9]+times[10]);
}
int main(int argc,char** argv) { try {
  TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),
              "usage: sam3_boundary_bench OUTPUT.json [--check-only]");
  c10::InferenceMode guard;c10::cuda::set_device(0);at::manual_seed(7021);
  c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromPool());
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  std::ofstream out(argv[1]);out<<"{\"cases\":[";
  bool first=true;
  for(int n : {1,31,32,257,1024,4736,8192}) {
    const int m=n==4736?5184:37;
    auto acc=at::randint(-100000,100001,{m,n},opts.dtype(at::kInt));
    auto xs=at::rand({m},opts)*.002,ws=at::rand({n},opts)*.005;
    auto bias=(at::randn({n},opts)*.02).to(at::kHalf);
    auto half=at::empty({m,n},opts.dtype(at::kHalf));
    auto q=at::empty({m,n},opts.dtype(at::kChar)),qf=at::empty_like(q);
    auto s=at::empty({m},opts),sf=at::empty_like(s);
    const auto separate=[&]{approx_restore(acc,xs,ws,bias,half,true);approx_quant(half,q,s);};
    const auto fused=[&]{approx_restore_quant(acc,xs,ws,bias,qf,sf);};
    for(int kind=0;kind<3;++kind) {
      if(kind==1) { acc.zero_();bias.zero_(); }
      if(kind==2) { acc.fill_(1000);xs.fill_(.01);ws.fill_(.01);bias.fill_(.25); }
      separate();fused();
      TORCH_CHECK(at::equal(q,qf) && at::equal(s,sf),"fused boundary mismatch n=",n," case=",kind);
      check_quant(half,q,s);
    }
    // Benchmark a random distribution after checking zero and constant inputs.
    acc.random_(-100000,100001);xs.uniform_(0,.002);ws.uniform_(0,.005);bias.normal_(0,.02);
    separate();fused();
    TORCH_CHECK(at::equal(q,qf) && at::equal(s,sf),"benchmark input mismatch");
    // Check channel scaling/translation against an independent scalar FP16 reference.
    auto r=at::ones({n},opts),shift=at::zeros({n},opts);
    approx_restore_quant_affine(acc,xs,ws,bias,r,shift,qf,sf);
    TORCH_CHECK(at::equal(q,qf) && at::equal(s,sf),"identity calibration changes boundary");
    r.uniform_(.125,8);shift.uniform_(-.5,.5);
    auto scalar=half.cpu().contiguous(),rc=r.cpu(),bc=shift.cpu();
    auto* values=scalar.mutable_data_ptr<at::Half>();
    for(int64_t i=0;i<scalar.numel();++i) {
      const float scaled=float(values[i])/rc.const_data_ptr<float>()[i%n];
      values[i]=at::Half(scaled-bc.const_data_ptr<float>()[i%n]);
    }
    auto transformed=scalar.to(at::kCUDA);
    approx_quant(transformed,q,s);
    approx_restore_quant_affine(acc,xs,ws,bias,r,shift,qf,sf);
    TORCH_CHECK(at::equal(q,qf) && at::equal(s,sf),"affine calibration scalar mismatch n=",n);
    check_quant(scalar,qf,sf);
    std::vector<double> a,b;
    if(n==4736 && argc==2) {
      for(int i=0;i<100;++i) {separate();fused();}
      for(int pass=0;pass<4;++pass) {
        if(pass%2) { b.push_back(timing(fused));a.push_back(timing(separate)); }
        else { a.push_back(timing(separate));b.push_back(timing(fused)); }
      }
    }
    if(!first)out<<',';first=false;
    out<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"bit_equal\":true,\"affine_scalar_equal\":true,\"separate_ms\":[";
    for(size_t i=0;i<a.size();++i) {if(i)out<<',';out<<a[i];}
    out<<"],\"fused_ms\":[";
    for(size_t i=0;i<b.size();++i) {if(i)out<<',';out<<b[i];}
    out<<"]}";
    std::cout<<"width "<<n<<" passed"<<std::endl;
  }
  out<<"]}\n";TORCH_CHECK(out,"output write failed");return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;} }
