#include "../src/approx_kernels.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
void validate_kernels() {
  const auto o=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  auto x=at::randn({32,64},o),w=at::randn({32,64},o),bias=at::randn({32},o);
  x[0].zero_();w[0].zero_();x[1].fill_(.5);
  auto q=at::empty({32,64},o.dtype(at::kChar)),qw=at::empty_like(q);
  auto sx=at::empty({32},o.dtype(at::kFloat)),sw=at::empty_like(sx);
  approx_quant(x,q,sx);approx_quant(w,qw,sw);
  for(const auto& pair : std::vector<std::pair<at::Tensor,at::Tensor>>{{x,q},{w,qw}}) {
    auto scale=(at::amax(pair.first.to(at::kFloat).abs(),{1},true)/127).clamp_min(1e-12);
    auto expected=(pair.first.to(at::kFloat)/scale).round().clamp(-127,127).to(at::kChar);
    TORCH_CHECK(at::equal(expected,pair.second),"quantizer reference mismatch");
  }
  auto accum=at::_int_mm(q,qw.t());
  TORCH_CHECK(at::equal(accum.to(at::kFloat),at::mm(q.to(at::kFloat),qw.to(at::kFloat).t())),"integer GEMM reference mismatch");
  auto out=at::empty({32,32},o);
  for(bool gelu : {false,true}) {
    approx_restore(accum,sx,sw,bias,out,gelu);
    auto reference=accum.to(at::kFloat)*sx.unsqueeze(1)*sw.unsqueeze(0)+bias.to(at::kFloat);
    if(gelu)reference=at::gelu(reference,"none");
    TORCH_CHECK(at::allclose(out,reference.to(at::kHalf),1e-3,1e-3),"restoration reference mismatch");
    TORCH_CHECK(at::isfinite(out).all().item<bool>(),"nonfinite zero-row output");
  }
}
double measure(const std::function<void()>& fn) {
  for(int i=0;i<5;++i)fn();
  std::vector<float> ms;
  for(int i=0;i<20;++i){at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);a.record();fn();b.record();b.synchronize();ms.push_back(a.elapsed_time(b));}
  std::sort(ms.begin(),ms.end());return (ms[9]+ms[10])/2;
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==2,"usage: sam3_approx_bench OUTPUT.json");
  c10::InferenceMode inference;c10::cuda::set_device(0);at::manual_seed(4317);
  at::globalContext().setAllowTF32CuBLAS(false);
  validate_kernels();
  std::ofstream f(argv[1]);f<<std::setprecision(9)<<"{\"synthetic\":true,\"shapes\":[";
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  for(int shape=0;shape<2;++shape){
    const int m=5184,k=shape?4736:1024,n=shape?1024:4736;
    auto x=at::randn({m,k},opts),w=at::randn({n,k},opts)*.02,bias=at::randn({n},opts)*.02;
    if(shape)x=at::gelu(x,"none");
    auto xq=at::empty({m,k},opts.dtype(at::kChar)),wq=at::empty({n,k},opts.dtype(at::kChar));
    auto xs=at::empty({m},opts.dtype(at::kFloat)),ws=at::empty({n},opts.dtype(at::kFloat));
    auto out=at::empty({m,n},opts),restored=at::empty({m,n},opts),accum=at::empty({m,n},opts.dtype(at::kInt));
    approx_quant(x,xq,xs);approx_quant(w,wq,ws);
    std::vector<std::string> names={"fp16_exact","fp16_tanh","fp16_addmm_gelu","int8_gemm_only","int8_quant_gemm_restore"};
    std::vector<std::function<void()>> funcs={
      [&]{out=at::linear(x,w,bias);if(!shape)at::gelu_(out,"none");},
      [&]{out=at::linear(x,w,bias);if(!shape)at::gelu_(out,"tanh");},
      [&]{out=shape?at::linear(x,w,bias):at::_addmm_activation(bias,x,w.t(),1,1,true);},
      [&]{at::_int_mm_out(accum,xq,wq.t());},
      [&]{approx_quant(x,xq,xs);at::_int_mm_out(accum,xq,wq.t());approx_restore(accum,xs,ws,bias,restored,!shape);}
    };
    funcs[0]();auto reference=out.clone();
    for(int warm=0;warm<200;++warm)funcs[0]();
    std::vector<std::vector<double>> times(names.size());
    for(int pass=0;pass<4;++pass)for(int j=0;j<int(names.size());++j){int i=pass%2?int(names.size())-1-j:j;times[i].push_back(measure(funcs[i]));}
    if(shape)f<<',';f<<"{\"m\":"<<m<<",\"k\":"<<k<<",\"n\":"<<n<<",\"variants\":[";
    for(int i=0;i<int(names.size());++i){
      funcs[i](); if(i==3)approx_restore(accum,xs,ws,bias,restored,!shape);
      auto actual=i>=3?restored:out;auto diff=actual.to(at::kFloat)-reference.to(at::kFloat);
      TORCH_CHECK(at::isfinite(actual).all().item<bool>(),"nonfinite");
      if(i)f<<',';f<<"{\"name\":\""<<names[i]<<"\",\"max_abs\":"<<diff.abs().max().item<float>()<<",\"rmse\":"<<diff.square().mean().sqrt().item<float>()<<",\"ms\":[";
      for(int j=0;j<4;++j){if(j)f<<',';f<<times[i][j];}f<<"]}";
      std::cout<<shape<<' '<<names[i]<<' '<<times[i][0]<<" ms\n";
    }f<<"]}";
  }f<<"]}\n";TORCH_CHECK(f,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
