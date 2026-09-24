#include "sam3/kitchen_attention.h"
#include "../src/vision_experiments.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

void check_centering(const char* output) {
  const auto& center=sam3::detail::kitchen_center();
  auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);
  std::ofstream file(output);TORCH_CHECK(file,"cannot open centering report");
  file<<"{\"center\":\""<<center<<"\",\"repeats\":5,\"tolerance_max_abs\":0.05,\"tolerance_rmse\":0.005,\"cases\":[";
  bool first=true;
  for(const auto& shape:std::vector<std::pair<int,int>>{{1,1},{2,65},{1,129},{9,576},{1,5184}}) {
    const int batch=shape.first,length=shape.second,heads=16;
    auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
    for(const std::string kind:{"random","shifted","constant","zero"}) {
      auto packed=at::randn({batch,length,3,heads,64},options);
      auto q=packed.select(2,0).contiguous().transpose(1,2);
      auto k=packed.select(2,1).contiguous().transpose(1,2);
      auto v=packed.select(2,2).transpose(1,2); // production packed V strides
      if(kind=="shifted")k=(k+at::randn({batch,heads,1,64},options)*16).contiguous();
      if(kind=="constant")k=at::randn({batch,heads,1,64},options).expand({batch,heads,length,64}).contiguous();
      if(kind=="zero"){q.zero_();k.zero_();v.zero_();}
      at::Tensor reference;
      double exact_shift_error=0;
      if(length<=129) {
        auto qc=q.to(at::kCPU,at::kDouble),kc=k.to(at::kCPU,at::kDouble),vc=v.to(at::kCPU,at::kDouble);
        auto exact=at::matmul(at::softmax(at::matmul(qc,kc.transpose(-1,-2))*.125,-1),vc);
        auto centered=kc-kc.mean(-2,true);
        auto shifted=at::matmul(at::softmax(at::matmul(qc,centered.transpose(-1,-2))*.125,-1),vc);
        exact_shift_error=(exact-shifted).abs().max().item<double>();
        TORCH_CHECK(exact_shift_error<1e-10,"K mean shift changed exact attention");
        reference=exact.to(options);
      } else if(kind=="constant" || kind=="zero") {
        reference=v.to(at::kFloat).mean(-2,true).expand(v.sizes()).to(at::kHalf);
      } else reference=at::scaled_dot_product_attention(q,k,v);
      for(bool rotation:{false,true}) {
        auto result=sam3::kitchen_attention(q,k,v,rotation,true);
        auto diff=result.to(at::kFloat)-reference.to(at::kFloat);
        const float maxabs=diff.abs().max().item<float>(),rmse=diff.square().mean().sqrt().item<float>();
        TORCH_CHECK(at::isfinite(result).all().item<bool>(),"nonfinite centered attention");
        int repeat_changes=0;float repeat_max=0;
        for(int repeat=0;repeat<5;++repeat) {
          auto again=sam3::kitchen_attention(q,k,v,rotation,true);
          const float delta=(again.to(at::kFloat)-result.to(at::kFloat)).abs().max().item<float>();
          repeat_changes+=!at::equal(again,result);repeat_max=std::max(repeat_max,delta);
        }
        const bool pass=maxabs<=.05 && rmse<=.005;
        if(!first)file<<',';first=false;
        file<<"{\"batch\":"<<batch<<",\"length\":"<<length<<",\"kind\":\""<<kind<<"\",\"rotation\":"<<(rotation?"true":"false")
            <<",\"max_abs\":"<<maxabs<<",\"rmse\":"<<rmse<<",\"within_tolerance\":"<<(pass?"true":"false")
            <<",\"exact_shift_error\":"<<exact_shift_error<<",\"changed_repeats\":"<<repeat_changes<<",\"repeat_max_abs\":"<<repeat_max<<'}';
        file.flush();
        TORCH_CHECK(center=="none" || pass,"centered attention exceeds preset tolerance: ",kind," N=",length);
      }
      std::cout<<center<<' '<<kind<<' '<<batch<<" x "<<length<<" checked"<<std::endl;
    }
  }
  stream.synchronize();file<<"]}\n";TORCH_CHECK(file,"centering report write failed");
}
double measure(const std::function<at::Tensor()>& fn) {
  for(int i=0;i<100;++i)fn();
  std::vector<float> times;
  for(int i=0;i<20;++i) {
    at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);
    a.record();auto out=fn();b.record();b.synchronize();times.push_back(a.elapsed_time(b));
  }
  std::sort(times.begin(),times.end());return .5*(times[9]+times[10]);
}
int main(int argc,char** argv) {try {
  TORCH_CHECK(argc==2 || (argc==3 && (std::string(argv[2])=="--check-only" || std::string(argv[2])=="--center-check")),"usage: sam3_kitchen_bench OUTPUT.json [--check-only|--center-check]");
  c10::InferenceMode inference;c10::cuda::set_device(0);at::manual_seed(7429);
  if(argc==3 && std::string(argv[2])=="--center-check"){check_centering(argv[1]);return 0;}
  std::ofstream file(argv[1]);file<<"{\"tolerance_max_abs\":0.05,\"tolerance_rmse\":0.005,\"cases\":[";bool first=true;
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
    auto small=[&] {return sam3::kitchen_attention(q,k,v,false);};
    auto large=[&] {return sam3::kitchen_attention(q,k,v,true);};
    auto ref=baseline();auto a=small(),b=large();
    auto sequence=sam3::kitchen_attention(q,k,v,false,true);
    auto sequence_rot=sam3::kitchen_attention(q,k,v,true,true);
    TORCH_CHECK(at::equal(a,sequence) && at::equal(b,sequence_rot),"output layout changed attention values");
    auto concatenated=sequence.transpose(1,2).reshape({batch,length,heads*64});
    TORCH_CHECK(concatenated.const_data_ptr()==sequence.const_data_ptr() && concatenated.is_contiguous(),"head concatenation still copies");
    stream.synchronize();

    if(length<=129)
      ref=at::matmul(at::softmax(at::matmul(q.to(at::kFloat),k.to(at::kFloat).transpose(-1,-2))*.125,-1),v.to(at::kFloat)).to(at::kHalf);
    auto diff=a.to(at::kFloat)-ref.to(at::kFloat),diffb=b.to(at::kFloat)-ref.to(at::kFloat);
    const float maxabs=diff.abs().max().item<float>(),rmse=diff.square().mean().sqrt().item<float>();
    const float maxrot=diffb.abs().max().item<float>(),rmserot=diffb.square().mean().sqrt().item<float>();
    TORCH_CHECK(at::isfinite(a).all().item<bool>() && at::isfinite(b).all().item<bool>() && maxabs<=.05 && rmse<=.005 && maxrot<=.05 && rmserot<=.005,
                "INT8 attention error exceeds preset tolerance at N=",length," max=",maxabs," rmse=",rmse," rotated max=",maxrot," rmse=",rmserot);
    std::vector<double> ts,ta,tb;
    if(length>=576 && argc==2) {
      for(int pass=0;pass<4;++pass) {
        if(pass%2) {tb.push_back(measure(large));ta.push_back(measure(small));ts.push_back(measure(baseline));}
        else {ts.push_back(measure(baseline));ta.push_back(measure(small));tb.push_back(measure(large));}
      }
    }
    if(!first)file<<',';first=false;
    file<<"{\"batch\":"<<batch<<",\"length\":"<<length<<",\"max_abs\":"<<maxabs<<",\"rmse\":"<<rmse<<",\"rotation_max_abs\":"<<maxrot<<",\"rotation_rmse\":"<<rmserot;
    auto write=[&](const char* name,const auto& values) {file<<",\""<<name<<"\":[";for(size_t i=0;i<values.size();++i){if(i)file<<',';file<<values[i];}file<<']';};
    write("sdpa_ms",ts);write("kitchen_ms",ta);write("kitchen_rot_ms",tb);file<<'}';file.flush();
    std::cout<<batch<<" x "<<length<<" max="<<maxabs<<" rmse="<<rmse<<std::endl;
  }
  file<<"]}\n";TORCH_CHECK(file,"write failed");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
