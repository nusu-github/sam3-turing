#include "../src/vision_calibration.h"
#include "sam3/vision_fusion.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <iostream>
#ifdef SAM3_WITH_CUDA
#include "../src/approx_kernels.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

template<class F> void rejects(F&& f) {
  bool rejected=false;
  try { f(); } catch(const c10::Error&) { rejected=true; }
  TORCH_CHECK(rejected,"invalid calibration was accepted");
}

void cpu_checks() {
  auto opts=at::TensorOptions().dtype(at::kDouble);
  auto x=at::randn({17,1024},opts),g=at::rand({1024},opts)+.5,b=at::randn({1024},opts)*.1;
  auto w=at::randn({37,1024},opts)*.03,bias=at::randn({37},opts)*.02;
  auto r=at::rand({1024},opts)*3.75+.25,shift=at::randn({1024},opts)*.25;
  auto [fg,fb]=sam3::detail::fold_qkv_norm(g,b,r,shift);
  auto original=at::layer_norm(x,{1024},g,b,1e-5);
  auto transformed=at::layer_norm(x,{1024},fg,fb,1e-5);
  auto reference=at::linear(original,w,bias);
  auto candidate=at::linear(transformed,w*r,bias+at::mv(w*r,shift));
  const double equivalence=(candidate-reference).abs().max().item<double>();
  TORCH_CHECK(equivalence<1e-11,"FP64 folded QKV transform is not equivalent");
  auto [ig,ib]=sam3::detail::fold_qkv_norm(g,b,at::ones_like(r),at::zeros_like(shift));
  TORCH_CHECK(at::equal(g,ig) && at::equal(b,ib),"identity norm folding changed parameters");
  rejects([&]{sam3::detail::fold_qkv_norm(g,b,at::zeros_like(r),shift);});
  rejects([&]{sam3::detail::fold_qkv_norm(g,b,r,-shift/0.);});
  rejects([&]{sam3::detail::fold_qkv_norm(g,b,r.slice(0,0,7),shift);});
  rejects([&]{sam3::detail::read_qkv_data(32,{},true,at::kCPU);});
  // Independent empirical mean oracle, without quantizing the activations.
  auto inputs=original.to(at::kFloat),mean=inputs.mean(0);
  auto wh=w.to(at::kHalf),bh=bias.to(at::kHalf),rf=r.to(at::kFloat),sf=shift.to(at::kFloat);
  auto qw=at::randint(-127,128,w.sizes(),opts.dtype(at::kChar));
  auto scales=at::rand({37},opts.dtype(at::kFloat))*.003+.001;
  auto corrected=sam3::detail::linear_mean_bias(wh,bh,qw,scales,mean,rf,sf);
  auto xd=inputs.to(at::kDouble),wd=qw.to(at::kDouble)*scales.to(at::kDouble).unsqueeze(1);
  auto target=at::linear(xd,wh.to(at::kDouble),bh.to(at::kDouble));
  auto actual=at::linear(xd/rf.to(at::kDouble)-sf.to(at::kDouble),wd,corrected.to(at::kDouble));
  const double mean_error=(target-actual).mean(0).abs().max().item<double>();
  TORCH_CHECK(mean_error<2e-5,"QKV corrected weight mean disagrees with FP64 oracle");
  std::cout<<"fp64_equivalence_max="<<equivalence<<" compensated_mean_max="<<mean_error<<std::endl;
}

#ifdef SAM3_WITH_CUDA
void cuda_checks() {
  c10::cuda::set_device(0);
  c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromPool(false,0));
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  auto input=at::randn({1,72,72,1024},opts);
  auto g=at::rand({1024},opts)+.5,b=at::randn({1024},opts)*.1;
  auto r=at::rand({1024},opts)*1.5+.5,shift=at::randn({1024},opts)*.25;
  for(bool windowed:{false,true}) {
    auto base=sam3::vision_norm_projection(input,g,b,at::kHalf,windowed);
    auto [ig,ib]=sam3::detail::fold_qkv_norm(g,b,at::ones_like(r),at::zeros_like(shift));
    auto identity=sam3::vision_norm_projection(input,ig,ib,at::kHalf,windowed);
    TORCH_CHECK(at::equal(base,identity),"identity changes CUDA normalized projection");
    auto [fg,fb]=sam3::detail::fold_qkv_norm(g,b,r,shift);
    auto actual=sam3::vision_norm_projection(input,fg,fb,at::kHalf,windowed);
    auto cpu=input.cpu().to(at::kDouble),gd=g.cpu().to(at::kDouble),bd=b.cpu().to(at::kDouble);
    // Explicit scalar normalization algebra in float64; independent from the
    // folded gamma/beta helper and the CUDA fused normalization implementation.
    auto centered=cpu-cpu.mean(-1,true);
    auto expected=((centered/(centered.square().mean(-1,true)+1e-5).sqrt())*gd+bd)/r.cpu().to(at::kDouble)-shift.cpu().to(at::kDouble);
    if(windowed)expected=expected.view({1,3,24,3,24,1024}).permute({0,1,3,2,4,5}).reshape({9,24,24,1024});
    auto diff=actual.cpu().to(at::kDouble)-expected;
    const double maxerr=diff.abs().max().item<double>(),rmse=diff.square().mean().sqrt().item<double>();
    TORCH_CHECK(maxerr<.01 && rmse<.001,"folded norm disagrees with FP64 scalar reference");
    auto projection=(at::randn_like(input)*.02).to(at::kHalf);
    auto [sum,prepared]=sam3::vision_residual_norm_projection(input,projection,fg,fb,at::kHalf,windowed);
    auto direct=sam3::vision_norm_projection(sum,fg,fb,at::kHalf,windowed);
    TORCH_CHECK(at::equal(prepared,direct),"folded norm differs in prepared projection path");
    // Exercise real QKV shape and corrected bias through integer GEMM/restore.
    auto flat=actual.reshape({5184,1024}).contiguous();
    auto qa=at::empty(flat.sizes(),opts.dtype(at::kChar)),sa=at::empty({5184},opts);
    auto weight=(at::randn({3072,1024},opts)*.02).to(at::kHalf),bias=(at::randn({3072},opts)*.01).to(at::kHalf);
    auto tw=(weight.to(at::kFloat)*r).to(at::kHalf).contiguous();
    auto qw=at::empty(tw.sizes(),opts.dtype(at::kChar)),sw=at::empty({3072},opts);
    approx_quant(flat,qa,sa);approx_quant(tw,qw,sw);
    auto corrected=sam3::detail::linear_mean_bias(weight,bias,qw,sw,base.reshape({5184,1024}).to(at::kFloat).mean(0),r,shift).to(at::kHalf);
    auto acc=at::_int_mm(qa,qw.t());
    auto output=at::empty({5184,3072},opts.dtype(at::kHalf));
    approx_restore(acc,sa,sw,corrected,output,false);
    auto integer_reference=at::matmul(qa.slice(0,0,8).cpu().to(at::kLong),qw.cpu().to(at::kLong).t());
    TORCH_CHECK(at::equal(integer_reference,acc.slice(0,0,8).cpu().to(at::kLong)),"QKV integer GEMM differs");
    auto restored=integer_reference.to(at::kDouble)*sa.slice(0,0,8).cpu().to(at::kDouble).unsqueeze(1)*sw.cpu().to(at::kDouble)+corrected.cpu().to(at::kDouble);
    auto od=output.slice(0,0,8).cpu().to(at::kDouble)-restored;
    const double restore_error=od.abs().max().item<double>();
    TORCH_CHECK(restore_error<.004,"QKV restore differs from independent integer-dot/bias oracle");
    std::cout<<"windowed="<<windowed<<" norm_max="<<maxerr<<" norm_rmse="<<rmse<<" restore_max="<<restore_error<<std::endl;
  }
  c10::cuda::getCurrentCUDAStream().synchronize();
}
#endif

int main(int argc,char** argv) {try {
  TORCH_CHECK(argc==2,"usage: qkv_calibration_test cpu|cuda");
  c10::InferenceMode guard;at::set_num_threads(4);at::manual_seed(7431);
  cpu_checks();
  const std::string mode=argv[1];
#ifdef SAM3_WITH_CUDA
  if(mode=="cuda")cuda_checks();
  else
#endif
  TORCH_CHECK(mode=="cpu","unsupported test mode");
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
