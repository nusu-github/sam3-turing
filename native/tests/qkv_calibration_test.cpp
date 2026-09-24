#include "../src/vision_calibration.h"
#include "sam3/vision_fusion.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
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

void output_statistics_checks() {
  auto opts=at::TensorOptions().dtype(at::kFloat);
  auto reference=(at::randn({37,11},opts)*2).to(at::kHalf);
  auto candidate=(reference.to(at::kFloat)+.125+at::randn({37,11},opts)*.01).to(at::kHalf);
  auto bias=(at::randn({11},opts)*.02).to(at::kHalf);
  auto stats=sam3::detail::qkv_error_statistics(reference,candidate,bias);
  // Independent scalar accumulation in FP64, not the statistics helper's reductions.
  auto r=reference.to(at::kDouble),c=candidate.to(at::kDouble);
  auto expected=at::zeros({5,11},opts.dtype(at::kDouble));
  auto* e=expected.mutable_data_ptr<double>();
  for(int col=0;col<11;++col) {
    for(int row=0;row<37;++row) {
      const double rv=r.const_data_ptr<double>()[row*11+col],cv=c.const_data_ptr<double>()[row*11+col];
      e[col]+=(rv-cv)/37;e[11+col]+=(rv-cv)*(rv-cv)/37;
      e[22+col]+=rv/37;e[33+col]+=cv/37;
    }
    e[44+col]=bias.to(at::kDouble).const_data_ptr<double>()[col];
  }
  const auto difference=(stats.to(at::kDouble)-expected).abs().max().item<double>();
  TORCH_CHECK(difference<2e-6,"QKV error statistics differ from scalar oracle");
  const auto delta=stats[0].to(at::kDouble);
  const double before=(r-c).square().mean().item<double>();
  const double after=(r-c-delta).square().mean().item<double>();
  TORCH_CHECK(after<before*.02,"mean output correction did not remove known channel offset");
  rejects([&]{sam3::detail::qkv_error_statistics(reference,candidate.slice(0,0,3),bias);});
  rejects([&]{sam3::detail::qkv_error_statistics(reference,candidate,bias.slice(0,0,3));});
  rejects([&]{sam3::detail::qkv_error_statistics(reference,candidate.clone().fill_(INFINITY),bias);});
  rejects([&]{sam3::detail::read_qkv_output_bias(32,{},at::kCPU);});
  const auto folder=std::filesystem::temp_directory_path()/
      ("sam3-qkv-bias-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(folder);
  const auto file=folder/"qkv-output-bias.f32.bin";
  const auto write=[&](const at::Tensor& value) {
    std::ofstream out(file,std::ios::binary|std::ios::trunc);
    out.write(static_cast<const char*>(value.const_data_ptr()),value.nbytes());
    TORCH_CHECK(out,"test bias fixture write failed");
  };
  auto values=at::randn({32,3072},opts);
  write(values);
  TORCH_CHECK(at::equal(sam3::detail::read_qkv_output_bias(3,folder,at::kCPU),values[3].to(at::kHalf)),
      "QKV output bias reader loaded wrong layer or dtype");
  write(values[0]);
  rejects([&]{sam3::detail::read_qkv_output_bias(0,folder,at::kCPU);});
  write(values.fill_(std::numeric_limits<float>::quiet_NaN()));
  rejects([&]{sam3::detail::read_qkv_output_bias(0,folder,at::kCPU);});
  write(values.fill_(70000));
  rejects([&]{sam3::detail::read_qkv_output_bias(0,folder,at::kCPU);});
  std::filesystem::remove(file);std::filesystem::remove(folder);
  std::cout<<"output_stats_scalar_max="<<difference<<" corrected_mse_ratio="<<after/before<<std::endl;
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
    auto fp16_reference=at::linear(base.reshape({5184,1024}),weight,bias);
    // A known bias perturbation makes this a sensitive compensation regression.
    auto perturbed=(corrected.to(at::kFloat)+.125).to(at::kHalf);
    approx_restore(acc,sa,sw,perturbed,output,false);
    auto stats=sam3::detail::qkv_error_statistics(fp16_reference,output,perturbed);
    auto cpu_error=fp16_reference.cpu().to(at::kDouble)-output.cpu().to(at::kDouble);
    const double mean_stats_error=(stats[0].cpu().to(at::kDouble)-cpu_error.mean(0)).abs().max().item<double>();
    TORCH_CHECK(mean_stats_error<2e-6,"CUDA mean output error disagrees with FP64 oracle");
    auto empirical=(perturbed.to(at::kFloat)+stats[0]).to(at::kHalf).contiguous();
    approx_restore(acc,sa,sw,empirical,output,false);
    const auto post=sam3::detail::qkv_error_statistics(fp16_reference,output,empirical);
    const double before_mean=stats[0].abs().max().item<double>(),after_mean=post[0].abs().max().item<double>();
    TORCH_CHECK(after_mean<.001 && after_mean<before_mean*.02,"empirical QKV correction failed after actual Half restore");
    std::cout<<"windowed="<<windowed<<" norm_max="<<maxerr<<" norm_rmse="<<rmse<<" restore_max="<<restore_error
      <<" empirical_mean_before="<<before_mean<<" empirical_mean_after="<<after_mean<<" stats_error="<<mean_stats_error<<std::endl;
  }
  c10::cuda::getCurrentCUDAStream().synchronize();
}
#endif

int main(int argc,char** argv) {try {
  TORCH_CHECK(argc==2,"usage: qkv_calibration_test cpu|cuda");
  c10::InferenceMode guard;at::set_num_threads(4);at::manual_seed(7431);
  cpu_checks();
  output_statistics_checks();
  const std::string mode=argv[1];
#ifdef SAM3_WITH_CUDA
  if(mode=="cuda")cuda_checks();
  else
#endif
  TORCH_CHECK(mode=="cpu","unsupported test mode");
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
