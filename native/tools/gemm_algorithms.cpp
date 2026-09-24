// Diagnostic only: search cuBLASLt heuristics for SAM3's two MLP shapes.
// Synthetic contiguous FP16 operands; excludes casts, GELU, model loading.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cublasLt.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
void check(cublasStatus_t status) { TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(status)); }
template<class F> double measure(F&& fn) {
  for (int i=0; i<3; ++i) fn();
  c10::cuda::device_synchronize();
  std::vector<double> times;
  for(int i=0; i<10; ++i) {
    at::cuda::CUDAEvent start(cudaEventDefault), end(cudaEventDefault);
    start.record(); fn(); end.record(); end.synchronize();
    times.push_back(start.elapsed_time(end));
  }
  std::sort(times.begin(),times.end());
  return (times[4]+times[5])/2;
}
struct Lt {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t a{},b{},c{};
  cublasLtMatmulPreference_t pref{};
  ~Lt() {
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    if(c)cublasLtMatrixLayoutDestroy(c);
    if(b)cublasLtMatrixLayoutDestroy(b);
    if(a)cublasLtMatrixLayoutDestroy(a);
    if(desc)cublasLtMatmulDescDestroy(desc);
    if(handle)cublasLtDestroy(handle);
  }
};
}
int main(int argc,char** argv) { try {
  TORCH_CHECK(argc==2,"usage: sam3_gemm_algorithms OUTPUT.json");
  c10::InferenceMode inference;
  c10::cuda::set_device(0);
  at::manual_seed(4302);
  at::globalContext().setAllowTF32CuBLAS(false);
  const auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  const size_t workspace_bytes=32ULL<<20;
  auto workspace=at::empty({int64_t(workspace_bytes)},options.dtype(at::kByte));
  std::ofstream file(argv[1]);
  file<<std::setprecision(10)<<"{\"workspace_bytes\":"<<workspace_bytes<<",\"shapes\":[";
  for(int shape=0;shape<2;++shape) {
    const int64_t m=5184,k=shape?4736:1024,n=shape?1024:4736;
    auto x=at::randn({m,k},options), w=at::randn({n,k},options)*.02, bias=at::randn({n},options)*.02;
    auto reference=at::linear(x,w,bias);
    auto out=at::empty({m,n},options);
    Lt lt; check(cublasLtCreate(&lt.handle));
    check(cublasLtMatmulDescCreate(&lt.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    const auto trans=CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(lt.desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    const auto epilogue=CUBLASLT_EPILOGUE_BIAS;
    check(cublasLtMatmulDescSetAttribute(lt.desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
    const auto* bias_ptr=bias.const_data_ptr();
    check(cublasLtMatmulDescSetAttribute(lt.desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias_ptr,sizeof(bias_ptr)));
    // Column-major D[n,m] = transpose(W[k,n]) * X[k,m] + bias[n].
    check(cublasLtMatrixLayoutCreate(&lt.a,CUDA_R_16F,k,n,k));
    check(cublasLtMatrixLayoutCreate(&lt.b,CUDA_R_16F,k,m,k));
    check(cublasLtMatrixLayoutCreate(&lt.c,CUDA_R_16F,n,m,n));
    check(cublasLtMatmulPreferenceCreate(&lt.pref));
    check(cublasLtMatmulPreferenceSetAttribute(lt.pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
    std::vector<cublasLtMatmulHeuristicResult_t> candidates(32);
    int count=0;
    check(cublasLtMatmulAlgoGetHeuristic(lt.handle,lt.desc,lt.a,lt.b,lt.c,lt.c,lt.pref,int(candidates.size()),candidates.data(),&count));
    candidates.resize(count);
    const float alpha=1,beta=0;
    auto run=[&](int index) { check(cublasLtMatmul(lt.handle,lt.desc,&alpha,w.const_data_ptr(),lt.a,x.const_data_ptr(),lt.b,&beta,out.const_data_ptr(),lt.c,out.mutable_data_ptr(),lt.c,&candidates[index].algo,workspace.mutable_data_ptr(),workspace_bytes,c10::cuda::getCurrentCUDAStream())); };
    auto aten=[&] { out=at::linear(x,w,bias); };
    // Keep the GPU busy before the first measurement so process startup/idle
    // clock ramp is not mistaken for an algorithm improvement.
    for(int warm=0;warm<200;++warm) aten();
    c10::cuda::device_synchronize();
    const auto aten_before=measure(aten);
    std::vector<double> first(count),second(count),errors(count);
    std::vector<int64_t> different(count);
    std::vector<int> ids(count);
    for(int i=0;i<count;++i) {
      check(candidates[i].state);
      first[i]=measure([&]{run(i);});
      auto diff=(out.to(at::kFloat)-reference.to(at::kFloat)).abs();
      TORCH_CHECK(at::isfinite(out).all().item<bool>(),"nonfinite candidate");
      errors[i]=diff.max().item<double>();
      different[i]=(out!=reference).sum().item<int64_t>();
      size_t written=0;
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_ID,&ids[i],sizeof(ids[i]),&written));
    }
    for(int i=count-1;i>=0;--i) second[i]=measure([&]{run(i);});
    const auto aten_after=measure(aten);
    // Compare adjacent ATen/candidate measurements in alternating order,
    // without validation kernels between them, to expose clock/order effects.
    std::vector<std::vector<double>> paired_aten(count),paired_candidate(count);
    for(int pass=0;pass<4;++pass) for(int index=0;index<count;++index) {
      const int i=pass%2 ? count-1-index : index;
      double a,b;
      if(pass%2) { b=measure([&]{run(i);});a=measure(aten); }
      else { a=measure(aten);b=measure([&]{run(i);}); }
      paired_aten[i].push_back(a);paired_candidate[i].push_back(b);
    }
    if(shape) file<<',';
    file<<"{\"m\":"<<m<<",\"k\":"<<k<<",\"n\":"<<n<<",\"aten_before_ms\":"<<aten_before<<",\"aten_after_ms\":"<<aten_after<<",\"candidates\":[";
    for(int i=0;i<count;++i) {
      if(i)file<<',';
      uint32_t tile=0,split=0,stages=0;size_t written=0;
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written));
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written));
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stages,sizeof(stages),&written));
      file<<"{\"rank\":"<<i<<",\"algo_id\":"<<ids[i]<<",\"tile_id\":"<<tile<<",\"split_k\":"<<split<<",\"stages_id\":"<<stages<<",\"workspace_bytes\":"<<candidates[i].workspaceSize<<",\"first_ms\":"<<first[i]<<",\"reverse_ms\":"<<second[i]<<",\"max_abs_error\":"<<errors[i]<<",\"different_elements\":"<<different[i];
      const auto values=[&](const char* name,const auto& v) {file<<",\""<<name<<"\":[";for(size_t j=0;j<v.size();++j){if(j)file<<',';file<<v[j];}file<<']';};
      values("paired_aten_ms",paired_aten[i]);values("paired_candidate_ms",paired_candidate[i]);file<<'}';
    }
    file<<"]}";
    std::cout<<"shape "<<m<<'x'<<k<<'x'<<n<<" candidates="<<count<<" ATen ms="<<aten_before<<","<<aten_after<<std::endl;
  }
  file<<"]}\n";TORCH_CHECK(file,"cannot write report");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
