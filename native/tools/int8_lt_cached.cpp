#include "../src/approx_kernels.h"
// Opt-in SM75 experiment: cache the measured INT8 cuBLASLt configuration.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <map>
#include <memory>
#include <tuple>
namespace sam3 {
namespace {
void check(cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"INT8 LT status ",int(s));}
struct Config {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{},b{},c{};
  cublasLtMatmulPreference_t pref{};
  cublasLtMatmulAlgo_t algo{};
  ~Config(){if(pref)cublasLtMatmulPreferenceDestroy(pref);if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);if(a)cublasLtMatrixLayoutDestroy(a);if(op)cublasLtMatmulDescDestroy(op);}
  void initialize(cublasLtHandle_t handle,int64_t m,int64_t n,int64_t k) {
    check(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I));
    const auto trans=CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    check(cublasLtMatrixLayoutCreate(&a,CUDA_R_8I,k,n,k));
    check(cublasLtMatrixLayoutCreate(&b,CUDA_R_8I,k,m,k));
    check(cublasLtMatrixLayoutCreate(&c,CUDA_R_32I,n,m,n));
    check(cublasLtMatmulPreferenceCreate(&pref));
    const size_t workspace=0;
    check(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
    cublasLtMatmulHeuristicResult_t candidates[32];int count=0;
    check(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,c,pref,32,candidates,&count));
    for(int i=0;i<count;++i) {
      if(candidates[i].state!=CUBLAS_STATUS_SUCCESS || candidates[i].workspaceSize)continue;
      int id=-1;uint32_t tile=0,split=0;size_t size=0;
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&size));
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&size));
      check(cublasLtMatmulAlgoConfigGetAttribute(&candidates[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&size));
      if(id==21 && tile==20 && split==1){algo=candidates[i].algo;return;}
    }
    TORCH_CHECK(false,"measured INT8 LT algorithm 21/tile20/split1 unavailable");
  }
};
}
at::Tensor int8_lt_cached(const at::Tensor& a,const at::Tensor& w) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kChar && w.scalar_type()==at::kChar &&
      a.device()==w.device() && a.dim()==2 && w.dim()==2 && a.is_contiguous() && w.is_contiguous() &&
      a.size(1)==w.size(1) && a.size(0)>16 && w.size(0)>0 && w.size(0)%8==0 && a.size(1)>0 &&
      a.size(1)%8==0 && a.size(1)<=8192,"invalid cached INT8 LT operands");
  const c10::cuda::CUDAGuard guard(a.device());
  const auto m=a.size(0),n=w.size(0),k=a.size(1);
  auto handle=at::cuda::getCurrentCUDABlasLtHandle();
  using Key=std::tuple<int,int64_t,int64_t,int64_t>;
  static thread_local std::map<Key,std::unique_ptr<Config>> cache;
  const Key key{a.get_device(),m,n,k};
  auto found=cache.find(key);
  if(found==cache.end()) {
    cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
    TORCH_CHECK(prop.major==7 && prop.minor==5,"cached INT8 LT experiment requires SM75");
    auto config=std::make_unique<Config>();config->initialize(handle,m,n,k);
    found=cache.emplace(key,std::move(config)).first;
  }
  auto& c=*found->second;
  auto out=at::empty({m,n},a.options().dtype(at::kInt));
  const int32_t alpha=1,beta=0;
  check(cublasLtMatmul(handle,c.op,&alpha,w.const_data_ptr(),c.a,a.const_data_ptr(),c.b,&beta,
      out.const_data_ptr(),c.c,out.mutable_data_ptr(),c.c,&c.algo,nullptr,0,c10::cuda::getCurrentCUDAStream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
