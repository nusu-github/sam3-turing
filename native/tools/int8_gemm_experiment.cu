#include "sam3/int8_gemm_experiment.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <climits>
namespace sam3 {
namespace {
template<int M,int N,int K,int WM,int WN>
using Gemm=cutlass::gemm::device::Gemm<int8_t,cutlass::layout::RowMajor,
    int8_t,cutlass::layout::ColumnMajor,int32_t,cutlass::layout::RowMajor,int32_t,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<M,N,K>,cutlass::gemm::GemmShape<WM,WN,K>,
    cutlass::gemm::GemmShape<8,8,16>,
    cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t,cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,16,16,false,cutlass::arch::OpMultiplyAddSaturate>;
template<class G> void run(const at::Tensor& a,const at::Tensor& w,at::Tensor& out) {
  const int m=int(a.size(0)),n=int(w.size(0)),k=int(a.size(1));
  typename G::Arguments args({m,n,k},{a.const_data_ptr<int8_t>(),k},
      {w.const_data_ptr<int8_t>(),k},{out.const_data_ptr<int32_t>(),n},
      {out.mutable_data_ptr<int32_t>(),n},typename G::EpilogueOutputOp::Params(1,0));
  TORCH_CHECK(G::can_implement(args)==cutlass::Status::kSuccess,"unsupported INT8 GEMM operands");
  TORCH_CHECK(G::get_workspace_size(args)==0,"unexpected INT8 GEMM workspace");
  G op;const auto status=op(args,nullptr,c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==cutlass::Status::kSuccess,"CUTLASS INT8 GEMM failed: ",int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<class G> std::vector<int64_t> resources() {
  using Kernel=typename G::GemmKernel;
  const int smem=sizeof(typename Kernel::SharedStorage);
  cudaFuncAttributes attrs;int blocks=0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs,cutlass::Kernel<Kernel>));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,cutlass::Kernel<Kernel>,Kernel::kThreadCount,smem));
  return {attrs.numRegs,int64_t(attrs.localSizeBytes),smem,Kernel::kThreadCount,blocks};
}
#define TILES(F) \
  case 0:return F<Gemm<128,128,64,64,64>>(); \
  case 1:return F<Gemm<64,64,64,32,32>>(); \
  case 2:return F<Gemm<64,128,64,32,64>>(); \
  case 3:return F<Gemm<128,64,64,64,32>>(); \
  case 4:return F<Gemm<64,64,32,32,32>>(); \
  case 5:return F<Gemm<64,128,32,32,64>>();
}
at::Tensor int8_gemm_experiment(const at::Tensor& a,const at::Tensor& w,int tile) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kChar && w.scalar_type()==at::kChar &&
      w.device()==a.device() && a.dim()==2 && w.dim()==2 && a.is_contiguous() && w.is_contiguous() &&
      a.size(1)==w.size(1) && a.size(0)>0 && w.size(0)>0 && a.size(1)>0 && a.size(1)<=8192 &&
      a.size(1)%16==0 && w.size(0)%4==0 && a.numel()<INT_MAX && w.numel()<INT_MAX &&
      a.size(0)*w.size(0)<INT_MAX,"invalid experimental INT8 GEMM shape/layout/type");
  const c10::cuda::CUDAGuard guard(a.device());
  cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
  TORCH_CHECK(prop.major==7 && prop.minor==5,"INT8 GEMM experiment requires SM75");
  auto out=at::empty({a.size(0),w.size(0)},a.options().dtype(at::kInt));
  switch(tile) {
    case 0:run<Gemm<128,128,64,64,64>>(a,w,out);break;
    case 1:run<Gemm<64,64,64,32,32>>(a,w,out);break;
    case 2:run<Gemm<64,128,64,32,64>>(a,w,out);break;
    case 3:run<Gemm<128,64,64,64,32>>(a,w,out);break;
    case 4:run<Gemm<64,64,32,32,32>>(a,w,out);break;
    case 5:run<Gemm<64,128,32,32,64>>(a,w,out);break;
    default:TORCH_CHECK(false,"unknown INT8 GEMM tile");
  }
  return out;
}
std::vector<int64_t> int8_gemm_resources(int tile) {
  switch(tile) {TILES(resources) default:TORCH_CHECK(false,"unknown INT8 GEMM tile");}
}
#undef TILES
}
