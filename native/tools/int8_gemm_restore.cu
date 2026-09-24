#include "sam3/int8_gemm_experiment.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <climits>
namespace sam3 {
namespace {
using namespace cute;
namespace ep = cutlass::epilogue::threadblock;
using Shape=cutlass::gemm::GemmShape<128,128,64>;
using Warp=cutlass::gemm::GemmShape<64,64,64>;
using Half=cutlass::half_t;
using Map=ep::OutputTileThreadLayout<Shape,Warp,Half,8,1>;
using XScale=ep::VisitorColBroadcast<Map,float>;
using WScale=ep::VisitorRowBroadcast<Map,float,Stride<_0,_1,_0>>;
using Bias=ep::VisitorRowBroadcast<Map,Half,Stride<_0,_1,_0>>;
using Mul=ep::VisitorCompute<cutlass::multiplies,float,float,cutlass::FloatRoundStyle::round_to_nearest>;
using Madd=ep::VisitorCompute<cutlass::multiply_add,Half,float,cutlass::FloatRoundStyle::round_to_nearest>;
using Scaled=ep::Sm80EVT<Mul,ep::VisitorAccFetch,XScale>;
using Restored=ep::Sm80EVT<Madd,Scaled,WScale,Bias>;
using Store=ep::VisitorAuxStore<Map,Half,cutlass::FloatRoundStyle::round_to_nearest,Stride<int64_t,_1,int64_t>>;
using Tree=ep::Sm80EVT<Store,Restored>;
using Kernel=typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
  int8_t,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,16,
  int8_t,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,16,
  Half,cutlass::layout::RowMajor,8,int32_t,float,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,Shape,Warp,cutlass::gemm::GemmShape<8,8,16>,Tree,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,cutlass::arch::OpMultiplyAddSaturate>::GemmKernel;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
}
at::Tensor int8_gemm_restore_experiment(const at::Tensor& a,const at::Tensor& w,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kChar && w.scalar_type()==at::kChar &&
      a.dim()==2 && w.dim()==2 && a.size(1)==w.size(1) && a.size(0)>0 && w.size(0)>0 &&
      a.size(1)>0 && a.size(1)<=8192 && a.size(1)%16==0 && w.size(0)%8==0 &&
      a.numel()<INT_MAX && w.numel()<INT_MAX && a.size(0)*w.size(0)<INT_MAX,"invalid fused GEMM inputs");
  for(const auto& t:{a,w,xs,ws,bias})TORCH_CHECK(t.device()==a.device() && t.is_contiguous(),"fused GEMM device/layout mismatch");
  TORCH_CHECK(xs.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat && bias.scalar_type()==at::kHalf &&
      xs.dim()==1 && ws.dim()==1 && bias.dim()==1 && xs.numel()==a.size(0) && ws.numel()==w.size(0) && bias.numel()==w.size(0),"invalid restore operands");
  const c10::cuda::CUDAGuard guard(a.device());
  cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));
  TORCH_CHECK(prop.major==7 && prop.minor==5,"fused GEMM experiment requires SM75");
  const int m=int(a.size(0)),n=int(w.size(0)),k=int(a.size(1));
  auto out=at::empty({m,n},a.options().dtype(at::kHalf));
  Tree::Arguments tree{{{{},{xs.const_data_ptr<float>(),0,{}},{}},
      {ws.const_data_ptr<float>(),0,{}},{reinterpret_cast<const Half*>(bias.const_data_ptr<at::Half>()),Half(0),{}},{}},
      {reinterpret_cast<Half*>(out.mutable_data_ptr<at::Half>()),{int64_t(n),_1{},int64_t(m)*n}}};
  Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k},1,tree,
      a.const_data_ptr<int8_t>(),w.const_data_ptr<int8_t>(),nullptr,nullptr,
      int64_t(m)*k,int64_t(n)*k,0,0,k,k,0,0);
  TORCH_CHECK(Gemm::can_implement(args)==cutlass::Status::kSuccess,"unsupported fused GEMM");
  TORCH_CHECK(Gemm::get_workspace_size(args)==0,"unexpected fused GEMM workspace");
  Gemm op;auto status=op(args,nullptr,c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==cutlass::Status::kSuccess,"fused GEMM failed: ",int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
