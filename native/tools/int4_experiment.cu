#include "sam3/int4_experiment.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <climits>
namespace sam3 { namespace {

// Test four clipping ranges against reconstruction SSE; includes unclipped range.
// All threads participate, with one independent decision per row.
template<bool Affine> __device__ void choose_mse_scale(const float (&v)[32],float mx,float mn,float& result,int& zero){
  __shared__ float errors[8],best_error,best_scale;__shared__ int best_zero;
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  if(threadIdx.x==0)best_error=3.402823466e38f;__syncthreads();
  #pragma unroll
  for(int candidate=0;candidate<4;++candidate){
    const float factor=1.f-.15f*candidate;
    const float sc=fmaxf((mx*factor-(Affine?mn:0.f))/(Affine?15.f:7.f),1e-12f);
    const int zp=Affine?__float2int_rn(fminf(15.f,fmaxf(0.f,-mn/sc))):0;
    float error=0;
    #pragma unroll
    for(int j=0;j<32;++j){int q=__float2int_rn(fminf(Affine?15.f:7.f,fmaxf(Affine?0.f:-7.f,v[j]/sc+zp)));float delta=v[j]-(q-zp)*sc;error+=delta*delta;}
    for(int d=16;d;d>>=1)error+=__shfl_down_sync(0xffffffff,error,d);
    if(lane==0)errors[warp]=error;__syncthreads();
    if(threadIdx.x==0){float total=0;for(int w=0;w<8;++w)total+=errors[w];if(total<best_error){best_error=total;best_scale=sc;best_zero=zp;}}
    __syncthreads();
  }
  result=best_scale;zero=best_zero;
}
// One thread owns both nibbles of each byte; no shared-byte write races.
template<bool Boundary,bool MSE=false,int Rotation=0> __global__ void quant4(const half* x,const int* acc,const float* xs,const float* ws,const half* bias,unsigned char* q,float* scales,int n){
  const int row=blockIdx.x;
  float v[32],mx=0;
  #pragma unroll
  for(int j=0;j<32;++j){const int col=threadIdx.x*2+(j/2)*512+j%2;float z=0;
    if(col<n){if constexpr(Boundary){z=float(acc[row*n+col])*xs[row]*ws[col]+__half2float(bias[col]);z=.5f*z*(1.f+erff(z*.7071067811865475f));z=__half2float(__float2half_rn(z));}else z=__half2float(x[row*n+col]);}
    v[j]=z;mx=fmaxf(mx,fabsf(z));}
  // Normalized regular H4 = (J - 2*reverse(I))/2, Kronecker-composed.
  // Two adjacent channels/thread make groups of 64 exactly one warp.
  // Inspired by ConvRot, arXiv:2512.03673; independent shuffle implementation.
  if constexpr(Rotation>0){mx=0;
    #pragma unroll
    for(int j=0;j<16;++j){float lo=v[2*j],hi=v[2*j+1];
      float peerlo=__shfl_xor_sync(0xffffffff,lo,1),peerhi=__shfl_xor_sync(0xffffffff,hi,1);
      float sum=(lo+hi)+(peerlo+peerhi);lo=sum*.5f-peerhi;hi=sum*.5f-peerlo;
      #pragma unroll
      for(int step=2;step<Rotation/2;step*=4){
        float al=__shfl_xor_sync(0xffffffff,lo,step),bl=__shfl_xor_sync(0xffffffff,lo,2*step),cl=__shfl_xor_sync(0xffffffff,lo,3*step);
        float ah=__shfl_xor_sync(0xffffffff,hi,step),bh=__shfl_xor_sync(0xffffffff,hi,2*step),ch=__shfl_xor_sync(0xffffffff,hi,3*step);
        lo=((lo+al)+(bl+cl))*.5f-cl;hi=((hi+ah)+(bh+ch))*.5f-ch;
      }
      v[2*j]=__half2float(__float2half_rn(lo));v[2*j+1]=__half2float(__float2half_rn(hi));
      mx=fmaxf(mx,fmaxf(fabsf(v[2*j]),fabsf(v[2*j+1])));
    }
  }
  using Reduce = cub::BlockReduce<float,256,cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float scale,row_max;
  mx=Reduce(storage).Reduce(mx,cuda::maximum<>{});
  if(threadIdx.x==0){row_max=mx;scale=fmaxf(mx/7.f,1e-12f);scales[row]=scale;}
  __syncthreads();
  if constexpr(MSE){float selected;int zero;choose_mse_scale<false>(v,row_max,0.f,selected,zero);if(threadIdx.x==0){scale=selected;scales[row]=selected;}__syncthreads();}
  #pragma unroll
  for(int j=0;j<16;++j){int col=threadIdx.x*2+j*512;if(col<n){int lo=__float2int_rn(fminf(7.f,fmaxf(-7.f,v[j*2]/scale)));int hi=__float2int_rn(fminf(7.f,fmaxf(-7.f,v[j*2+1]/scale)));q[row*(n/2)+col/2]=(lo&15)|((hi&15)<<4);}}
}

// Affine quantization represented as signed nibbles to reuse the S4 MMA.
// x ~= scale * (signed_nibble + offset), offset = 8 - unsigned_zero_point.
template<bool Boundary,bool MSE=false> __global__ void quant4_affine(const half* x,const int* acc,const float* xs,const float* ws,const half* bias,unsigned char* q,float* scales,int* offsets,int n){
  const int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
  float v[32],mx=0,mn=0;
  #pragma unroll
  for(int j=0;j<32;++j){const int col=threadIdx.x*2+(j/2)*512+j%2;float z=0;
    if(col<n){if constexpr(Boundary){z=float(acc[row*n+col])*xs[row]*ws[col]+__half2float(bias[col]);z=.5f*z*(1.f+erff(z*.7071067811865475f));z=__half2float(__float2half_rn(z));}else z=__half2float(x[row*n+col]);}
    v[j]=z;mx=fmaxf(mx,z);mn=fminf(mn,z);}
  for(int d=16;d;d>>=1){mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));mn=fminf(mn,__shfl_down_sync(0xffffffff,mn,d));}
  __shared__ float maxima[8],minima[8],scale,row_max,row_min;__shared__ int zp;
  if(lane==0){maxima[warp]=mx;minima[warp]=mn;}__syncthreads();
  if(warp==0){mx=lane<8?maxima[lane]:0;mn=lane<8?minima[lane]:0;
    for(int d=16;d;d>>=1){mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,d));mn=fminf(mn,__shfl_down_sync(0xffffffff,mn,d));}
    if(lane==0){row_max=mx;row_min=mn;scale=fmaxf((mx-mn)/15.f,1e-12f);zp=__float2int_rn(fminf(15.f,fmaxf(0.f,-mn/scale)));scales[row]=scale;offsets[row]=8-zp;}}
  __syncthreads();
  if constexpr(MSE){float selected;int zero;choose_mse_scale<true>(v,row_max,row_min,selected,zero);if(threadIdx.x==0){scale=selected;zp=zero;scales[row]=selected;offsets[row]=8-zero;}__syncthreads();}
  #pragma unroll
  for(int j=0;j<16;++j){int col=threadIdx.x*2+j*512;if(col<n){int lo=__float2int_rn(fminf(15.f,fmaxf(0.f,v[j*2]/scale+zp)))-8;int hi=__float2int_rn(fminf(15.f,fmaxf(0.f,v[j*2+1]/scale+zp)))-8;q[row*(n/2)+col/2]=(lo&15)|((hi&15)<<4);}}
}
__global__ void correct4(int* a,const int* offsets,const int* sums,int n,int count){int i=blockIdx.x*256+threadIdx.x;if(i<count)a[i]+=offsets[i/n]*sums[i%n];}
void check_matrix(const at::Tensor& t,at::ScalarType type){TORCH_CHECK(t.is_cuda() && t.scalar_type()==type && t.dim()==2 && t.is_contiguous() && t.size(0)>0 && t.size(1)>0 && t.size(1)<=8192 && t.size(1)%32==0 && t.numel()<INT_MAX,"invalid INT4 matrix");}
template<int M,int N,int WM,int WN>using G=cutlass::gemm::device::Gemm<cutlass::int4b_t,cutlass::layout::RowMajor,cutlass::int4b_t,cutlass::layout::ColumnMajor,int32_t,cutlass::layout::RowMajor,int32_t,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,cutlass::gemm::GemmShape<M,N,128>,cutlass::gemm::GemmShape<WM,WN,128>,cutlass::gemm::GemmShape<8,8,32>,cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t,cutlass::epilogue::thread::ScaleType::Nothing>,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,32,32,false,cutlass::arch::OpMultiplyAddSaturate>;
template<class Gemm>void launch(const at::Tensor& a,const at::Tensor& w,at::Tensor& out){int m=int(a.size(0)),n=int(w.size(0)),k=int(a.size(1)*2);
  typename Gemm::Arguments args({m,n,k},{reinterpret_cast<const cutlass::int4b_t*>(a.const_data_ptr<uint8_t>()),k},{reinterpret_cast<const cutlass::int4b_t*>(w.const_data_ptr<uint8_t>()),k},{out.const_data_ptr<int>(),n},{out.mutable_data_ptr<int>(),n},typename Gemm::EpilogueOutputOp::Params(1,0));
  TORCH_CHECK(Gemm::can_implement(args)==cutlass::Status::kSuccess && Gemm::get_workspace_size(args)==0,"unsupported INT4 GEMM");Gemm op;TORCH_CHECK(op(args,nullptr,c10::cuda::getCurrentCUDAStream())==cutlass::Status::kSuccess,"INT4 GEMM launch failed");}
}
std::tuple<at::Tensor,at::Tensor> int4_quant(const at::Tensor& x,bool mse){check_matrix(x,at::kHalf);c10::cuda::CUDAGuard guard(x.device());auto q=at::empty({x.size(0),x.size(1)/2},x.options().dtype(at::kByte)),s=at::empty({x.size(0)},x.options().dtype(at::kFloat));if(mse){quant4<false,true><<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),nullptr,nullptr,nullptr,nullptr,q.mutable_data_ptr<uint8_t>(),s.mutable_data_ptr<float>(),int(x.size(1)));}else{quant4<false><<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),nullptr,nullptr,nullptr,nullptr,q.mutable_data_ptr<uint8_t>(),s.mutable_data_ptr<float>(),int(x.size(1)));};C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,s};}
std::tuple<at::Tensor,at::Tensor> int4_boundary(const at::Tensor& a,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias){check_matrix(a,at::kInt);for(const auto& t:{xs,ws,bias})TORCH_CHECK(t.device()==a.device() && t.is_contiguous() && t.dim()==1,"INT4 boundary layout/device mismatch");TORCH_CHECK(xs.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat && bias.scalar_type()==at::kHalf && xs.numel()==a.size(0) && ws.numel()==a.size(1) && bias.numel()==a.size(1),"invalid INT4 boundary operands");c10::cuda::CUDAGuard guard(a.device());auto q=at::empty({a.size(0),a.size(1)/2},a.options().dtype(at::kByte)),s=at::empty({a.size(0)},a.options().dtype(at::kFloat));quant4<true><<<a.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(nullptr,a.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<uint8_t>(),s.mutable_data_ptr<float>(),int(a.size(1)));C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,s};}

std::tuple<at::Tensor,at::Tensor,at::Tensor> int4_quant_affine(const at::Tensor& x,bool mse){
  check_matrix(x,at::kHalf);c10::cuda::CUDAGuard guard(x.device());
  auto q=at::empty({x.size(0),x.size(1)/2},x.options().dtype(at::kByte)),sc=at::empty({x.size(0)},x.options().dtype(at::kFloat)),z=at::empty({x.size(0)},x.options().dtype(at::kInt));
  if(mse){quant4_affine<false,true><<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),nullptr,nullptr,nullptr,nullptr,q.mutable_data_ptr<uint8_t>(),sc.mutable_data_ptr<float>(),z.mutable_data_ptr<int>(),int(x.size(1)));}else{quant4_affine<false><<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),nullptr,nullptr,nullptr,nullptr,q.mutable_data_ptr<uint8_t>(),sc.mutable_data_ptr<float>(),z.mutable_data_ptr<int>(),int(x.size(1)));};C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,sc,z};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> int4_boundary_affine(const at::Tensor& a,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias,bool mse){
  check_matrix(a,at::kInt);for(const auto& t:{xs,ws,bias})TORCH_CHECK(t.device()==a.device() && t.is_contiguous() && t.dim()==1,"affine boundary layout/device mismatch");
  TORCH_CHECK(xs.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat && bias.scalar_type()==at::kHalf && xs.numel()==a.size(0) && ws.numel()==a.size(1) && bias.numel()==a.size(1),"invalid affine boundary operands");
  c10::cuda::CUDAGuard guard(a.device());auto q=at::empty({a.size(0),a.size(1)/2},a.options().dtype(at::kByte)),sc=at::empty({a.size(0)},a.options().dtype(at::kFloat)),z=at::empty({a.size(0)},a.options().dtype(at::kInt));
  if(mse){quant4_affine<true,true><<<a.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(nullptr,a.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<uint8_t>(),sc.mutable_data_ptr<float>(),z.mutable_data_ptr<int>(),int(a.size(1)));}else{quant4_affine<true><<<a.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(nullptr,a.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<uint8_t>(),sc.mutable_data_ptr<float>(),z.mutable_data_ptr<int>(),int(a.size(1)));};C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,sc,z};
}
void int4_correct(at::Tensor& a,const at::Tensor& z,const at::Tensor& sums){
  TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kInt && a.is_contiguous() && a.dim()==2 && a.numel()>0 && a.numel()<INT_MAX && z.dim()==1 && sums.dim()==1 && z.numel()==a.size(0) && sums.numel()==a.size(1),"invalid affine correction shape");
  for(const auto& t:{z,sums})TORCH_CHECK(t.device()==a.device() && t.scalar_type()==at::kInt && t.is_contiguous(),"invalid affine correction operands");
  c10::cuda::CUDAGuard guard(a.device());correct4<<<(a.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(a.mutable_data_ptr<int>(),z.const_data_ptr<int>(),sums.const_data_ptr<int>(),int(a.size(1)),int(a.numel()));C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor,at::Tensor> int4_quant_rht(const at::Tensor& x,int group){
  check_matrix(x,at::kHalf);TORCH_CHECK((group==16 || group==64) && x.size(1)%group==0,"invalid regular rotation group");
  c10::cuda::CUDAGuard guard(x.device());auto q=at::empty({x.size(0),x.size(1)/2},x.options().dtype(at::kByte)),sc=at::empty({x.size(0)},x.options().dtype(at::kFloat));
#define RHT_QUANT(GROUP) quant4<false,false,GROUP><<<x.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),nullptr,nullptr,nullptr,nullptr,q.mutable_data_ptr<uint8_t>(),sc.mutable_data_ptr<float>(),int(x.size(1)))
  if(group==16){RHT_QUANT(16);}else{RHT_QUANT(64);}
#undef RHT_QUANT
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,sc};
}
std::tuple<at::Tensor,at::Tensor> int4_boundary_rht(const at::Tensor& a,const at::Tensor& xs,const at::Tensor& ws,const at::Tensor& bias,int group){
  check_matrix(a,at::kInt);TORCH_CHECK((group==16 || group==64) && a.size(1)%group==0,"invalid regular rotation group");
  for(const auto& t:{xs,ws,bias})TORCH_CHECK(t.device()==a.device() && t.is_contiguous() && t.dim()==1,"RHT boundary layout/device mismatch");
  TORCH_CHECK(xs.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat && bias.scalar_type()==at::kHalf && xs.numel()==a.size(0) && ws.numel()==a.size(1) && bias.numel()==a.size(1),"invalid RHT boundary operands");
  c10::cuda::CUDAGuard guard(a.device());auto q=at::empty({a.size(0),a.size(1)/2},a.options().dtype(at::kByte)),sc=at::empty({a.size(0)},a.options().dtype(at::kFloat));
#define RHT_BOUNDARY(GROUP) quant4<true,false,GROUP><<<a.size(0),256,0,c10::cuda::getCurrentCUDAStream()>>>(nullptr,a.const_data_ptr<int>(),xs.const_data_ptr<float>(),ws.const_data_ptr<float>(),reinterpret_cast<const half*>(bias.const_data_ptr<at::Half>()),q.mutable_data_ptr<uint8_t>(),sc.mutable_data_ptr<float>(),int(a.size(1)))
  if(group==16){RHT_BOUNDARY(16);}else{RHT_BOUNDARY(64);}
#undef RHT_BOUNDARY
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,sc};
}
at::Tensor int4_mm(const at::Tensor& a,const at::Tensor& w,int tile){
  TORCH_CHECK(a.is_cuda() && a.scalar_type()==at::kByte && w.scalar_type()==at::kByte && a.dim()==2 && w.dim()==2 && a.is_contiguous() && w.is_contiguous() && a.device()==w.device() && a.size(0)>0 && w.size(0)>0 && a.size(1)==w.size(1) && a.size(1)>0 && a.size(1)<=4096 && a.size(1)%16==0 && w.size(0)%4==0 && a.numel()*2<INT_MAX && w.numel()*2<INT_MAX && a.size(0)*w.size(0)<INT_MAX,"invalid packed INT4 GEMM operands");
  c10::cuda::CUDAGuard guard(a.device());cudaDeviceProp prop;C10_CUDA_CHECK(cudaGetDeviceProperties(&prop,a.get_device()));TORCH_CHECK(prop.major==7 && prop.minor==5,"INT4 experiment requires SM75");auto out=at::empty({a.size(0),w.size(0)},a.options().dtype(at::kInt));
  if(tile==0)launch<G<128,128,64,64>>(a,w,out);else if(tile==1)launch<G<64,128,32,64>>(a,w,out);else TORCH_CHECK(false,"unknown INT4 tile");C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}
