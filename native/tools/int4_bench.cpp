#include "../src/approx_kernels.h"
#include "sam3/int4_experiment.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <cmath>
#include <cstdint>
uint64_t quant_hash=14695981039346656037ull;
void fingerprint(const at::Tensor& tensor){
 auto cpu=tensor.cpu().contiguous();auto* bytes=static_cast<const unsigned char*>(cpu.const_data_ptr());
 for(size_t i=0;i<cpu.nbytes();++i){quant_hash^=bytes[i];quant_hash*=1099511628211ull;}
}
at::Tensor unpack(const at::Tensor& p){auto lo=(p.to(at::kInt)&15),hi=at::bitwise_right_shift(p.to(at::kInt),4);lo=at::where(lo>=8,lo-16,lo);hi=at::where(hi>=8,hi-16,hi);return at::stack({lo,hi},-1).reshape({p.size(0),p.size(1)*2}).to(at::kChar);}
double measure(const std::function<at::Tensor()>& fn){for(int i=0;i<10;++i)fn();std::vector<float> t;for(int i=0;i<20;++i){at::cuda::CUDAEvent a(cudaEventDefault),b(cudaEventDefault);a.record();auto out=fn();b.record();b.synchronize();t.push_back(a.elapsed_time(b));}std::sort(t.begin(),t.end());return (t[9]+t[10])*.5;}
int main(int argc,char**argv){try{
 TORCH_CHECK(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"usage: int4_bench OUTPUT [--check-only]");c10::InferenceMode inf;c10::cuda::CUDAGuard dev(0);at::manual_seed(73918);auto stream=c10::cuda::getStreamFromPool(false,0);c10::cuda::CUDAStreamGuard guard(stream);auto opt=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);std::ofstream f(argv[1]);f<<"{\"cases\":[";bool first=true;
 for(auto shape:std::vector<std::vector<int64_t>>{{80,80,128},{5184,1024,4736}}){auto m=shape[0],n=shape[1],k=shape[2];auto x=at::randn({m,k},opt),w=at::randn({n,k},opt);at::Tensor aq,as,wq,ws;
 for(int kind=0;kind<3;++kind){if(kind==1){x.zero_();w.zero_();}if(kind==2){x.normal_();w.normal_();}std::tie(aq,as)=sam3::int4_quant(x);std::tie(wq,ws)=sam3::int4_quant(w);
 for(const auto& pair:std::vector<std::pair<at::Tensor,at::Tensor>>{{x,aq},{w,wq}}){auto cpu=pair.first.to(at::kCPU).to(at::kFloat).contiguous();auto ref=at::empty(cpu.sizes(),cpu.options().dtype(at::kChar));auto pv=cpu.const_data_ptr<float>();auto pr=ref.mutable_data_ptr<int8_t>();for(int64_t row=0;row<cpu.size(0);++row){float mx=0;for(int64_t c=0;c<k;++c)mx=std::max(mx,std::abs(pv[row*k+c]));float scale=std::max(mx/7.f,1e-12f);for(int64_t c=0;c<k;++c)pr[row*k+c]=int8_t(std::nearbyint(std::clamp(pv[row*k+c]/scale,-7.f,7.f)));}TORCH_CHECK(at::equal(ref,unpack(pair.second).cpu()),"INT4 quant mismatch");}
 for(const auto& t:{aq,as,wq,ws})fingerprint(t);
 auto ref=at::_int_mm(unpack(aq),unpack(wq).t());for(int tile=0;tile<2;++tile)TORCH_CHECK(at::equal(ref,sam3::int4_mm(aq,wq,tile)),"INT4 GEMM mismatch");}
 // All 256 byte patterns exercise both signed nibbles, including -8.
 auto rawA=at::randint(0,256,aq.sizes(),aq.options()),rawW=at::randint(0,256,wq.sizes(),wq.options());auto rawRef=at::_int_mm(unpack(rawA),unpack(rawW).t());for(int tile=0;tile<2;++tile)TORCH_CHECK(at::equal(rawRef,sam3::int4_mm(rawA,rawW,tile)),"signed nibble mismatch");
 auto acc=at::randint(-100000,100001,{m,k},opt.dtype(at::kInt)),xs=at::rand({m},opt.dtype(at::kFloat))*.002,bs=at::rand({k},opt.dtype(at::kFloat))*.005,bias=(at::randn({k},opt)*.02);auto hidden=at::empty({m,k},opt);approx_restore(acc,xs,bs,bias,hidden,true);auto separate=sam3::int4_quant(hidden),fused=sam3::int4_boundary(acc,xs,bs,bias);TORCH_CHECK(at::equal(std::get<0>(separate),std::get<0>(fused)) && at::equal(std::get<1>(separate),std::get<1>(fused)),"INT4 boundary mismatch");

 for(bool mse:{false,true})for(int kind=0;kind<3;++kind){auto input=kind==0?hidden:kind==1?at::zeros_like(hidden):at::full_like(hidden,-.125);
 auto [pq,ps,pz]=sam3::int4_quant_affine(input,mse);auto values=unpack(pq);auto dense=(values.to(at::kInt)+pz.unsqueeze(1)).to(at::kChar);
 for(const auto& t:{pq,ps,pz})fingerprint(t);
 auto expected=at::_int_mm(dense,unpack(wq).t()),actual=sam3::int4_mm(pq,wq);auto sums=unpack(wq).sum(1,false,at::kInt).contiguous();sam3::int4_correct(actual,pz,sums);TORCH_CHECK(at::equal(expected,actual),"affine GEMM correction mismatch");
 auto cpu=input.to(at::kCPU).to(at::kFloat).contiguous(),sc=ps.cpu(),zs=pz.cpu(),dq=values.cpu();auto pv=cpu.const_data_ptr<float>();auto psr=sc.const_data_ptr<float>();auto pzr=zs.const_data_ptr<int>();auto pqr=dq.const_data_ptr<int8_t>();
 for(int64_t row=0;row<m;++row){float mn=0,mx=0;for(int64_t c=0;c<k;++c){mn=std::min(mn,pv[row*k+c]);mx=std::max(mx,pv[row*k+c]);}float scale=std::max((mx-mn)/15.f,1e-12f);if(!mse)TORCH_CHECK(std::abs(scale-psr[row])<=1e-6f*scale,"affine scale mismatch");int zp=int(std::nearbyint(std::clamp(-mn/psr[row],0.f,15.f)));TORCH_CHECK(pzr[row]==8-zp,"affine offset mismatch");for(int64_t c=0;c<k;++c)TORCH_CHECK(pqr[row*k+c]==int(std::nearbyint(std::clamp(pv[row*k+c]/psr[row]+float(zp),0.f,15.f)))-8,"affine quant mismatch");}
 if(kind==0){auto [fq,fs,fz]=sam3::int4_boundary_affine(acc,xs,bs,bias,mse);TORCH_CHECK(at::equal(fq,pq)&&at::equal(fs,ps)&&at::equal(fz,pz),"affine boundary mismatch");}}

 {auto [mq,ms]=sam3::int4_quant(w,true);auto plain=(w.to(at::kFloat)-unpack(wq).to(at::kFloat)*ws.unsqueeze(1)).square().sum(1);auto selected=(w.to(at::kFloat)-unpack(mq).to(at::kFloat)*ms.unsqueeze(1)).square().sum(1);TORCH_CHECK((selected<=plain+1e-4).all().item<bool>(),"weight MSE selection worsened SSE");}
 {auto [pq,ps,pz]=sam3::int4_quant_affine(hidden);auto [mq,ms,mz]=sam3::int4_quant_affine(hidden,true);auto plain=(hidden.to(at::kFloat)-(unpack(pq).to(at::kFloat)+pz.unsqueeze(1))*ps.unsqueeze(1)).square().sum(1);auto selected=(hidden.to(at::kFloat)-(unpack(mq).to(at::kFloat)+mz.unsqueeze(1))*ms.unsqueeze(1)).square().sum(1);TORCH_CHECK((selected<=plain+1e-4).all().item<bool>(),"activation MSE selection worsened SSE");}

 for(int group:{16,64}){
   auto h4=(at::ones({4,4},opt.dtype(at::kFloat))-2*at::eye(4,opt.dtype(at::kFloat)).flip({1}))*.5;
   auto h=h4;while(h.size(0)<group)h=at::kron(h,h4);
   TORCH_CHECK(at::equal(at::mm(h,h.t()),at::eye(group,opt.dtype(at::kFloat))),"RHT is not orthogonal");
   auto integers=at::randint(-4,5,{m,k},opt);
   auto rotated=at::mm(integers.to(at::kFloat).reshape({-1,group}),h).reshape({m,k}).to(at::kHalf);
   auto reference=sam3::int4_quant(rotated),actual=sam3::int4_quant_rht(integers,group);
   TORCH_CHECK(at::equal(std::get<0>(reference),std::get<0>(actual)) && at::equal(std::get<1>(reference),std::get<1>(actual)),"RHT shuffle/matrix mismatch");
   auto separate=sam3::int4_quant_rht(hidden,group),fused=sam3::int4_boundary_rht(acc,xs,bs,bias,group);
   fingerprint(std::get<0>(actual));fingerprint(std::get<1>(actual));
   fingerprint(std::get<0>(fused));fingerprint(std::get<1>(fused));
   TORCH_CHECK(at::equal(std::get<0>(separate),std::get<0>(fused)) && at::equal(std::get<1>(separate),std::get<1>(fused)),"RHT boundary mismatch");
 }
 auto au=unpack(aq),wu=unpack(wq);std::vector<std::vector<double>> ts(3);if(m==5184 && argc==2){for(int i=0;i<100;++i)at::_int_mm(au,wu.t());for(int pass=0;pass<4;++pass)for(int j=0;j<3;++j){int mode=pass%2?2-j:j;ts[mode].push_back(measure([&]{return mode==0?at::_int_mm(au,wu.t()):sam3::int4_mm(aq,wq,mode-1);}));}}
 if(!first)f<<',';first=false;f<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"correct\":true,\"gemm_ms\":[";for(int i=0;i<3;++i){if(i)f<<',';f<<'[';for(size_t j=0;j<ts[i].size();++j){if(j)f<<',';f<<ts[i][j];}f<<']';}f<<"],\"quant_ms\":[";
 if(m==5184 && argc==2)for(int mode=0;mode<6;++mode){
   if(mode)f<<',';
   f<<measure([&]{
     if(mode<2)return std::get<0>(sam3::int4_quant(x,mode==1));
     if(mode<4)return std::get<0>(sam3::int4_quant_affine(x,mode==3));
     return std::get<0>(sam3::int4_quant_rht(x,mode==4?16:64));
   });
 }
 f<<"]}";f.flush();std::cout<<m<<" passed"<<std::endl;}
 f<<"],\"quant_fingerprint_fnv1a64\":\""<<quant_hash<<"\"}\n";TORCH_CHECK(f,"write failed");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
