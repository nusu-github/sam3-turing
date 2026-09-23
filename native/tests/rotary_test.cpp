#include "sam3/rotary.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#ifdef SAM3_TEST_CUDA_STREAM
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <iostream>
#include <limits>

at::Tensor reference(const at::Tensor& x, const at::Tensor& f) {
  return at::view_as_real(at::view_as_complex(x.to(at::kFloat).reshape(
      {x.size(0), x.size(1), x.size(2), x.size(3)/2, 2})) *
      f.view({1,1,x.size(2),x.size(3)/2})).flatten(3).to(x.scalar_type());
}
int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 2, "expected cpu|cuda");
    c10::InferenceMode inference; at::set_num_threads(4); at::manual_seed(2841);
    const at::Device device(argv[1]);
#ifdef SAM3_TEST_CUDA_STREAM
    const auto stream = c10::cuda::getStreamFromPool();
    const c10::cuda::CUDAStreamGuard stream_guard(stream);
#endif
    int cases = 0;
    for (auto dtype : {at::kFloat, at::kHalf, at::kBFloat16}) {
      const auto options = at::TensorOptions().device(device).dtype(dtype);
      for (const auto& shape : std::vector<std::vector<int64_t>>{
               {1,16,576,64}, {2,16,5184,64}, {1,1,5184,256}, {2,8,72,32}}) {
        const auto b=shape[0], h=shape[1], n=shape[2], d=shape[3];
        auto base = at::randn({b,n+3,3,h,d}, options);
        for (int layout = 0; layout < 3; ++layout) {
          auto x=base.select(2,1).permute({0,2,1,3}).slice(2,1,n+1);
          if (layout==1) x=x.contiguous();
          if (layout==2) x=x.slice(0,0,1).expand({b,h,n,d});
          const auto saved=x.clone();
          auto angles=at::randn({n,d/2},options.dtype(at::kFloat));
          auto f=at::polar(at::ones_like(angles),angles);
          const auto expected=reference(x,f),actual=sam3::rotary_embedding(x,f);
          TORCH_CHECK(expected.sizes()==actual.sizes() && expected.strides()==actual.strides(),
                      "rotary layout mismatch: ",expected.strides()," vs ",actual.strides());
          TORCH_CHECK(at::equal(expected.contiguous().view(at::kByte),actual.contiguous().view(at::kByte)),
                      "rotary bit mismatch dtype=",dtype," shape=",x.sizes()," layout=",layout);
          TORCH_CHECK(at::equal(x,saved),"rotary input modified");
          ++cases;
        }
      }
      // Lazy conjugate and strided frequency views use the reference path.
      auto x=at::randn({2,3,7,8},options);
      auto f=at::randn({7,8},options.dtype(at::kComplexFloat)).slice(1,0,8,2).conj();
      TORCH_CHECK(at::equal(reference(x,f),sam3::rotary_embedding(x,f)),"frequency fallback");
      ++cases;
    }
    auto x=at::zeros({1,1,3,4},at::TensorOptions().device(device));
    auto f=at::ones({3,2},x.options().dtype(at::kComplexFloat));
    for(auto dtype:{at::kFloat,at::kHalf,at::kBFloat16}) {
      const float inf=std::numeric_limits<float>::infinity();
      auto special=at::tensor({0.f,-0.f,inf,-inf,std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::denorm_min(),1e-30f,1e30f}).to(device).to(dtype).view({1,1,2,4});
      auto freq=f.slice(0,0,2);
      auto a=reference(special,freq),b=sam3::rotary_embedding(special,freq);
      TORCH_CHECK(at::equal(a.contiguous().view(at::kByte),b.contiguous().view(at::kByte)),
                  "rotary nonfinite/subnormal/signed-zero mismatch");
      ++cases;
    }
    auto empty=x.slice(0,0,0);
    auto a=reference(empty,f),b=sam3::rotary_embedding(empty,f);
    TORCH_CHECK(a.sizes()==b.sizes() && a.strides()==b.strides(),"empty layout mismatch");
    auto wide=x.to(at::kDouble);
    TORCH_CHECK(at::equal(reference(wide,f),sam3::rotary_embedding(wide,f)),"FP64 fallback");
    cases+=2;
    // Cancellation exposes which product was rounded before FMA contraction.
    // Compare with this LibTorch's reference rather than imposing one platform's
    // contraction order on all supported builds.
    auto cancellation=at::tensor({0x1.000002p0f,-1.f}).to(device).view({1,1,1,2});
    auto cancellation_freq=at::view_as_complex(
        at::tensor({1.f,0x1.fffffep-1f}).to(device).view({1,1,2}));
    auto cancellation_expected=reference(cancellation,cancellation_freq);
    auto cancellation_actual=sam3::rotary_embedding(cancellation,cancellation_freq);
    TORCH_CHECK(at::equal(cancellation_expected.contiguous().view(at::kByte),
                         cancellation_actual.contiguous().view(at::kByte)),
                "rotary cancellation/FMA mismatch");
    ++cases;
    int rejected=0;
    for(const auto& bad:std::vector<at::Tensor>{at::real(f),f.slice(0,0,1)}){
      try{sam3::rotary_embedding(x,bad);}catch(const c10::Error&){++rejected;}
    }
    TORCH_CHECK(rejected==2,"invalid frequencies accepted");
    std::cout<<"PASS rotary "<<device<<" exact cases="<<cases<<" invalid="<<rejected<<'\n';
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
