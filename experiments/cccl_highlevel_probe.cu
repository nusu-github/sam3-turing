// Standalone feasibility probe; no production dispatch changes or LibTorch dependency.
#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/mdspan>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <exception>

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

int main() { try {
  constexpr int n=128;
  float *input, *output;
  check(cudaMalloc(&input,n*sizeof(float)));
  check(cudaMalloc(&output,n*sizeof(float)));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  auto policy=thrust::cuda::par_nosync.on(stream);
  cuda::std::span<float> destination(output,n);
  using Extents=cuda::std::extents<size_t,8,16>;
  using View=cuda::std::mdspan<float,Extents,cuda::std::layout_stride>;
  View view(input,View::mapping_type(Extents{},cuda::std::array<size_t,2>{16,1}));
  const auto run=[&] {
    thrust::tabulate(policy,input,input+n,[] __device__(int i) { return float(i%9); });
    thrust::transform(policy,input,input+n,output,[] __device__(float x) { return x+1; });
    thrust::for_each_n(policy,thrust::make_counting_iterator(0),n,[=] __device__(int i) {
      destination[i]+=view(i/16,i%16);
    });
    auto pairs=thrust::make_zip_iterator(thrust::make_tuple(input,output));
    thrust::for_each_n(policy,pairs,n,[] __device__(auto pair) {
      thrust::get<1>(pair)+=thrust::get<0>(pair);
    });
  };
  run();check(cudaStreamSynchronize(stream)); // Warm dispatch before capture.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
  run();
  check(cudaStreamEndCapture(stream,&graph));
  check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  for(int i=0;i<3;++i) check(cudaGraphLaunch(exec,stream));
  float actual[n];
  check(cudaMemcpyAsync(actual,output,sizeof(actual),cudaMemcpyDeviceToHost,stream));
  check(cudaStreamSynchronize(stream));
  for(int i=0;i<n;++i) if(actual[i]!=3.f*(i%9)+1.f) return 2;
  check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
  check(cudaFree(output));check(cudaFree(input));check(cudaStreamDestroy(stream));
  std::puts("PASS: tabulate, transform, for_each_n, zip, span, layout_stride mdspan; C++17 SM75 nondefault stream + graph replay");
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; } }
