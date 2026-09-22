#include "sam3/multiplex.h"
#include "sam3/autocast.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <numeric>

namespace {
template<class F> void rejects(F&& operation) {
  bool rejected=false;
  try {operation();} catch (const c10::Error&) {rejected=true;}
  TORCH_CHECK(rejected,"invalid operation unexpectedly succeeded");
}
void require_equal(const at::Tensor& a,const at::Tensor& b) {
  TORCH_CHECK(a.sizes()==b.sizes() && at::equal(a,b),"multiplex tensor mismatch");
}
}

int main(int argc,char** argv) {
  try {
    c10::InferenceMode inference;at::set_num_threads(4);
    // Exact FP32 round trips require full-precision matmul, as in parity tests.
    // The library itself leaves the caller's backend policy unchanged.
    at::globalContext().setAllowTF32CuBLAS(false);
    const at::Device device(argc>1?argv[1]:"cpu");
    const std::string mode=argc>2?argv[2]:"fp32";
    TORCH_CHECK(mode=="fp32" || mode=="fp16" || mode=="bf16_reference","invalid precision");
    const auto dtype=mode=="fp32"?at::kFloat:(mode=="fp16"?at::kHalf:at::kBFloat16);
    sam3::AutocastGuard autocast(device.type(),mode!="fp32",dtype);
    const auto options=at::TensorOptions().device(device).dtype(dtype);
    using State=sam3::MultiplexState;
    State state({{0,1,2,3},{4,5,-1,-1}},device,dtype,4,std::vector<int64_t>{100,101,102,103,104,105});
    TORCH_CHECK(state.remove_objects({0,4})==std::vector<int64_t>({0,1}),"wrong retained buckets");
    TORCH_CHECK(state.object_count()==4 && state.occupied_count()==6 && state.available_slots()==2,"removed slots must remain occupied");
    const auto assignments=state.assignments();const auto ids=state.object_ids();const auto matrix=state.mux_matrix().clone();
    // Both failures occur after a partial modification in the source. Native
    // operations validate/allocate on a copy so callers can recover safely.
    rejects([&] {state.add_objects({4,5,6},std::vector<int64_t>{106,107,108});});
    rejects([&] {state.remove_objects({0,900});});
    rejects([&] {state.add_objects({5},std::vector<int64_t>{106},true);});
    rejects([&] {state.add_objects({4},std::nullopt,true);});
    rejects([&] {state.add_objects({4},std::vector<int64_t>{106},false,true);});
    TORCH_CHECK(state.assignments()==assignments && state.object_ids()==ids,"failed mutation changed state");
    require_equal(state.mux_matrix(),matrix);
    const auto saved=state;
    state.add_objects({4,5},std::vector<int64_t>{106,107});
    TORCH_CHECK(saved.assignments()==assignments && saved.object_ids()==ids,"copy changed during mutation");
    require_equal(saved.mux_matrix(),matrix);
    TORCH_CHECK(state.remove_objects({0,1,2})==std::vector<int64_t>({1}),"empty bucket must be dropped");
    TORCH_CHECK(state.assignments()==State::Assignments({{State::removed,0,1,2}}),"wrong dense renumbering");
    TORCH_CHECK(*state.object_ids()==std::vector<int64_t>({105,106,107}),"external IDs changed identity");
    rejects([&] {state.next_indices(1);});
    state.add_objects(state.next_indices(1,true),std::vector<int64_t>{108},true);
    state.add_objects(state.next_indices(1,true,true),std::vector<int64_t>{109},true,true);
    TORCH_CHECK(state.bucket_count()==3 && state.object_count()==5,"prefer-new allocation must skip available slots");
    for (const auto& shape:std::vector<std::vector<int64_t>>{{5},{5,0,3},{5,2,3}}) {
      const auto x=at::randn(shape,options);require_equal(state.demux(state.mux(x)),x);
    }
    const auto noncontiguous=at::randn({5,3,7},options).transpose(1,2);
    require_equal(state.demux(state.mux(noncontiguous)),noncontiguous);
    TORCH_CHECK(state.remove_objects({999},false)==std::vector<int64_t>({0,1,2}),"non-strict unknown removal changed buckets");
    TORCH_CHECK(state.remove_objects({0,1,2,3,4}).empty(),"all-removed state retained buckets");
    TORCH_CHECK(!state.valid() && state.bucket_count()==0 && state.object_count()==0 && state.object_ids()->empty(),"invalid state metadata not cleared");
    rejects([&] {state.mux_matrix();});rejects([&] {state.next_indices(1,true);});
    rejects([&] {state.mux(noncontiguous);});rejects([&] {state.demux(noncontiguous);});
    rejects([&] {state.add_objects({0},std::vector<int64_t>{100},true);});
    // Object counts greater than a single bucket or the 200 detection queries
    // remain supported by the controller; there is no added object-count cap.
    for (const int64_t count:{1,37,257}) {
      const auto many=sam3::MultiplexController().get_state(count,device,dtype,true);
      TORCH_CHECK(many.bucket_count()==(count+15)/16,"controller silently capped objects");
      const auto x=at::randn({count,3,7},options).transpose(1,2);
      require_equal(many.demux(many.mux(x)),x);
      TORCH_CHECK(many.valid_object_mask().sum().item<int64_t>()==count,"valid-object mask mismatch");
    }
    rejects([&] {State({{0,0}},device,dtype,2);});
    rejects([&] {State({{1,-1}},device,dtype,2);});
    rejects([&] {State({{0,1}},device,dtype,1);});
    rejects([&] {sam3::MultiplexController(16,false,17);});
    std::cout<<"{\"device\":\""<<device<<"\",\"mode\":\""<<mode<<"\",\"state_safety\":true,\"roundtrip\":true,\"max_objects_tested\":257}\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
