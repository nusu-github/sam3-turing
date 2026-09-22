#include "sam3/multiplex.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <numeric>
#include <set>
namespace sam3 {
MultiplexState::MultiplexState(Assignments assignments,at::Device device,at::ScalarType dtype,int64_t capacity,
    std::optional<std::vector<int64_t>> ids):device_(device),dtype_(dtype),capacity_(capacity) {
  initialize(std::move(assignments),std::move(ids));
}
void MultiplexState::initialize(Assignments assignments,std::optional<std::vector<int64_t>> ids) {
  c10::InferenceMode inference;
  TORCH_CHECK(!assignments.empty() && !assignments.front().empty(),"multiplex state needs nonempty buckets");
  const auto width=static_cast<int64_t>(assignments.front().size());
  TORCH_CHECK(capacity_>=1 && capacity_<=width,"invalid multiplex bucket capacity");
  int64_t objects=0,occupied=0;std::set<int64_t> indices;
  for (const auto& bucket:assignments) {
    TORCH_CHECK(static_cast<int64_t>(bucket.size())==width,"multiplex buckets must have equal width");
    int64_t used=0;
    for (const auto index:bucket) {
      TORCH_CHECK(index>=0 || index==padding || index==removed,"unknown multiplex slot marker");
      if (index!=padding) ++used;
      if (index>=0) {++objects;TORCH_CHECK(indices.insert(index).second,"duplicate multiplex object index");}
    }
    TORCH_CHECK(used<=capacity_,"multiplex bucket capacity exceeded");occupied+=used;
  }
  int64_t expected=0;for (const auto index:indices) TORCH_CHECK(index==expected++,"multiplex indices must be dense from zero");
  TORCH_CHECK(!ids || static_cast<int64_t>(ids->size())==objects,"object IDs must match valid object count");
  TORCH_CHECK(at::isFloatingType(dtype_),"multiplex matrices require a floating dtype");
  // Build exact 0/1 matrices on CPU, then transfer once instead of issuing a
  // device write for every object. Keep both matrices contiguous like source.
  auto matrix=at::zeros({static_cast<int64_t>(assignments.size())*width,objects},at::TensorOptions().dtype(at::kFloat));
  auto access=matrix.accessor<float,2>();
  for (size_t i=0;i<assignments.size();++i)
    for (int64_t j=0;j<width;++j) if (assignments[i][j]>=0) access[i*width+j][assignments[i][j]]=1.f;
  // Allocate the transposed shape explicitly: contiguous() preserves unusual
  // singleton strides, whereas the source allocates both matrices separately.
  auto inverse=at::empty({objects,matrix.size(0)},matrix.options());
  inverse.copy_(matrix.transpose(0,1));
  auto mux=matrix.to(device_,dtype_),demux=inverse.to(device_,dtype_);
  assignments_=std::move(assignments);object_ids_=std::move(ids);width_=width;objects_=objects;occupied_=occupied;
  // Resolve a caller's unindexed "cuda" device to the allocation's actual GPU.
  device_=mux.device();mux_=std::move(mux);demux_=std::move(demux);valid_=true;
}
void MultiplexState::require_valid() const {TORCH_CHECK(valid_,"multiplex state was invalidated by removing all objects");}
int64_t MultiplexState::available_slots() const {require_valid();return bucket_count()*capacity_-occupied_;}
const at::Tensor& MultiplexState::mux_matrix() const {require_valid();return mux_;}
const at::Tensor& MultiplexState::demux_matrix() const {require_valid();return demux_;}
std::vector<int64_t> MultiplexState::next_indices(int64_t count,bool allow_new,bool /*prefer_new*/) const {
  require_valid();TORCH_CHECK(count>0,"new object count must be positive");
  TORCH_CHECK(allow_new || count<=available_slots(),"insufficient slots without new buckets");
  std::vector<int64_t> result(count);std::iota(result.begin(),result.end(),objects_);return result;
}
void MultiplexState::add_objects(const std::vector<int64_t>& indices,std::optional<std::vector<int64_t>> ids,bool allow_new,bool prefer_new) {
  if (indices.empty()) return;
  require_valid();TORCH_CHECK(!prefer_new || allow_new,"prefer_new_buckets requires allow_new_buckets");
  TORCH_CHECK(ids.has_value()==object_ids_.has_value(),"object IDs must always be supplied or always omitted");
  TORCH_CHECK(!ids || ids->size()==indices.size(),"new object ID count mismatch");
  for (size_t i=0;i<indices.size();++i) TORCH_CHECK(indices[i]==objects_+static_cast<int64_t>(i),"new indices must be consecutive and follow current objects");
  auto assignments=assignments_;auto all_ids=object_ids_;size_t next=0;
  const auto pop=[&]() {if (all_ids) all_ids->push_back((*ids)[next]);return indices[next++];};
  if (!prefer_new) {
    for (auto& bucket:assignments) {
      for (int64_t slot=0;slot<capacity_ && next<indices.size();++slot) if (bucket[slot]==padding) bucket[slot]=pop();
      if (next==indices.size()) break;
    }
  }
  TORCH_CHECK(next==indices.size() || allow_new,"cannot place objects without creating new buckets");
  while (next<indices.size()) {
    std::vector<int64_t> bucket(width_,padding);
    for (int64_t slot=0;slot<capacity_ && next<indices.size();++slot) bucket[slot]=pop();
    assignments.push_back(std::move(bucket));
  }
  // Commit only after validation/allocation succeeds; errors leave state intact.
  initialize(std::move(assignments),std::move(all_ids));
}
std::vector<int64_t> MultiplexState::remove_objects(const std::vector<int64_t>& indices,bool strict) {
  require_valid();auto assignments=assignments_;auto pending=indices;
  for (auto& bucket:assignments) for (auto& index:bucket) {
    const auto found=std::find(pending.begin(),pending.end(),index);
    if (found!=pending.end()) {index=removed;pending.erase(found);}
  }
  TORCH_CHECK(!strict || pending.empty(),"requested object index was not found");
  std::vector<int64_t> kept;Assignments remaining;std::set<int64_t> live;
  for (size_t i=0;i<assignments.size();++i) {
    const bool any=std::any_of(assignments[i].begin(),assignments[i].end(),[](int64_t index){return index>=0;});
    if (!any) continue;
    kept.push_back(i);remaining.push_back(std::move(assignments[i]));
    for (const auto index:remaining.back()) if (index>=0) live.insert(index);
  }
  if (kept.empty()) {
    assignments_.clear();if (object_ids_) object_ids_->clear();objects_=occupied_=0;valid_=false;
    mux_=at::Tensor();demux_=at::Tensor();return kept;
  }
  const std::vector<int64_t> old_indices(live.begin(),live.end());
  std::optional<std::vector<int64_t>> ids;
  if (object_ids_) {ids.emplace();for (const auto index:old_indices) ids->push_back((*object_ids_)[index]);}
  for (auto& bucket:remaining) for (auto& index:bucket)
    if (index>=0) index=std::lower_bound(old_indices.begin(),old_indices.end(),index)-old_indices.begin();
  initialize(std::move(remaining),std::move(ids));return kept;
}
at::Tensor MultiplexState::mux(const at::Tensor& x) const {
  c10::InferenceMode inference;require_valid();
  TORCH_CHECK(x.dim()>=1 && x.size(0)==objects_ && objects_>0 && x.device()==device_,"invalid multiplex input shape/device");
  auto shape=x.sizes().vec();shape.erase(shape.begin());shape.insert(shape.begin(),{bucket_count(),width_});
  return at::matmul(mux_,x.reshape({objects_,-1})).view(shape);
}
at::Tensor MultiplexState::demux(const at::Tensor& x) const {
  c10::InferenceMode inference;require_valid();
  TORCH_CHECK(x.dim()>=2 && x.size(0)==bucket_count() && x.size(1)==width_ && x.device()==device_,"invalid demultiplex input shape/device");
  auto shape=x.sizes().vec();shape.erase(shape.begin(),shape.begin()+2);shape.insert(shape.begin(),objects_);
  return at::matmul(demux_,x.reshape({bucket_count()*width_,-1})).view(shape);
}
at::Tensor MultiplexState::valid_object_mask() const {
  c10::InferenceMode inference;require_valid();return (mux_.sum(1)>0).reshape({bucket_count(),width_});
}
MultiplexController::MultiplexController(int64_t width,bool full_shuffle,int64_t capacity)
    :width_(width),capacity_(capacity<0?width:capacity),full_shuffle_(full_shuffle) {
  TORCH_CHECK(width_>0 && capacity_>0 && capacity_<=width_,"invalid controller capacity");
}
MultiplexState MultiplexController::get_state(int64_t objects,at::Device device,at::ScalarType dtype,bool random,
    std::optional<std::vector<int64_t>> object_ids) const {
  c10::InferenceMode inference;TORCH_CHECK(objects>0,"controller needs at least one object");
  const auto buckets=(objects+capacity_-1)/capacity_;const auto options=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  at::Tensor ids;
  if (full_shuffle_) {
    ids=at::cat({at::arange(objects,options),at::full({buckets*width_-objects},MultiplexState::padding,options)});
    if (random) ids=ids.index_select(0,at::randperm(ids.size(0),options));
  } else {
    ids=random?at::randperm(objects,options):at::arange(objects,options);
    if (ids.size(0)<buckets*capacity_) ids=at::cat({ids,at::full({buckets*capacity_-objects},MultiplexState::padding,options)});
  }
  MultiplexState::Assignments assignments(buckets,std::vector<int64_t>(width_,MultiplexState::padding));
  const auto* values=ids.const_data_ptr<int64_t>();const auto filled_width=full_shuffle_?width_:capacity_;
  for (int64_t i=0;i<buckets;++i) for (int64_t j=0;j<filled_width;++j) assignments[i][j]=values[i*filled_width+j];
  return MultiplexState(std::move(assignments),device,dtype,capacity_,std::move(object_ids));
}
}
