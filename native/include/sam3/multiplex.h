#pragma once
#include "sam3/weights.h"
#include <optional>
namespace sam3 {
class SAM3_NATIVE_EXPORT MultiplexState {
 public:
  using Assignments=std::vector<std::vector<int64_t>>;
  static constexpr int64_t padding=-1;
  static constexpr int64_t removed=-1116;
  MultiplexState(Assignments,at::Device device,at::ScalarType dtype,int64_t capacity,
      std::optional<std::vector<int64_t>> object_ids=std::nullopt);
  bool valid() const {return valid_;}
  int64_t bucket_count() const {return assignments_.size();}
  int64_t width() const {return width_;}
  int64_t capacity() const {return capacity_;}
  int64_t object_count() const {return objects_;}
  int64_t occupied_count() const {return occupied_;}
  int64_t available_slots() const;
  const Assignments& assignments() const {return assignments_;}
  const std::optional<std::vector<int64_t>>& object_ids() const {return object_ids_;}
  const at::Tensor& mux_matrix() const;
  const at::Tensor& demux_matrix() const;
  std::vector<int64_t> next_indices(int64_t count,bool allow_new_buckets=false,bool prefer_new_buckets=false) const;
  void add_objects(const std::vector<int64_t>& indices,std::optional<std::vector<int64_t>> ids=std::nullopt,
      bool allow_new_buckets=false,bool prefer_new_buckets=false);
  std::vector<int64_t> remove_objects(const std::vector<int64_t>& indices,bool strict=true);
  at::Tensor mux(const at::Tensor&) const;
  at::Tensor demux(const at::Tensor&) const;
  at::Tensor valid_object_mask() const;
 private:
  void initialize(Assignments,std::optional<std::vector<int64_t>>);
  void require_valid() const;
  Assignments assignments_;
  std::optional<std::vector<int64_t>> object_ids_;
  at::Tensor mux_,demux_;
  at::Device device_;
  at::ScalarType dtype_;
  int64_t capacity_,width_=0,objects_=0,occupied_=0;
  bool valid_=false;
};
class SAM3_NATIVE_EXPORT MultiplexController {
 public:
  explicit MultiplexController(int64_t width=16,bool full_shuffle=false,int64_t eval_capacity=-1);
  // Uses the ATen CPU RNG, exactly as the original inference controller.
  MultiplexState get_state(int64_t objects,at::Device device,at::ScalarType dtype,bool random=true,
      std::optional<std::vector<int64_t>> object_ids=std::nullopt) const;
 private:
  int64_t width_,capacity_;
  bool full_shuffle_;
};
}
