#pragma once
#include "sam3/memory_attention.h"
#include <optional>
namespace sam3 {
// Vectors preserve Python dict insertion order, including equal-distance ties.
struct TemporalFrame {
  int64_t index;
  at::Tensor features,position,pointer;
  at::Tensor effective_iou; // optional scalar; retain dtype for threshold rounding
};
struct TemporalState { std::vector<TemporalFrame> conditioning,tracked; };
struct TemporalOptions {
  int64_t memory_slots=7,max_conditioning_frames=4,max_pointer_frames=16,stride=1;
  bool keep_first=false,select_by_score=false;
  double score_threshold=.01;
};
struct TemporalReference { int64_t frame,position;bool conditioning; };
struct TemporalPlan {
  std::vector<int64_t> selected_conditioning,unselected_conditioning,filtered_frames;
  std::vector<TemporalReference> spatial,pointers;
};
SAM3_NATIVE_EXPORT std::pair<std::vector<int64_t>,std::vector<int64_t>> select_conditioning_frames(
    int64_t current,const std::vector<int64_t>& insertion_order,int64_t limit,bool keep_first=false);
SAM3_NATIVE_EXPORT TemporalPlan plan_sam3_memory(int64_t frame,int64_t frame_count,bool reverse,
    const TemporalState&,const TemporalOptions& options={});
SAM3_NATIVE_EXPORT at::Tensor memory_confidence(const at::Tensor& object_logits,const at::Tensor& iou);
struct TemporalAssembly {
  at::Tensor memory,position;
  int64_t pointer_tokens=0;
  TemporalPlan plan;
};
class SAM3_NATIVE_EXPORT Sam3MemoryConditioner {
 public:
  Sam3MemoryConditioner(const WeightStore&,at::Device device=at::kCPU);
  TemporalAssembly assemble(int64_t frame,int64_t frame_count,bool reverse,const TemporalState&,
      const TemporalOptions& options={},const std::string& mode="fp32") const;
  // Source and source_position: [H*W,B,256]. Default model grid is 72x72.
  at::Tensor forward(const at::Tensor& source,const at::Tensor& source_position,int64_t height,int64_t width,
      int64_t frame,int64_t frame_count,bool initial,bool reverse,bool use_previous,const TemporalState&,
      const TemporalOptions& options={},const std::string& mode="fp32",TemporalAssembly* trace=nullptr) const;
 private:
  MemoryAttention attention_;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
}
