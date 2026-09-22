#pragma once
#include "sam3/temporal_memory.h"
#include "sam3/multiplex.h"
namespace sam3 {
struct MultiplexTemporalFrame {
  int64_t index;
  at::Tensor features,position,pointer,effective_iou,image,image_position;
};
struct MultiplexTemporalState { std::vector<MultiplexTemporalFrame> conditioning,tracked; };
struct MultiplexTemporalOptions:TemporalOptions {
  bool only_past_pointers=false,signed_pointer_time=false;
  bool temporal_v2=true,encode_pointer_time=true,use_pointers=true;
};
struct MultiplexTemporalAssembly {
  at::Tensor memory,position,image,image_position;
  int64_t pointer_tokens=0;
  bool fuse=false;
  TemporalPlan plan;
};
class SAM3_NATIVE_EXPORT MultiplexMemoryConditioner {
 public:
  MultiplexMemoryConditioner(const WeightStore&,at::Device device=at::kCPU);
  // State must already be aligned to the current bucket allocation. Legacy
  // 5D per-slot memories are demuxed and cached in state, as in the source.
  MultiplexTemporalAssembly assemble(int64_t frame,int64_t frame_count,bool reverse,
      MultiplexTemporalState&,const MultiplexState&,const MultiplexTemporalOptions& options={},
      const std::string& mode="fp32") const;
  // Shared image/position are [H*W,1,256]; bucket-batched input is also accepted.
  // Initial/bypass frames must use interactive/direct-mask heads instead, unless
  // memory is disabled. Cleared memory falls back to current image features.
  at::Tensor forward(const at::Tensor& source,const at::Tensor& source_position,int64_t height,int64_t width,
      int64_t frame,int64_t frame_count,bool initial,bool reverse,bool use_previous,
      MultiplexTemporalState&,const MultiplexState&,const MultiplexTemporalOptions& options={},
      const std::string& mode="fp32",MultiplexTemporalAssembly* trace=nullptr) const;
 private:
  MemoryAttention attention_;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
}
