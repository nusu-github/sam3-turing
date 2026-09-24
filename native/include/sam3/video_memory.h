#pragma once
#include "sam3/association.h"
#include "sam3/multiplex_session.h"
namespace sam3 {
struct VideoMemoryInputs {at::Tensor high_masks,object_logits;};
// Globally resolve severe shrinkage before assigning objects to local states.
// The source's threshold is .3; output masks may still overlap. SAM3 warmup
// skips suppression; SAM3.1 always applies it and preserves single-mask inputs.
SAM3_NATIVE_EXPORT VideoMemoryInputs prepare_video_memory(const at::Tensor& low_masks,
    AssociationPolicy policy,bool warmup_complete=true);
// Explicit IDs preserve each state's row order after dynamic insertion/removal.
// Global IDs must be unique and each local ID must occur globally.
SAM3_NATIVE_EXPORT std::vector<std::vector<int64_t>> video_memory_rows(
    const std::vector<int64_t>& global_ids,const std::vector<std::vector<int64_t>>& state_ids);
// The caller supplies globally ordered masks and IDs, including remote ranks
// when global overlap suppression is required. These functions only execute
// local sessions; they do not perform multi-GPU gathering/communication.
SAM3_NATIVE_EXPORT void update_video_memories(int64_t frame,const at::Tensor& low_masks,
    const std::vector<int64_t>& global_ids,const std::vector<Sam3TrackingSession*>&,
    bool warmup_complete=true);
SAM3_NATIVE_EXPORT void update_video_memories(int64_t frame,const at::Tensor& low_masks,
    const std::vector<int64_t>& global_ids,const std::vector<Sam31TrackingSession*>&,
    bool reapply_no_object_pointer=false);
}
