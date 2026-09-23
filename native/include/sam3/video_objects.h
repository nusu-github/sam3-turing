#pragma once
#include "sam3/multiplex_session.h"
namespace sam3 {
enum class VideoObjectPlacement { BestFit, FirstState, NewState };
// Returns -1 when a new session is needed. Stable ties keep the first state.
// Placement changes grouping, never limits the number of accepted objects.
SAM3_NATIVE_EXPORT int64_t video_object_destination(const std::vector<int64_t>& available_slots,
    int64_t object_count,VideoObjectPlacement placement=VideoObjectPlacement::BestFit);
SAM3_NATIVE_EXPORT at::Tensor prepare_video_object_masks(const at::Tensor& logits);
using Sam3VideoSessions=std::vector<std::unique_ptr<Sam3TrackingSession>>;
using Sam31VideoSessions=std::vector<std::unique_ptr<Sam31TrackingSession>>;
using Sam3SessionFactory=std::function<std::unique_ptr<Sam3TrackingSession>()>;
using Sam31SessionFactory=std::function<std::unique_ptr<Sam31TrackingSession>()>;
// Factories supply empty sessions sharing the caller's frame core and feature
// cache. IDs must be new across the local collection. Empty additions are no-ops.
// A newly created session is published only after successful preflight. Edits
// of existing sessions use their per-operation rollback, not a list transaction.
SAM3_NATIVE_EXPORT int64_t add_video_objects(int64_t frame,const std::vector<int64_t>& ids,
    const at::Tensor& logits,Sam3VideoSessions&,const Sam3SessionFactory&);
SAM3_NATIVE_EXPORT int64_t add_video_objects(int64_t frame,const std::vector<int64_t>& ids,
    const at::Tensor& logits,Sam31VideoSessions&,const Sam31SessionFactory&,
    VideoObjectPlacement placement=VideoObjectPlacement::BestFit);
// Unknown IDs are ignored. Empty sessions are destroyed after removal, releasing
// their temporary disk archives. Collection order is preserved.
SAM3_NATIVE_EXPORT void remove_video_objects(const std::vector<int64_t>& ids,Sam3VideoSessions&);
SAM3_NATIVE_EXPORT void remove_video_objects(const std::vector<int64_t>& ids,Sam31VideoSessions&);
}
