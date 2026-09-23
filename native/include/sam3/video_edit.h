#pragma once
#include "sam3/video_interaction.h"
namespace sam3 {
struct VideoEditOptions {
  int64_t rank=0,cleanup_area=16,conditioning_window=16,confirmation_threshold=3;
  bool use_previous_memory=false,stateless_refinement=false;
};
using VideoSuppressionHistory=std::map<int64_t,std::set<int64_t>>;
// SAM3 high-level instance edits. Factories share cores/features and supply empty
// sessions. Caller assigns a new object's rank before invoking its local edit;
// transport of the returned mask to other ranks remains the caller's job.
// Points replace previous points, matching SAM3's high-level source API.
SAM3_NATIVE_EXPORT VideoOutput edit_video_points(int64_t frame,int64_t id,
    const TrackingPoints&,Sam3VideoSessions&,const Sam3SessionFactory&,
    VideoMetadata&,VideoInteractionState&,VideoSuppressionHistory&,
    const VideoEditOptions& options={});
SAM3_NATIVE_EXPORT VideoOutput edit_video_mask(int64_t frame,int64_t id,
    const at::Tensor& mask,Sam3VideoSessions&,const Sam3SessionFactory&,
    VideoMetadata&,VideoInteractionState&,VideoSuppressionHistory&,
    const VideoEditOptions& options={});
SAM3_NATIVE_EXPORT void remove_video_user_object(int64_t id,Sam3VideoSessions&,
    VideoMetadata&,VideoInteractionState&,bool record_action=true);
}
