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
// SAM3.1 first refinement extracts grouped objects into a singleton. Retained
// dense history is re-encoded, including memory upstream's slot demux loses.
struct Sam31VideoEditOptions : VideoEditOptions {
  Sam31VideoEditOptions(){cleanup_area=0;}
  bool clear_old_points=true;
  // Optional retained original image mask when all point prompts are cleared.
  // This restores an annotation and keeps it usable for later edits.
  at::Tensor empty_points_mask;
};
SAM3_NATIVE_EXPORT VideoOutput edit_video_points(int64_t frame,int64_t id,
    const TrackingPoints&,Sam31VideoSessions&,const Sam31SessionFactory&,
    VideoMetadata&,VideoInteractionState&,VideoSuppressionHistory&,
    const Sam31VideoEditOptions& options={});
SAM3_NATIVE_EXPORT VideoOutput edit_video_mask(int64_t frame,int64_t id,
    const at::Tensor& mask,Sam31VideoSessions&,const Sam31SessionFactory&,
    VideoMetadata&,VideoInteractionState&,VideoSuppressionHistory&,
    const Sam31VideoEditOptions& options={});
SAM3_NATIVE_EXPORT void remove_video_user_object(int64_t id,Sam31VideoSessions&,
    VideoMetadata&,VideoInteractionState&,bool record_action=true);
// Coordinated edits return the selected mask to output_device before merging
// with displayed masks. Existing overloads retain local-device behavior.
SAM3_NATIVE_EXPORT VideoOutput edit_video_points(int64_t frame,int64_t id,const TrackingPoints& points,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options,at::Device target);
SAM3_NATIVE_EXPORT VideoOutput edit_video_mask(int64_t frame,int64_t id,const at::Tensor& mask,Sam3VideoSessions& sessions,const Sam3SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const VideoEditOptions& options,at::Device target);
SAM3_NATIVE_EXPORT VideoOutput edit_video_points(int64_t frame,int64_t id,const TrackingPoints& points,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options,at::Device target);
SAM3_NATIVE_EXPORT VideoOutput edit_video_mask(int64_t frame,int64_t id,const at::Tensor& mask,Sam31VideoSessions& sessions,const Sam31SessionFactory& factory,VideoMetadata& metadata,VideoInteractionState& interaction,VideoSuppressionHistory& suppressions,const Sam31VideoEditOptions& options,at::Device target);
}
