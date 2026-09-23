#pragma once
#include "sam3/video_output.h"
#include "sam3/video_update.h"
namespace sam3 {
enum class VideoActionType {Add,Remove,Refine,Full,Partial,Fetch,Cancel};
struct VideoAction {
  VideoActionType type=VideoActionType::Full;
  std::optional<int64_t> frame;
  std::optional<std::vector<int64_t>> ids;
};
struct VideoActionRoute {VideoActionType type;std::optional<std::vector<int64_t>> ids;};
SAM3_NATIVE_EXPORT VideoActionRoute route_video_actions(const std::vector<VideoAction>&,
    int64_t frame_count,AssociationPolicy policy=AssociationPolicy::Sam3);
SAM3_NATIVE_EXPORT bool video_object_was_refined(const std::vector<VideoAction>&,int64_t id);
// Source reverse propagation begins one frame BEFORE the supplied start.
struct VideoProcessingRange {int64_t first,end,step;bool empty;};
SAM3_NATIVE_EXPORT VideoProcessingRange video_processing_range(int64_t frame_count,
    const std::set<int64_t>& initialized,std::optional<int64_t> start={},
    std::optional<int64_t> max_steps={},bool reverse=false);
using RefinedVideoObjects=std::map<int64_t,std::pair<at::Tensor,at::Tensor>>; // score, low [H,W]
// Owns displayed-frame cache and action history; neural state/weights stay in the
// existing sessions. The caller resets neural/metadata/prompt state together
// when replacing a semantic prompt. No prompt is fixed by this class.
class SAM3_NATIVE_EXPORT VideoInteractionState {
 public:
  VideoInteractionState(AssociationPolicy,int64_t frames,int64_t height,int64_t width);
  void append(const VideoAction&);
  VideoActionRoute route(const std::vector<int64_t>& active_ids={},bool force_tracker=false)const;
  void record(int64_t frame,const VideoOutput&);
  VideoOutput fetch(int64_t frame,const VideoMetadata&,const std::set<int64_t>& suppressed={})const;
  VideoOutput merge_refined(int64_t frame,const RefinedVideoObjects&,VideoMetadata&,
      const std::set<int64_t>& suppressed={});
  void forget_object(int64_t id);
  void reset();
  int64_t frame_count()const{return frames_;}
  AssociationPolicy policy()const{return policy_;}
  const std::vector<VideoAction>& actions()const{return actions_;}
  const std::map<int64_t,std::map<int64_t,at::Tensor>>& cached_frames()const{return cache_;}
 private:
  void check_frame(int64_t)const;
  VideoRawOutput raw(int64_t,const VideoMetadata&,const std::set<int64_t>&)const;
  AssociationPolicy policy_;int64_t frames_,height_,width_;
  std::vector<VideoAction> actions_;
  std::map<int64_t,std::map<int64_t,at::Tensor>> cache_;
};
// Propagate every local session containing a requested ID, with memory encoding,
// and collect only requested IDs. No detector/association phase runs here.
// Scores deliberately remain raw tracker outputs (source partial-path behavior).
// Caller performs inter-rank exchange, if any, before merging into the cache.
SAM3_NATIVE_EXPORT RefinedVideoObjects propagate_video_refinements(int64_t frame,bool reverse,
    const std::vector<int64_t>& ids,const std::vector<Sam3TrackingSession*>&,
    int64_t cleanup_area=16,bool preflight=true);
SAM3_NATIVE_EXPORT RefinedVideoObjects propagate_video_refinements(int64_t frame,bool reverse,
    const std::vector<int64_t>& ids,const std::vector<Sam31TrackingSession*>&,
    int64_t cleanup_area=0,bool preflight=true);
}
