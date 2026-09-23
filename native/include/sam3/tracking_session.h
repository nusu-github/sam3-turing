#pragma once
#include "sam3/tracking_frame.h"
#include "sam3/ordered_frames.h"
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <set>
namespace sam3 {
struct TrackingPoints { at::Tensor points,labels,box;bool normalized=true; };
struct TrackingEdit { TrackingFrame frame;at::Tensor video_mask; };
struct TrackingObjectState {
  int64_t id;
  OrderedFrames<TrackingPoints> points;
  OrderedFrames<at::Tensor> masks;
  TrackingHistory history;
  OrderedFrames<TrackingEdit> pending_conditioning,pending_tracked;
};
struct TrackingSessionState {
  std::vector<TrackingObjectState> objects;
  TrackingHistory history;
  std::set<int64_t> consolidated_conditioning,consolidated_tracked;
  std::map<int64_t,bool> tracked_direction;
  std::optional<int64_t> first_annotation;
  bool started=false;
};
struct TrackingSessionOptions {
  TrackingFrameOptions frame;
  bool offload_state=false,non_overlap_output=false,clear_near_input=true,clear_near_multi_object=false;
  bool all_edits_conditioning=true,always_start_at_first_annotation=false;
  int64_t max_points=0,fill_hole_area=0; // zero retains all supplied points
};
struct TrackingSessionOutput {
  int64_t index;
  std::vector<int64_t> object_ids;
  at::Tensor low_masks,masks,object_logits;
};
struct TrackingPropagation {
  std::optional<int64_t> start,max_steps;
  bool reverse=false,encode_memory=true,preflight=true;
};
SAM3_NATIVE_EXPORT at::Tensor postprocess_tracking_masks(const at::Tensor&,int64_t height,int64_t width,
    bool non_overlap=false,int64_t hole_area=0);
// SAM3's low-level interactive tracker session. A provider returns one cached
// image's projected TrackingFeatures; the session expands it across objects.
// The shared frame core can serve multiple sessions without copying weights.
class SAM3_NATIVE_EXPORT Sam3TrackingSession {
 public:
  using FeatureProvider=std::function<TrackingFeatures(int64_t)>;
  using OutputCallback=std::function<bool(const TrackingSessionOutput&)>;
  Sam3TrackingSession(std::shared_ptr<const Sam3TrackingFrame>,FeatureProvider,
      int64_t frames,int64_t height,int64_t width,at::Device device=at::kCPU,
      const std::string& mode="fp32",const TrackingSessionOptions& options={});
  TrackingSessionOutput add_points(int64_t frame,int64_t object,const TrackingPoints&,
      bool clear_old=true,bool use_previous_memory=false);
  TrackingSessionOutput add_mask(int64_t frame,int64_t object,const at::Tensor& mask);
  // Replace memory for existing current outputs without changing predicted
  // masks/scores/pointers. Missing frames are a no-op, matching the source host.
  void update_memory(int64_t frame,const at::Tensor& high_masks,const at::Tensor& proxy_logits);
  void preflight(bool encode_memory=true);
  // Callback false or cancel() stops after a consistent completed frame.
  void propagate(const TrackingPropagation&,const OutputCallback&);
  void cancel() noexcept {cancelled_.store(true);}
  TrackingSessionOutput clear_input(int64_t frame,int64_t object);
  std::vector<TrackingSessionOutput> remove_object(int64_t object,bool strict=false);
  void reset();
  std::vector<int64_t> object_ids() const;
  const TrackingSessionState& state() const {return state_;}
 private:
  void check_frame(int64_t) const;
  size_t object_index(int64_t,bool create=true);
  TrackingFeatures features(int64_t,int64_t);
  at::Tensor cache_position(const at::Tensor&);
  TrackingFrame run(int64_t,int64_t,TrackingFrameRequest,TrackingHistory&);
  TrackingEdit consolidate(int64_t,bool conditioning,bool encode,bool video_resolution);
  void split(const TrackingFrame&,bool conditioning);
  void clear_near(int64_t);
  void clear_input_impl(int64_t,size_t);
  void reset_results();
  TrackingSessionOutput output(const TrackingEdit&,bool propagation=false) const;
  std::shared_ptr<const Sam3TrackingFrame> core_;
  FeatureProvider provider_;
  int64_t frames_,height_,width_,cached_index_=-1;
  at::Device device_,storage_;
  std::string mode_;
  TrackingSessionOptions options_;
  TrackingSessionState state_;
  TrackingFeatures cached_;
  at::Tensor position_cache_;
  std::atomic<bool> cancelled_{false};
};
}
