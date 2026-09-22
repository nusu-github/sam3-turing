#pragma once
#include "sam3/multiplex_history.h"
#include "sam3/tracking_session.h"
namespace sam3 {
struct MultiplexTrackingFeatures {TrackingFeatures interactive,propagation;};
struct MultiplexSessionObject {
  int64_t id;
  OrderedFrames<TrackingPoints> points;
  OrderedFrames<at::Tensor> masks,video_edits;
  std::set<int64_t> refined;
};
struct MultiplexSessionState {
  std::optional<MultiplexState> buckets;
  std::vector<MultiplexSessionObject> objects;
  MultiplexFrameHistory history;
  std::set<int64_t> dirty,annotated;
  std::map<int64_t,bool> tracked_direction;
  std::optional<int64_t> first_annotation;
  bool started=false;
};
struct MultiplexSessionOptions {
  // Session retention overrides frame.offload_output/trim_history/save_image;
  // use offload_state below to retain reconstruction inputs in CPU memory.
  MultiplexFrameOptions frame;
  bool offload_state=false,non_overlap_output=true,all_edits_conditioning=true;
  bool always_start_at_first_annotation=false;
  int64_t fill_hole_area=0;
};
// Dynamic SAM3.1 sessions keep reconstruction inputs for future layout changes.
// One core and one visual provider can be shared; session state owns no weights.
class SAM3_NATIVE_EXPORT Sam31TrackingSession {
 public:
  using FeatureProvider=std::function<MultiplexTrackingFeatures(int64_t)>;
  using OutputCallback=std::function<bool(const TrackingSessionOutput&)>;
  Sam31TrackingSession(std::shared_ptr<const Sam31TrackingFrame>,FeatureProvider,
      int64_t frames,int64_t height,int64_t width,at::Device device=at::kCPU,
      const std::string& mode="fp32",const MultiplexSessionOptions& options={});
  TrackingSessionOutput add_points(int64_t frame,int64_t object,const TrackingPoints&,bool clear_old=true,bool use_previous_memory=false);
  TrackingSessionOutput add_mask(int64_t frame,int64_t object,const at::Tensor&);
  TrackingSessionOutput add_masks(int64_t frame,const std::vector<int64_t>& objects,const at::Tensor& masks);
  void preflight(bool encode_memory=true);
  void propagate(const TrackingPropagation&,const OutputCallback&);
  void cancel() noexcept {cancelled_.store(true);}
  TrackingSessionOutput clear_input(int64_t frame,int64_t object);
  void remove_object(int64_t object,bool strict=false);
  void reset();
  std::vector<int64_t> object_ids() const;
  const MultiplexSessionState& state() const {return state_;}
 private:
  void check_frame(int64_t) const;
  size_t ensure_object(int64_t,bool prefer_new);
  MultiplexTrackingFeatures features(int64_t);
  MultiplexFrame blank(int64_t);
  MultiplexFrame* find(int64_t);
  void store(MultiplexFrame&,bool compress=true);
  void put(MultiplexFrame,bool conditioning);
  void merge_edit(int64_t,size_t,const MultiplexFrame&,const at::Tensor& video);
  void classify_inputs();
  TrackingSessionOutput output(const MultiplexFrame&,bool preview=false) const;
  void remap(const MultiplexState& next);
  std::shared_ptr<const Sam31TrackingFrame> core_;
  FeatureProvider provider_;
  int64_t frames_,height_,width_,cached_index_=-1;
  at::Device device_,storage_;
  std::string mode_;
  MultiplexSessionOptions options_;
  MultiplexFrameOptions frame_options_;
  MultiplexSessionState state_;
  MultiplexTrackingFeatures cached_;
  std::atomic<bool> cancelled_{false};
};
}
