#pragma once
#include "sam3/multiplex_history.h"
#include "sam3/tracking_session.h"
namespace sam3 {
struct MultiplexTrackingFeatures {TrackingFeatures interactive,propagation;};
struct MultiplexSessionObject {
  int64_t id;
  OrderedFrames<TrackingPoints> points;
  OrderedFrames<at::Tensor> masks,video_edits; // video_edits contains pending previews only
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
  // all_edits_conditioning controls mask/detector corrections. Point interaction
  // establishes conditioning, and repeated clicks refresh that conditioning.
  bool offload_state=false,non_overlap_output=true,all_edits_conditioning=true;
  bool always_start_at_first_annotation=false;
  int64_t fill_hole_area=0;
  // Optional lossless disk paging; caller owns the parent directory.
  // Archives are temporary and live as long as the referencing frame/state.
  std::filesystem::path history_directory;
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
  // Correct existing IDs in a frame already held by this session. The bucket
  // layout is unchanged; memory is rebuilt by preflight after the edits.
  TrackingSessionOutput recondition_masks(int64_t frame,const std::vector<int64_t>& objects,const at::Tensor& masks);
  void update_memory(int64_t frame,const at::Tensor& high_masks,const at::Tensor& proxy_logits,
      bool reapply_no_object_pointer=false);
  void preflight(bool encode_memory=true);
  void propagate(const TrackingPropagation&,const OutputCallback&);
  void cancel() noexcept {cancelled_.store(true);}
  TrackingSessionOutput clear_input(int64_t frame,int64_t object);
  void remove_object(int64_t object,bool strict=false);
  // Deduplicated batch removal remaps/re-encodes each affected history once.
  void remove_objects(const std::vector<int64_t>& objects,bool strict=false);
  // Move one object to a fresh singleton layout. Dense spatial history is
  // reconstructed from retained image/mask inputs, never demuxed as slot data.
  // Both sessions share their immutable core/provider. Historical output is
  // retained, while tracked-direction flags restart for first point refinement.
  std::unique_ptr<Sam31TrackingSession> extract_object(int64_t object);
  // High-level SAM3.1 point edits remove mask-only input annotations without
  // deleting already encoded history, matching the source consolidation repair.
  void discard_mask_only_inputs();
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
  void merge_edit(int64_t,size_t,const MultiplexFrame&,const at::Tensor& video,bool point_edit=false);
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
