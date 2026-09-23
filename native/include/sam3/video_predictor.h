#pragma once
#include "sam3/video_frame.h"
#include "sam3/video_edit.h"
#include <atomic>
namespace sam3 {
struct VideoPredictorOptions {
  AssociationPolicy model=AssociationPolicy::Sam3;
  std::string mode="fp32";
  VideoDetectionOptions detection;
  VideoUpdateOptions update;
  TrackingSessionOptions sam3_session;
  MultiplexSessionOptions sam31_session;
  int64_t output_batch_size=1;
  bool centers=false;
};
SAM3_NATIVE_EXPORT VideoPredictorOptions video_predictor_defaults(AssociationPolicy);
struct VideoSemanticPrompt {
  std::optional<std::string> text; // UTF-8; absent means box/visual prompting
  at::Tensor boxes_xywh,box_labels; // normalized [N,4], [N]; optional, no object cap
  at::Tensor visual_features,visual_padding; // optional caller-encoded exemplar tokens
};
struct VideoPredictorPropagation {
  std::optional<int64_t> start,max_steps;
  bool reverse=false,force_tracker=false;
};
// Owns one local video: prompts, shared modules/features, neural sessions,
// metadata, displayed-frame cache and action/output scheduling. The frame
// provider returns U8 RGB [3,H,W]; codecs and distributed transport are separate.
// Calls are exclusive except cancel(), which may be called from another thread.
class SAM3_NATIVE_EXPORT VideoPredictor {
 public:
  using FrameProvider=std::function<at::Tensor(int64_t)>;
  using OutputCallback=std::function<bool(int64_t,const VideoOutput&)>;
  VideoPredictor(const WeightStore&,const std::filesystem::path& vocabulary,
      FrameProvider,int64_t frames,int64_t height,int64_t width,at::Device,
      const VideoPredictorOptions&);
  ~VideoPredictor();
  VideoPredictor(VideoPredictor&&) noexcept;
  VideoPredictor& operator=(VideoPredictor&&) noexcept;
  // Semantic replacement resets observations/cache/actions while retaining
  // model modules. Validation precedes reset; execution is not transactional.
  VideoOutput add_prompt(int64_t frame,const VideoSemanticPrompt&);
  VideoOutput add_points(int64_t frame,int64_t id,const TrackingPoints&,
      bool clear_old=true,bool use_previous=false,bool stateless=false);
  VideoOutput add_mask(int64_t frame,int64_t id,const at::Tensor&); // SAM3 instance masks
  void remove_object(int64_t id);
  VideoOutput fetch(int64_t frame)const;
  void propagate(const VideoPredictorPropagation&,const OutputCallback&);
  void cancel() noexcept;
  void reset();
  const VideoMetadata& metadata()const;
  const VideoInteractionState& interaction()const;
  int64_t visual_encodes()const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
