#pragma once
#include "sam3/preprocess.h"
#include "sam3/video_edit.h"
#include "sam3/video_frame.h"
#include <atomic>
namespace sam3 {
struct VideoPredictorOptions {
  AssociationPolicy model = AssociationPolicy::Sam3;
  std::string mode = "fp32";
  VideoDetectionOptions detection;
  VideoUpdateOptions update;
  TrackingSessionOptions sam3_session;
  MultiplexSessionOptions sam31_session;
  int64_t output_batch_size = 1;
  bool centers = false,
       image_only = false; // explicit image source, not a one-frame video
  double image_detection_threshold = .5; // SAM3.1 image-mode birth threshold
};
SAM3_NATIVE_EXPORT
    VideoPredictorOptions video_predictor_defaults(AssociationPolicy);
struct VideoSemanticPrompt {
  std::optional<std::string> text; // UTF-8; absent means box/visual prompting
  at::Tensor boxes_xywh,
      box_labels; // normalized [N,4], [N]; optional, no object cap
  at::Tensor visual_features,
      visual_padding; // optional caller-encoded exemplar tokens
};
struct VideoPredictorPropagation {
  std::optional<int64_t> start, max_steps;
  bool reverse = false, force_tracker = false;
};
// Optional immutable modules shared across owners on the same model/device.
// Missing modules are loaded from the supplied store. Callers must match the
// selected model/device; sessions retain shared ownership independently.
struct VideoPredictorModules {
  std::shared_ptr<const VisionEncoder> vision;
  std::shared_ptr<const GroundingDetector> detector;
  std::shared_ptr<const Sam3TrackingFrame> sam3;
  std::shared_ptr<const Sam31TrackingFrame> sam31;
};
// Owns one local video: prompts, shared modules/features, neural sessions,
// metadata, displayed-frame cache and action/output scheduling. The frame
// provider returns U8 RGB [3,H,W]; codecs and distributed transport are
// separate. Calls are exclusive except cancel(), which may be called from
// another thread.
class SAM3_NATIVE_EXPORT VideoPredictor {
public:
  using FrameProvider = std::function<at::Tensor(int64_t)>;
  using OutputCallback = std::function<bool(int64_t, const VideoOutput &)>;
  VideoPredictor(const WeightStore &, const std::filesystem::path &vocabulary,
                 FrameProvider, int64_t frames, int64_t height, int64_t width,
                 at::Device, const VideoPredictorOptions &);
  VideoPredictor(const WeightStore &, const std::filesystem::path &vocabulary,
                 FrameProvider, int64_t frames, int64_t height, int64_t width,
                 at::Device, const VideoPredictorOptions &,
                 const VideoPredictorModules &);
  ~VideoPredictor();
  VideoPredictor(VideoPredictor &&) noexcept;
  VideoPredictor &operator=(VideoPredictor &&) noexcept;
  // Semantic replacement resets observations/cache/actions while retaining
  // model modules. Validation precedes reset; execution is not transactional.
  VideoOutput add_prompt(int64_t frame, const VideoSemanticPrompt &);
  VideoOutput add_points(int64_t frame, int64_t id, const TrackingPoints &,
                         bool clear_old = true, bool use_previous = false,
                         bool stateless = false);
  VideoOutput add_mask(int64_t frame, int64_t id,
                       const at::Tensor &); // authoritative instance mask
  void remove_object(int64_t id);
  VideoOutput fetch(int64_t frame) const;
  void propagate(const VideoPredictorPropagation &, const OutputCallback &);
  void cancel() noexcept;
  void reset();
  // Configure before use or after reset. The constructor device remains the
  // coordinator for vision/detection/output. Each entry owns a tracking rank;
  // repeated devices share immutable cores and copied frame features. Rank
  // execution is synchronous. No on-disk weight variants are created.
  void set_tracking_devices(const std::vector<at::Device> &);
  std::vector<at::Device> tracking_devices() const;
  // Configure before first frame encoding, or after reset. Does not alter
  // image/video birth thresholds or model weights.
  void set_preprocess(VideoPreprocess);
  VideoPreprocess preprocess_policy() const;
  const VideoMetadata &metadata() const;
  const VideoInteractionState &interaction() const;
  int64_t visual_encodes() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace sam3
