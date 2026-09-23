#include "sam3/video_predictor.h"
#include "rank_executor.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include "sam3/video_collective.h"
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <type_traits>
namespace sam3 {
namespace {
VideoOutput centers(VideoOutput out, int64_t h, int64_t w, bool enabled) {
  if (enabled && !out.centers.defined()) {
    const auto o = out.masks.options().dtype(at::kFloat);
    const auto y = at::arange(h, o).view({1, h, 1}),
               x = at::arange(w, o).view({1, 1, w});
    const auto mass = out.masks.sum(at::IntArrayRef{1, 2}).clamp_min(1e-6);
    out.centers =
        at::stack({(out.masks * x).sum(at::IntArrayRef{1, 2}) / mass / w,
                   (out.masks * y).sum(at::IntArrayRef{1, 2}) / mass / h},
                  1);
  }
  return out;
}
} // namespace
VideoPredictorOptions video_predictor_defaults(AssociationPolicy policy) {
  TORCH_CHECK(policy == AssociationPolicy::Sam3 ||
                  policy == AssociationPolicy::Sam31,
              "invalid video model");
  const bool mux = policy == AssociationPolicy::Sam31;
  VideoPredictorOptions o;
  o.model = policy;
  o.detection.model = policy;
  o.detection.nms = mux ? VideoNmsMode::Sam31Batched : VideoNmsMode::Greedy;
  o.detection.score_threshold = mux ? .4 : .5;
  o.detection.use_iom = mux;
  o.detection.boundary_filter = mux;
  auto &u = o.update;
  u.association.policy = u.recondition.policy = policy;
  u.association.new_detection_threshold = mux ? .65 : .7;
  u.association.use_iom = mux;
  u.association.iom_recondition_threshold = mux ? .5 : .8;
  u.hotstart.delay = 15;
  u.hotstart.unmatched_threshold = 8;
  u.hotstart.duplicate_threshold = 8;
  u.hotstart.suppress_only_within_hotstart = !mux;
  if (!mux) {
    u.hotstart.min_keep_alive = -1;
    u.hotstart.max_keep_alive = 30;
    u.hotstart.initial_keep_alive = 30;
  }
  u.recondition.period = 16;
  u.occlusion_threshold = .7;
  u.cleanup_area = mux ? 0 : 16;
  u.confirmation_enabled = mux;
  o.sam3_session.offload_state = true;
  o.sam3_session.frame.temporal.select_by_score = true;
  o.sam31_session.offload_state = true;
  o.sam31_session.all_edits_conditioning = false;
  o.output_batch_size = mux ? 16 : 1;
  return o;
}
struct VideoPredictor::Impl {
  WeightStore store;
  Tokenizer tokenizer;
  FrameProvider provider;
  int64_t frames, h, w, cached_index = -1, encodes = 0;
  at::Device device;
  VideoPredictorOptions options;
  bool mux;
  bool parallel_tracking = false;
  int64_t worker_frame = -1;
  std::shared_ptr<const VisionEncoder> vision;
  std::shared_ptr<const GroundingDetector> detector;
  std::shared_ptr<const Sam3TrackingFrame> core3;
  std::shared_ptr<const Sam31TrackingFrame> core31;
  std::unique_ptr<VideoFrameEncoder> encoder;
  struct Worker {
    at::Device device;
    std::shared_ptr<const Sam3TrackingFrame> core3;
    std::shared_ptr<const Sam31TrackingFrame> core31;
    MultiplexTrackingFeatures cached;
    int64_t cached_index = -1;
    explicit Worker(at::Device d) : device(d) {}
  };
  struct RankState {
    std::shared_ptr<Worker> worker;
    Sam3VideoSessions sessions3;
    Sam31VideoSessions sessions31;
    Sam3SessionFactory factory3;
    Sam31SessionFactory factory31;
  };
  std::vector<std::unique_ptr<RankState>> ranks;
  VideoFrameFeatures cached;
  VideoMetadata metadata;
  VideoInteractionState interaction;
  VideoOutputCacheStorage cache_storage = VideoOutputCacheStorage::Resident;
  std::unique_ptr<VideoMaskCache> mask_cache;
  bool cache_inspection_pinned = false;
  VideoSuppressionHistory suppressions;
  std::set<int64_t> initialized;
  std::optional<int64_t> prompt_frame;
  VideoPreprocess preprocess = VideoPreprocess::ImageFolder;
  bool preprocess_locked = false;
  std::map<int64_t, at::Tensor> image_masks;
  VideoSemanticPrompt semantic;
  GroundingPrompt encoded;
  bool has_text = false;
  std::atomic<bool> cancelled{false};
  Impl(const WeightStore &s, const std::filesystem::path &vocabulary,
       FrameProvider f, int64_t n, int64_t height, int64_t width, at::Device d,
       const VideoPredictorOptions &o, const VideoPredictorModules &modules)
      : store(s), tokenizer(vocabulary), provider(std::move(f)), frames(n),
        h(height), w(width),
        device(at::empty({0}, at::TensorOptions().device(d)).device()),
        options(o), mux(o.model == AssociationPolicy::Sam31),
        metadata(initialize_video_metadata(1, device)),
        interaction(o.model, n, height, width) {
    TORCH_CHECK(provider && n > 0 && h > 0 && w > 0 && o.output_batch_size > 0,
                "invalid video source/options");
    TORCH_CHECK(o.mode == "fp32" || o.mode == "fp16" ||
                    o.mode == "bf16_reference",
                "invalid neural mode");
    TORCH_CHECK(!o.image_only || n == 1,
                "an image source requires exactly one frame");
    TORCH_CHECK(std::isfinite(o.image_detection_threshold) &&
                    o.image_detection_threshold >= 0 &&
                    o.image_detection_threshold <= 1,
                "invalid image detection threshold");
    if (mux && o.image_only)
      options.update.association.new_detection_threshold =
          o.image_detection_threshold;
    TORCH_CHECK(o.detection.model == o.model &&
                    o.update.association.policy == o.model &&
                    o.update.recondition.policy == o.model,
                "video policies must match model");
    const auto model = mux ? "sam3.1" : "sam3";
    vision = modules.vision
                 ? modules.vision
                 : std::make_shared<VisionEncoder>(store, model, device);
    detector = modules.detector
                   ? modules.detector
                   : std::make_shared<GroundingDetector>(store, model, device);
    device = at::empty({0}, at::TensorOptions().device(device)).device();
    if (mux) {
      core31 = modules.sam31
                   ? modules.sam31
                   : std::make_shared<Sam31TrackingFrame>(store, device);
      encoder =
          std::make_unique<VideoFrameEncoder>(vision, detector, core31, device);
    } else {
      core3 = modules.sam3 ? modules.sam3
                           : std::make_shared<Sam3TrackingFrame>(store, device);
      encoder =
          std::make_unique<VideoFrameEncoder>(vision, detector, core3, device);
    }
    set_devices({device});
  }
  void set_devices(const std::vector<at::Device> &devices) {
    TORCH_CHECK(!preprocess_locked && initialized.empty() &&
                    metadata.object_ids().empty() &&
                    interaction.actions().empty(),
                "set tracking devices before use or after reset");
    TORCH_CHECK(!devices.empty(), "at least one tracking device is required");
    std::vector<at::Device> resolved;
    for (auto d : devices) {
      TORCH_CHECK(d.is_cpu() || d.is_cuda(),
                  "tracking devices must be CPU/CUDA");
      resolved.push_back(
          at::empty({0}, at::TensorOptions().device(d)).device());
    }
    std::map<std::string, std::shared_ptr<Worker>> workers;
    for (const auto &r : ranks)
      workers.emplace(r->worker->device.str(), r->worker);
    std::vector<std::unique_ptr<RankState>> next;
    for (auto d : resolved) {
      auto &worker = workers[d.str()];
      if (!worker) {
        worker = std::make_shared<Worker>(d);
        c10::DeviceGuard guard(d);
        if (mux)
          worker->core31 = d == device
                               ? core31
                               : std::make_shared<Sam31TrackingFrame>(store, d);
        else
          worker->core3 = d == device
                              ? core3
                              : std::make_shared<Sam3TrackingFrame>(store, d);
      }
      auto rank = std::make_unique<RankState>();
      rank->worker = worker;
      if (mux)
        rank->factory31 = [this, worker] {
          return std::make_unique<Sam31TrackingSession>(
              worker->core31,
              [this, worker](int64_t i) {
                return tracking_features(worker, i);
              },
              frames, h, w, worker->device, options.mode,
              options.sam31_session);
        };
      else
        rank->factory3 = [this, worker] {
          return std::make_unique<Sam3TrackingSession>(
              worker->core3,
              [this, worker](int64_t i) {
                return tracking_features(worker, i).propagation;
              },
              frames, h, w, worker->device, options.mode, options.sam3_session);
        };
      next.push_back(std::move(rank));
    }
    auto next_metadata = initialize_video_metadata(next.size(), device);
    ranks = std::move(next);
    metadata = std::move(next_metadata);
  }
  const MultiplexTrackingFeatures &
  tracking_features(const std::shared_ptr<Worker> &worker, int64_t frame) {
    if (worker_frame >= 0) {
      TORCH_CHECK(frame == worker_frame && cached_index == frame,
                  "rank requested features outside its prepared frame");
      if (worker->device == device)
        return cached.tracking;
      TORCH_CHECK(worker->cached_index == frame,
                  "rank features were not prepared");
      return worker->cached;
    }
    const auto &visual = features(frame);
    if (worker->device == device)
      return visual.tracking;
    if (worker->cached_index != frame) {
      const auto move = [&](const TrackingFeatures &f) {
        TrackingFeatures out;
        out.image = transfer_video_tensor(f.image, worker->device);
        out.position = transfer_video_tensor(f.position, worker->device);
        for (const auto &x : f.high)
          out.high.push_back(transfer_video_tensor(x, worker->device));
        return out;
      };
      worker->cached.interactive = move(visual.tracking.interactive);
      worker->cached.propagation =
          mux ? move(visual.tracking.propagation) : worker->cached.interactive;
      worker->cached_index = frame;
    }
    return worker->cached;
  }
  VideoRankExecution execution() const {
    return parallel_tracking ? VideoRankExecution::Parallel
                             : VideoRankExecution::Serial;
  }
  // Prepare every shared feature cache on the caller. Workers read only this
  // snapshot, never invoke the user's frame provider or the shared encoder.
  struct WorkerFrame {
    Impl &owner;
    WorkerFrame(Impl &s, int64_t frame) : owner(s) {
      if (!s.parallel_tracking || s.ranks.size() <= 1)
        return;
      TORCH_CHECK(s.worker_frame < 0, "nested rank dispatch");
      for (const auto &rank : s.ranks)
        s.tracking_features(rank->worker, frame);
      s.worker_frame = frame;
    }
    ~WorkerFrame() { owner.worker_frame = -1; }
  };
  size_t rank_for(int64_t id) const {
    for (size_t r = 0; r < ranks.size(); ++r)
      for (auto existing : metadata.ids_per_rank[r])
        if (existing == id)
          return r;
    size_t best = 0;
    for (size_t r = 1; r < ranks.size(); ++r)
      if (metadata.ids_per_rank[r].size() < metadata.ids_per_rank[best].size())
        best = r;
    return best;
  }
  template <class Rank> std::vector<Rank> descriptors() {
    std::vector<Rank> out;
    for (auto &r : ranks) {
      if constexpr (std::is_same_v<Rank, Sam3VideoRank>)
        out.push_back({r->worker->device, &r->sessions3, &r->factory3});
      else
        out.push_back({r->worker->device, &r->sessions31, &r->factory31});
    }
    return out;
  }
  void check(int64_t frame) const {
    TORCH_CHECK(frame >= 0 && frame < frames, "frame outside video");
  }
  const VideoFrameFeatures &features(int64_t frame) {
    c10::DeviceGuard guard(device);
    check(frame);
    if (cached_index != frame) {
      auto rgb = provider(frame);
      TORCH_CHECK(rgb.sizes() == at::IntArrayRef({3, h, w}) &&
                      rgb.scalar_type() == at::kByte,
                  "frame provider must return U8 RGB [3,H,W]");
      preprocess_locked = true;
      auto next = encoder->encode_preprocessed(
          preprocess_video_rgb(rgb, preprocess, device), options.mode);
      cached = std::move(next);
      cached_index = frame;
      for (auto &r : ranks) {
        r->worker->cached = {};
        r->worker->cached_index = -1;
      }
      ++encodes;
    }
    return cached;
  }
  GeometryPrompt empty_geometry() const {
    const auto o = at::TensorOptions().device(device);
    const auto labels = at::empty({0, 1}, o.dtype(at::kLong)),
               padding = at::empty({1, 0}, o.dtype(at::kBool));
    return {at::empty({0, 1, 2}, o), labels, padding,
            at::empty({0, 1, 4}, o), labels, padding};
  }
  void encode_text() {
    c10::DeviceGuard guard(device);
    const auto model = mux ? "sam3.1" : "sam3";
    const auto text = has_text ? *semantic.text : "<text placeholder>";
    const auto texts =
        mux ? std::vector<std::string>{text, "visual", "geometric"}
            : std::vector<std::string>{text, "visual"};
    std::vector<at::Tensor> rows;
    const auto o = at::TensorOptions().device(device);
    for (const auto &ids : tokenizer.tokenize(texts))
      rows.push_back(at::tensor(ids, o.dtype(at::kLong)));
    // Text weights are temporary; retained token features are small. Trunk and
    // tracker weights remain shared across semantic replacements and sessions.
    TextEncoder text_encoder(store, model, device);
    const auto result = text_encoder.forward(at::stack(rows), options.mode);
    encoded.text_padding = std::get<0>(result);
    encoded.text_features = std::get<1>(result);
    encoded.image_ids = at::zeros({1}, o.dtype(at::kLong));
    encoded.text_ids =
        at::full({1}, !mux && !has_text ? 1 : 0, o.dtype(at::kLong));
  }
  GroundingPrompt prompt(int64_t frame) {
    if (!encoded.text_features.defined())
      encode_text();
    auto result = encoded;
    result.geometry = empty_geometry();
    if (prompt_frame == frame && semantic.boxes_xywh.defined()) {
      auto boxes = semantic.boxes_xywh.to(device, at::kFloat).clone();
      boxes.slice(1, 0, 2).add_(boxes.slice(1, 2, 4) * .5);
      result.geometry.boxes = boxes.unsqueeze(1);
      result.geometry.box_labels =
          semantic.box_labels.to(device, at::kLong).unsqueeze(1);
      result.geometry.box_padding =
          at::zeros({1, boxes.size(0)},
                    at::TensorOptions().device(device).dtype(at::kBool));
    }
    result.visual_features = semantic.visual_features;
    result.visual_padding = semantic.visual_padding;
    return result;
  }
  void reset() {
    preprocess_locked = false;
    for (auto &r : ranks) {
      r->sessions3.clear();
      r->sessions31.clear();
      r->worker->cached = {};
      r->worker->cached_index = -1;
    }
    metadata = initialize_video_metadata(ranks.size(), device);
    interaction.reset();
    if (mask_cache) mask_cache->clear();
    cache_inspection_pinned = false;
    suppressions.clear();
    initialized.clear();
    prompt_frame.reset();
    image_masks.clear();
    semantic = {};
    encoded = {};
    has_text = false;
    cached = {};
    cached_index = -1;
    cancelled.store(false);
  }
  void cache_raw(const VideoRawOutput &raw) {
    VideoOutput cache;
    cache.cached_masks = raw.masks;
    interaction.record(raw.frame, cache);
    compact_cache(raw.frame);
  }
  template <class Rank>
  VideoRawOutput full_impl(int64_t frame, bool reverse, bool direct) {
    c10::DeviceGuard guard(device);
    const auto &visual = features(frame);
    auto raw = encoder->detect(visual, prompt(frame), options.mode);
    auto detection = options.detection;
    if (mux && direct)
      detection.nms = VideoNmsMode::Sam31Perflib;
    detection.allow_new_detections =
        mux || has_text ||
        (prompt_frame == frame && semantic.boxes_xywh.defined()) ||
        semantic.visual_features.defined();
    auto detections = postprocess_video_detections(raw.detection, detection);
    TORCH_CHECK(detections.size() == 1,
                "one semantic prompt belongs to this video session");
    auto &d = detections.front();
    const auto local_ranks = descriptors<Rank>();
    const WorkerFrame prepared(*this, frame);
    auto tracking = propagate_video_tracking_ranks(
        frame, reverse, local_ranks, metadata, device,
        options.update.cleanup_area, execution());
    const auto &low = tracking.masks;
    const auto &logits = tracking.logits;
    auto plan = plan_video_update(frame, reverse, d, low, logits, metadata,
                                  options.update);
    execute_video_update_ranks(frame, plan, d, local_ranks, options.update,
                               execution());
    auto masks_out = build_video_outputs(plan, d, h, w, options.update);
    finalize_video_scores(plan.metadata, frame, plan.previous_ids, logits);
    metadata = std::move(plan.metadata);
    metadata.host.removed.insert(plan.removed.begin(), plan.removed.end());
    if (mux && options.image_only)
      for (const auto &rank : ranks)
        for (const auto &session : rank->sessions31)
          for (const auto &object : session->state().objects)
            if (!image_masks.count(object.id) && object.masks.count(frame))
              image_masks[object.id] =
                  object.masks.at(frame).squeeze(0).squeeze(0).cpu().clone();
    VideoRawOutput out;
    out.frame = frame;
    out.masks = std::move(masks_out);
    out.scores = metadata.object_scores;
    out.tracker_scores = metadata.frame_scores[frame];
    out.removed = metadata.host.removed;
    out.frame_stats = {
        {"num_obj_tracked", int64_t(metadata.object_ids().size())},
        {"num_obj_dropped", 0}};
    // Source SAM3.1 computes a GPU suppression candidate but does not publish
    // it to rank0 suppressed_obj_ids. Only the published host set filters
    // outputs.
    if (metadata.host.suppressed.count(frame))
      out.suppressed = metadata.host.suppressed.at(frame);
    out.unconfirmed = std::set<int64_t>{};
    if (options.update.confirmation_enabled) {
      const auto all = metadata.object_ids();
      for (size_t i = 0; i < all.size(); ++i)
        if (metadata.confirmation.status[i] == 1)
          out.unconfirmed->insert(all[i]);
    }
    suppressions[frame] = out.suppressed;
    initialized.insert(frame);
    cache_raw(out);
    return out;
  }
  VideoRawOutput full(int64_t frame, bool reverse, bool direct) {
    return mux ? full_impl<Sam31VideoRank>(frame, reverse, direct)
               : full_impl<Sam3VideoRank>(frame, reverse, direct);
  }
  void restore_cache(int64_t frame) {
    if (mask_cache && mask_cache->contains(frame)) {
      VideoOutput restored;
      restored.cached_masks = mask_cache->read(frame);
      interaction.record(frame, restored);
      mask_cache->erase(frame);
    }
  }
  void compact_cache(int64_t frame) {
    if (!mask_cache || cache_inspection_pinned) return;
    const auto found = interaction.cached_frames().find(frame);
    if (found == interaction.cached_frames().end()) return;
    try {
      mask_cache->store(frame, found->second);
    } catch (...) {
      // The resident frame is the latest complete output. An older packed
      // version must not replace it on the next fetch after a failed write.
      mask_cache->erase(frame);
      throw;
    }
    interaction.forget_frame(frame);
  }
  void compact_cache_all() {
    if (!mask_cache || cache_inspection_pinned) return;
    std::vector<int64_t> frames;
    for(const auto& [frame,_]:interaction.cached_frames())frames.push_back(frame);
    for(auto frame:frames)compact_cache(frame);
  }
  void ensure_cache(int64_t frame) {
    restore_cache(frame);
    if (!interaction.cached_frames().count(frame))
      interaction.record(frame, VideoOutput{});
  }
};
void VideoPredictor::set_tracking_devices(
    const std::vector<at::Device> &devices) {
  c10::InferenceMode inference;
  impl_->set_devices(devices);
}
std::vector<at::Device> VideoPredictor::tracking_devices() const {
  std::vector<at::Device> out;
  for (const auto &r : impl_->ranks)
    out.push_back(r->worker->device);
  return out;
}
void VideoPredictor::set_parallel_tracking(bool enabled) {
  TORCH_CHECK(!impl_->preprocess_locked && impl_->initialized.empty() &&
                  impl_->metadata.object_ids().empty() &&
                  impl_->interaction.actions().empty(),
              "set parallel tracking before use or after reset");
  impl_->parallel_tracking = enabled;
}
bool VideoPredictor::parallel_tracking() const {
  return impl_->parallel_tracking;
}
void VideoPredictor::set_output_cache(VideoOutputCacheStorage storage,
                                      const std::filesystem::path& directory) {
  auto& s=*impl_;
  TORCH_CHECK(!s.preprocess_locked && s.initialized.empty() &&
                  s.metadata.object_ids().empty() && s.interaction.actions().empty(),
              "set output cache before use or after reset");
  TORCH_CHECK(storage==VideoOutputCacheStorage::Resident ||
                  storage==VideoOutputCacheStorage::PackedCPU ||
                  storage==VideoOutputCacheStorage::PackedDisk,"invalid output cache storage");
  std::unique_ptr<VideoMaskCache> next;
  if(storage!=VideoOutputCacheStorage::Resident)
    next=std::make_unique<VideoMaskCache>(s.h,s.w,
        storage==VideoOutputCacheStorage::PackedCPU?VideoMaskStorage::PackedCPU:VideoMaskStorage::PackedDisk,
        directory);
  s.mask_cache=std::move(next);s.cache_storage=storage;s.cache_inspection_pinned=false;
}
VideoPredictorCacheStats VideoPredictor::output_cache_stats() const {
  const auto& s=*impl_;VideoPredictorCacheStats out;
  out.storage=s.cache_storage;out.inspection_pinned=s.cache_inspection_pinned;
  if(s.mask_cache) {
    const auto p=s.mask_cache->stats();out.frames=p.frames;out.masks=p.masks;
    out.packed_bytes=p.packed_bytes;out.disk_bytes=p.disk_bytes;
  }
  for(const auto& [_,masks]:s.interaction.cached_frames()) {
    ++out.frames;out.masks+=masks.size();
    for(const auto& [id,mask]:masks)out.resident_bytes+=mask.nbytes();
  }
  return out;
}
std::vector<int64_t> VideoPredictor::cached_frame_indices() const {
  const auto& s=*impl_;std::set<int64_t> frames;
  if(s.mask_cache)for(auto frame:s.mask_cache->frames())frames.insert(frame);
  for(const auto& [frame,_]:s.interaction.cached_frames())frames.insert(frame);
  return {frames.begin(),frames.end()};
}
int64_t VideoPredictor::action_count() const {return impl_->interaction.actions().size();}
void VideoPredictor::set_preprocess(VideoPreprocess policy) {
  TORCH_CHECK(!impl_->preprocess_locked,
              "set preprocessing before frame encoding or after reset");
  TORCH_CHECK(video_preprocess_available(policy),
              "video preprocessing policy unavailable");
  TORCH_CHECK(policy != VideoPreprocess::TorchCodecCuda ||
                  impl_->device.is_cuda(),
              "TorchCodecCuda requires a CUDA predictor");
  impl_->preprocess = policy;
}
VideoPreprocess VideoPredictor::preprocess_policy() const {
  return impl_->preprocess;
}
VideoPredictor::VideoPredictor(const WeightStore &s,
                               const std::filesystem::path &vocabulary,
                               FrameProvider provider, int64_t n, int64_t h,
                               int64_t w, at::Device d,
                               const VideoPredictorOptions &o)
    : VideoPredictor(s, vocabulary, std::move(provider), n, h, w, d, o,
                     VideoPredictorModules{}) {}
VideoPredictor::VideoPredictor(const WeightStore &s,
                               const std::filesystem::path &vocabulary,
                               FrameProvider provider, int64_t n, int64_t h,
                               int64_t w, at::Device d,
                               const VideoPredictorOptions &o,
                               const VideoPredictorModules &modules)
    : impl_(std::make_unique<Impl>(s, vocabulary, std::move(provider), n, h, w,
                                   d, o, modules)) {}
VideoPredictor::~VideoPredictor() = default;
VideoPredictor::VideoPredictor(VideoPredictor &&) noexcept = default;
VideoPredictor &VideoPredictor::operator=(VideoPredictor &&) noexcept = default;
VideoOutput VideoPredictor::add_prompt(int64_t frame,
                                       const VideoSemanticPrompt &p) {
  c10::InferenceMode inference;
  auto &s = *impl_;
  s.check(frame);
  TORCH_CHECK(p.text || p.boxes_xywh.defined() || p.visual_features.defined(),
              "semantic input requires text, boxes or visual tokens");
  TORCH_CHECK(p.boxes_xywh.defined() == p.box_labels.defined(),
              "boxes require labels");
  if (p.boxes_xywh.defined()) {
    const auto b = p.boxes_xywh.to(at::kFloat);
    TORCH_CHECK(b.dim() == 2 && b.size(0) > 0 && b.size(1) == 4 &&
                    p.box_labels.dim() == 1 &&
                    p.box_labels.size(0) == b.size(0),
                "boxes must be [N,4] with [N] labels");
    TORCH_CHECK(
        at::isfinite(b).all().item<bool>() && b.ge(0).all().item<bool>() &&
            b.le(1).all().item<bool>() &&
            (b.slice(1, 0, 2) + b.slice(1, 2, 4) * .5).le(1).all().item<bool>(),
        "boxes must be normalized xywh with normalized centers");
    TORCH_CHECK(((p.box_labels == 0) | (p.box_labels == 1)).all().item<bool>(),
                "box labels must be 0 or 1");
  }
  TORCH_CHECK(p.visual_features.defined() == p.visual_padding.defined(),
              "visual tokens require padding");
  if (p.visual_features.defined())
    TORCH_CHECK(p.visual_features.dim() == 3 &&
                    p.visual_features.size(1) == 1 &&
                    p.visual_features.size(2) == 256 &&
                    p.visual_padding.scalar_type() == at::kBool &&
                    p.visual_padding.sizes() ==
                        at::IntArrayRef({1, p.visual_features.size(0)}),
                "visual tokens must be [N,1,256] with bool [1,N] padding");
  s.reset();
  s.semantic = p;
  for (auto *x : {&s.semantic.boxes_xywh, &s.semantic.box_labels,
                  &s.semantic.visual_features, &s.semantic.visual_padding})
    if (x->defined())
      *x = x->clone();
  s.has_text = p.text && (s.mux || *p.text != "visual");
  s.prompt_frame = frame;
  s.encode_text();
  auto raw = s.full(frame, false, true);
  auto out = postprocess_video_output(raw, s.h, s.w, {}, {}, s.options.centers);
  s.interaction.record(frame, out);
  s.compact_cache_all();
  return out;
}
VideoOutput VideoPredictor::add_points(int64_t frame, int64_t id,
                                       const TrackingPoints &p, bool clear,
                                       bool previous, bool stateless) {
  c10::InferenceMode inference;
  auto &s = *impl_;
  s.check(frame);
  TORCH_CHECK(s.mux || clear,
              "SAM3 high-level point API replaces previous points");
  // Validate the structural input before a stateless edit removes its old
  // owner.
  TORCH_CHECK(p.points.defined() == p.labels.defined() &&
                  (p.points.defined() || p.box.defined()),
              "points require labels or a box");
  if (p.points.defined()) {
    const auto &xy = p.points;
    const auto &labels = p.labels;
    TORCH_CHECK((xy.dim() == 2 || (xy.dim() == 3 && xy.size(0) == 1)) &&
                    xy.size(-1) == 2 &&
                    ((labels.dim() == 1 && labels.size(0) == xy.size(-2)) ||
                     (labels.dim() == 2 && labels.size(0) == 1 &&
                      labels.size(1) == xy.size(-2))),
                "invalid point/label shape");
  }
  TORCH_CHECK(!p.box.defined() || p.box.numel() == 4,
              "box must contain four coordinates");
  const auto active = s.metadata.object_ids();
  if (stateless &&
      std::find(active.begin(), active.end(), id) != active.end() &&
      !video_object_was_refined(s.interaction.actions(), id)) {
    auto &previous_rank = *s.ranks[s.rank_for(id)];
    c10::DeviceGuard old_guard(previous_rank.worker->device);
    if (s.mux)
      remove_video_user_object(id, previous_rank.sessions31, s.metadata,
                               s.interaction, false);
    else
      remove_video_user_object(id, previous_rank.sessions3, s.metadata,
                               s.interaction, false);
    if (s.mask_cache) s.mask_cache->forget_object(id);
  }
  s.ensure_cache(frame);
  const auto rank = s.rank_for(id);
  auto &local = *s.ranks[rank];
  c10::DeviceGuard guard(local.worker->device);
  auto input = p;
  for (auto *tensor : {&input.points, &input.labels, &input.box})
    if (tensor->defined() && tensor->device() != local.worker->device)
      *tensor = transfer_video_tensor(*tensor, local.worker->device);
  VideoOutput out;
  if (s.mux) {
    Sam31VideoEditOptions o;
    o.rank = rank;
    o.cleanup_area = s.options.update.cleanup_area;
    o.confirmation_threshold = s.options.update.confirmation_threshold;
    o.clear_old_points = clear;
    o.use_previous_memory = previous;
    o.stateless_refinement = stateless;
    if (s.options.image_only && p.points.defined() && !p.points.numel() &&
        !p.box.defined() && s.image_masks.count(id))
      o.empty_points_mask = s.image_masks.at(id);
    out = edit_video_points(frame, id, input, local.sessions31, local.factory31,
                            s.metadata, s.interaction, s.suppressions, o,
                            s.device);
  } else {
    TORCH_CHECK(clear, "SAM3 high-level point API replaces previous points");
    VideoEditOptions o;
    o.rank = rank;
    o.cleanup_area = s.options.update.cleanup_area;
    o.confirmation_threshold = s.options.update.confirmation_threshold;
    o.use_previous_memory = previous;
    o.stateless_refinement = stateless;
    out = edit_video_points(frame, id, input, local.sessions3, local.factory3,
                            s.metadata, s.interaction, s.suppressions, o,
                            s.device);
  }
  s.initialized.insert(frame);
  s.compact_cache_all();
  return centers(std::move(out), s.h, s.w, s.options.centers);
}
VideoOutput VideoPredictor::add_mask(int64_t frame, int64_t id,
                                     const at::Tensor &mask) {
  c10::InferenceMode inference;
  auto &s = *impl_;
  s.check(frame);
  TORCH_CHECK(mask.defined() && mask.dim() == 2 && mask.numel() > 0,
              "mask must be nonempty [H,W]");
  s.ensure_cache(frame);
  const auto rank = s.rank_for(id);
  auto &local = *s.ranks[rank];
  c10::DeviceGuard guard(local.worker->device);
  const auto input = mask.device() == local.worker->device
                         ? mask
                         : transfer_video_tensor(mask, local.worker->device);
  VideoOutput out;
  if (s.mux) {
    Sam31VideoEditOptions o;
    o.rank = rank;
    o.cleanup_area = s.options.update.cleanup_area;
    o.confirmation_threshold = s.options.update.confirmation_threshold;
    out =
        edit_video_mask(frame, id, input, local.sessions31, local.factory31,
                        s.metadata, s.interaction, s.suppressions, o, s.device);
    if (s.options.image_only)
      for (const auto &session : local.sessions31)
        for (const auto &object : session->state().objects)
          if (object.id == id)
            s.image_masks[id] =
                object.masks.at(frame).squeeze(0).squeeze(0).cpu().clone();
  } else {
    VideoEditOptions o;
    o.rank = rank;
    o.cleanup_area = s.options.update.cleanup_area;
    o.confirmation_threshold = s.options.update.confirmation_threshold;
    out =
        edit_video_mask(frame, id, input, local.sessions3, local.factory3,
                        s.metadata, s.interaction, s.suppressions, o, s.device);
  }
  s.initialized.insert(frame);
  s.compact_cache_all();
  return centers(std::move(out), s.h, s.w, s.options.centers);
}
void VideoPredictor::remove_object(int64_t id) {
  auto &s = *impl_;
  auto &local = *s.ranks[s.rank_for(id)];
  c10::DeviceGuard guard(local.worker->device);
  if (s.mux)
    remove_video_user_object(id, local.sessions31, s.metadata, s.interaction);
  else
    remove_video_user_object(id, local.sessions3, s.metadata, s.interaction);
  s.image_masks.erase(id);
  if (s.mask_cache) s.mask_cache->forget_object(id);
}
VideoOutput VideoPredictor::fetch(int64_t frame) const {
  auto &s = *impl_;
  s.restore_cache(frame);
  const auto found = s.suppressions.find(frame);
  auto out = centers(s.interaction.fetch(frame, s.metadata,
                                     found == s.suppressions.end()
                                         ? std::set<int64_t>{}
                                         : found->second),
                 s.h, s.w, s.options.centers);
  s.compact_cache_all();
  return out;
}
void VideoPredictor::propagate(const VideoPredictorPropagation &request,
                               const OutputCallback &callback) {
  c10::InferenceMode inference;
  auto &s = *impl_;
  TORCH_CHECK(callback, "output callback is required");
  const auto range =
      video_processing_range(s.frames, s.initialized, request.start,
                             request.max_steps, request.reverse);
  const auto route =
      s.interaction.route(s.metadata.object_ids(), request.force_tracker);
  s.interaction.append({route.type, request.start, route.ids});
  s.cancelled.store(false);
  if (range.empty)
    return;
  VideoOutputBufferOptions o;
  o.frame_count = s.frames;
  o.end_frame = range.end;
  o.height = s.h;
  o.width = s.w;
  o.reverse = request.reverse;
  o.hotstart_delay = s.options.update.hotstart.delay;
  o.confirmation_threshold = s.options.update.confirmation_threshold;
  o.batch_size = s.options.output_batch_size;
  o.centers = s.options.centers;
  VideoOutputBuffer buffer(o);
  try {
    for (auto frame = range.first;
         request.reverse ? frame >= range.end : frame <= range.end;
         frame += range.step) {
      if (s.cancelled.load())
        break;
      if (route.type == VideoActionType::Full) {
        for (const auto &emitted :
             buffer.push(s.full(frame, request.reverse, false))) {
          s.interaction.record(emitted.frame, emitted.output);
          s.compact_cache(emitted.frame);
          if (!callback(emitted.frame, emitted.output)) {
            s.cancelled.store(true);
            break;
          }
          if (s.cancelled.load())
            break;
        }
      } else if (route.type == VideoActionType::Fetch) {
        if (!callback(frame, fetch(frame)))
          s.cancelled.store(true);
      } else {
        TORCH_CHECK(route.type == VideoActionType::Partial && route.ids,
                    "invalid propagation action");
        s.ensure_cache(frame);
        const Impl::WorkerFrame prepared(s, frame);
        std::vector<RefinedVideoObjects> outputs(s.ranks.size());
        std::vector<at::Device> devices;
        for (const auto &rank : s.ranks)
          devices.push_back(rank->worker->device);
        detail::run_tracking_ranks(devices, s.execution(), [&](size_t r) {
          const auto &rank = s.ranks[r];
          c10::DeviceGuard guard(rank->worker->device);
          RefinedVideoObjects local;
          if (s.mux) {
            std::vector<Sam31TrackingSession *> owners;
            for (auto &x : rank->sessions31)
              owners.push_back(x.get());
            local = propagate_video_refinements(frame, request.reverse,
                                                *route.ids, owners,
                                                s.options.update.cleanup_area);
          } else {
            std::vector<Sam3TrackingSession *> owners;
            for (auto &x : rank->sessions3)
              owners.push_back(x.get());
            local = propagate_video_refinements(frame, request.reverse,
                                                *route.ids, owners,
                                                s.options.update.cleanup_area);
          }
          outputs[r] = std::move(local);
        });
        RefinedVideoObjects refined;
        for (auto &local : outputs) {
          for (auto &[id, pair] : local) {
            if (pair.first.device() != s.device)
              pair.first = transfer_video_tensor(pair.first, s.device);
            if (pair.second.device() != s.device)
              pair.second = transfer_video_tensor(pair.second, s.device);
            TORCH_CHECK(refined.emplace(id, std::move(pair)).second,
                        "duplicate refined object owner");
          }
        }
        auto out =
            centers(s.interaction.merge_refined(frame, refined, s.metadata,
                                                s.suppressions[frame]),
                    s.h, s.w, s.options.centers);
        s.initialized.insert(frame);
        s.compact_cache(frame);
        if (!callback(frame, out))
          s.cancelled.store(true);
      }
    }
  } catch (...) {
    s.cancelled.store(true);
    buffer.cancel();
    if (s.mux)
      s.interaction.append({VideoActionType::Cancel, {}, {}});
    throw;
  }
  if (s.cancelled.load()) {
    buffer.cancel();
    if (s.mux)
      s.interaction.append({VideoActionType::Cancel, {}, {}});
  }
  s.compact_cache_all();
}
void VideoPredictor::cancel() noexcept {
  if (impl_)
    impl_->cancelled.store(true);
}
void VideoPredictor::reset() { impl_->reset(); }
const VideoMetadata &VideoPredictor::metadata() const {
  return impl_->metadata;
}
const VideoInteractionState &VideoPredictor::interaction() const {
  auto& s=*impl_;
  if(s.mask_cache) {
    for(auto frame:s.mask_cache->frames())s.restore_cache(frame);
    s.cache_inspection_pinned=true;
  }
  return impl_->interaction;
}
int64_t VideoPredictor::visual_encodes() const { return impl_->encodes; }
} // namespace sam3
