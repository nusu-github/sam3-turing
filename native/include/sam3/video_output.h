#pragma once
#include "sam3/occlusion.h"
#include <deque>
#include <optional>
namespace sam3 {
struct VideoRawOutput {
  int64_t frame=0;
  std::map<int64_t,at::Tensor> masks; // bool [1,H,W], sorted by object ID
  std::map<int64_t,double> scores;
  std::map<int64_t,at::Tensor> tracker_scores; // scalar probabilities; missing -> 0
  std::set<int64_t> removed,suppressed;
  std::optional<std::set<int64_t>> unconfirmed;
  std::map<std::string,int64_t> frame_stats;
};
struct VideoOutput {
  at::Tensor ids,probabilities,boxes_xywh,masks,centers;
  // Cache keeps empty masks and pre-overlap values, as the source does.
  std::map<int64_t,at::Tensor> cached_masks;
  std::map<std::string,int64_t> frame_stats;
};
// Boxes are normalized inclusive-coordinate extents BEFORE overlap resolution.
// Empty rows are filtered before overlap only. Ties use sorted object ID order.
// Returned tensors stay on the input device except IDs/probabilities (CPU).
SAM3_NATIVE_EXPORT VideoOutput postprocess_video_output(const VideoRawOutput&,
    int64_t height,int64_t width,const std::set<int64_t>& removed={},
    const std::optional<std::set<int64_t>>& unconfirmed={},bool centers=false);
struct VideoOutputBufferOptions {
  int64_t frame_count=1,end_frame=0,height=1,width=1,hotstart_delay=15,confirmation_threshold=3,batch_size=1;
  bool reverse=false,centers=false;
};
struct VideoEmittedOutput {int64_t frame;VideoOutput output;std::set<int64_t> hidden;};
// One propagation invocation. Batch size controls emission timing, not an object
// or frame limit. End-frame arrival flushes all pending frames. cancel/reset drop
// pending output, matching destruction of an interrupted source generator.
class SAM3_NATIVE_EXPORT VideoOutputBuffer {
 public:
  explicit VideoOutputBuffer(const VideoOutputBufferOptions&);
  std::vector<VideoEmittedOutput> push(const VideoRawOutput&);
  void reset();
  void cancel();
  size_t pending()const{return delayed_.size()+ready_.size();}
 private:
  struct Ready {VideoRawOutput raw;std::set<int64_t> removed;};
  VideoOutputBufferOptions options_;
  std::deque<VideoRawOutput> delayed_;
  std::deque<Ready> ready_;
  std::set<int64_t> removed_;
  std::map<int64_t,std::set<int64_t>> unconfirmed_;
  std::optional<int64_t> last_;
  bool closed_=false;
};
}
