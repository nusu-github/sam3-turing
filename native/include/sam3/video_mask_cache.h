#pragma once
#include "sam3/tensor_archive.h"

namespace sam3 {
enum class VideoMaskStorage { PackedCPU = 1, PackedDisk = 2 };
struct VideoMaskCacheStats {
  int64_t frames = 0, masks = 0;
  uint64_t logical_bytes = 0, packed_bytes = 0, disk_bytes = 0;
};
// Lossless storage of every frame/object mask. Reads restore the original device
// and dense strides. Disk files are temporary and belong to this cache; the
// caller owns the parent directory. Calls are exclusive. No model weights are
// stored here. Inputs are dense bool [1,H,W]; each read returns an isolated copy.
class SAM3_NATIVE_EXPORT VideoMaskCache {
public:
  VideoMaskCache(int64_t height, int64_t width, VideoMaskStorage,
                 const std::filesystem::path& directory = {});
  ~VideoMaskCache();
  VideoMaskCache(VideoMaskCache&&) noexcept;
  VideoMaskCache& operator=(VideoMaskCache&&) noexcept;
  void store(int64_t frame, const std::map<int64_t, at::Tensor>& masks);
  std::map<int64_t, at::Tensor> read(int64_t frame) const;
  bool contains(int64_t frame) const;
  std::vector<int64_t> frames() const;
  void erase(int64_t frame);
  void forget_object(int64_t id);
  void clear();
  VideoMaskCacheStats stats() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
