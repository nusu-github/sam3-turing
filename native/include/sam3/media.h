#pragma once
#include "sam3/weights.h"
#include <memory>
namespace sam3 {
struct MediaOptions {bool image_only=false;int threads=1;};
struct MediaInfo {int64_t frames=0,height=0,width=0;double fps=0;bool image_only=false;};
struct MediaFrame {at::Tensor rgb;double seconds=0,duration=0;};
// Local file or numerically/lexicographically sorted image folder. Reads are
// serialized; returned RGB tensors own their pixels and survive the source.
class SAM3_NATIVE_EXPORT MediaSource {
 public:
  explicit MediaSource(const std::filesystem::path&,const MediaOptions& options={});
  ~MediaSource();
  MediaSource(const MediaSource&)=delete;MediaSource& operator=(const MediaSource&)=delete;
  const MediaInfo& info()const;
  MediaFrame read(int64_t index);
 private:struct Impl;std::unique_ptr<Impl> impl_;
};
SAM3_NATIVE_EXPORT bool media_available() noexcept;
SAM3_NATIVE_EXPORT void write_rgb_png(const std::filesystem::path&,const at::Tensor& rgb);
}
