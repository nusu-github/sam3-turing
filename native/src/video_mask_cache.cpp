#include "sam3/video_mask_cache.h"
#include "sam3/ops.h"
#include <c10/core/InferenceMode.h>

namespace sam3 {
struct VideoMaskCache::Impl {
  struct Mask { at::Device device; std::vector<int64_t> strides; };
  struct Frame {
    std::map<int64_t, Mask> masks;
    std::map<std::string, at::Tensor> packed;
    std::shared_ptr<const TensorArchive> archive;
  };
  int64_t height, width;
  VideoMaskStorage storage;
  std::filesystem::path directory;
  std::map<int64_t, Frame> frames;
};
VideoMaskCache::VideoMaskCache(int64_t h, int64_t w, VideoMaskStorage storage,
    const std::filesystem::path& directory)
    : impl_(std::make_unique<Impl>()) {
  TORCH_CHECK(h > 0 && w > 0 && h <= INT64_MAX / w, "invalid mask cache dimensions");
  TORCH_CHECK(storage == VideoMaskStorage::PackedCPU || storage == VideoMaskStorage::PackedDisk,
              "invalid mask cache storage");
  TORCH_CHECK(storage != VideoMaskStorage::PackedDisk || !directory.empty(),
              "disk mask cache requires a directory");
  impl_->height=h; impl_->width=w; impl_->storage=storage; impl_->directory=directory;
}
VideoMaskCache::~VideoMaskCache() = default;
VideoMaskCache::VideoMaskCache(VideoMaskCache&&) noexcept = default;
VideoMaskCache& VideoMaskCache::operator=(VideoMaskCache&&) noexcept = default;
void VideoMaskCache::store(int64_t frame, const std::map<int64_t, at::Tensor>& masks) {
  c10::InferenceMode inference;
  TORCH_CHECK(frame >= 0, "negative mask cache frame");
  auto& s=*impl_; Impl::Frame next;
  // Commit only after every mask has been encoded and the archive has closed.
  // A validation, transfer or I/O error leaves the prior frame intact.
  for(const auto& [id, mask]:masks) {
    TORCH_CHECK(mask.scalar_type()==at::kBool &&
                    mask.sizes()==at::IntArrayRef({1,s.height,s.width}) &&
                    mask.is_non_overlapping_and_dense(),
                "cached masks require dense bool [1,H,W]");
    next.masks.emplace(id,Impl::Mask{mask.device(),mask.strides().vec()});
    next.packed.emplace(std::to_string(id),pack_masks(mask).cpu());
  }
  if(s.storage==VideoMaskStorage::PackedDisk && !next.packed.empty()) {
    next.archive=TensorArchive::write(s.directory,next.packed);
    next.packed.clear();
  }
  s.frames.insert_or_assign(frame,std::move(next));
}
std::map<int64_t,at::Tensor> VideoMaskCache::read(int64_t frame) const {
  c10::InferenceMode inference;
  const auto& s=*impl_;const auto found=s.frames.find(frame);
  TORCH_CHECK(found!=s.frames.end(),"mask cache frame is missing");
  std::map<int64_t,at::Tensor> out;
  for(const auto& [id,meta]:found->second.masks) {
    const auto name=std::to_string(id);
    const auto packed=found->second.archive?found->second.archive->read(name):found->second.packed.at(name);
    auto mask=unpack_masks(packed.to(meta.device),s.height,s.width).squeeze(0);
    if(mask.strides()!=at::IntArrayRef(meta.strides)) {
      auto strided=at::empty_strided({1,s.height,s.width},meta.strides,mask.options());
      strided.copy_(mask);mask=std::move(strided);
    }
    out.emplace(id,std::move(mask));
  }
  return out;
}
bool VideoMaskCache::contains(int64_t frame) const {return impl_->frames.count(frame)!=0;}
std::vector<int64_t> VideoMaskCache::frames() const {
  std::vector<int64_t> out;for(const auto& [frame,_]:impl_->frames)out.push_back(frame);return out;
}
void VideoMaskCache::erase(int64_t frame) {impl_->frames.erase(frame);}
void VideoMaskCache::forget_object(int64_t id) {
  for(auto& [_,frame]:impl_->frames) {
    frame.masks.erase(id);frame.packed.erase(std::to_string(id));
    if(frame.masks.empty())frame.archive.reset();
  }
}
void VideoMaskCache::clear() {impl_->frames.clear();}
VideoMaskCacheStats VideoMaskCache::stats() const {
  const auto& s=*impl_;VideoMaskCacheStats out;out.frames=s.frames.size();
  const auto pixels=uint64_t(s.height)*s.width, bytes=(pixels+7)/8;
  for(const auto& [_,frame]:s.frames) {
    out.masks+=frame.masks.size();out.logical_bytes+=pixels*frame.masks.size();
    out.packed_bytes+=bytes*frame.masks.size();
    if(frame.archive)out.disk_bytes+=frame.archive->bytes();
  }
  return out;
}
}
