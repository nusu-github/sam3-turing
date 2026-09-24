#include "sam3/video_output.h"
#include <c10/core/InferenceMode.h>
#include <algorithm>
namespace sam3 {
namespace {
std::set<int64_t> hidden_ids(const VideoRawOutput& raw,const std::set<int64_t>& removed,const std::optional<std::set<int64_t>>& unconfirmed){auto hidden=removed;hidden.insert(raw.suppressed.begin(),raw.suppressed.end());if(unconfirmed)hidden.insert(unconfirmed->begin(),unconfirmed->end());return hidden;}
VideoRawOutput snapshot(const VideoRawOutput& raw){auto out=raw;for(auto& [id,x]:out.masks)x=x.clone();for(auto& [id,x]:out.tracker_scores)x=x.clone();return out;}
}
VideoOutput postprocess_video_output(const VideoRawOutput& raw,int64_t h,int64_t w,const std::set<int64_t>& removed,const std::optional<std::set<int64_t>>& unconfirmed,bool centers){
  c10::InferenceMode inference;TORCH_CHECK(h>0 && w>0,"video dimensions must be positive");
  const auto hidden=hidden_ids(raw,removed,unconfirmed);VideoOutput out;out.frame_stats=raw.frame_stats;
  const auto device=raw.masks.empty()?at::Device(at::kCPU):raw.masks.begin()->second.device();const auto opts=at::TensorOptions().device(device);
  std::vector<int64_t> ids;std::vector<float> probabilities;std::vector<at::Tensor> masks,scores;
  for(const auto& [id,mask]:raw.masks){
    TORCH_CHECK(mask.scalar_type()==at::kBool && mask.sizes()==at::IntArrayRef({1,h,w}) && mask.device()==device && raw.scores.count(id),"raw video masks must be bool [1,H,W] with scores on one device");
    if(hidden.count(id))continue;out.cached_masks.emplace(id,mask);
    ids.push_back(id);probabilities.push_back(float(raw.scores.at(id)));masks.push_back(mask);
    const auto it=raw.tracker_scores.find(id);auto score=it==raw.tracker_scores.end()?at::zeros({},opts.dtype(at::kFloat)):it->second.to(device);TORCH_CHECK(score.numel()==1 && score.is_floating_point(),"tracker probability must be a floating scalar");scores.push_back(score.reshape({}));
  }
  out.masks=masks.empty()?at::empty({0,h,w},opts.dtype(at::kBool)):at::cat(masks);
  if(!masks.empty()){
    const auto keep=out.masks.any(at::IntArrayRef{1,2}).cpu();std::vector<int64_t> selected,new_ids;std::vector<float> new_probabilities;std::vector<at::Tensor> new_scores;
    for(size_t i=0;i<ids.size();++i)if(keep[i].item<bool>()){selected.push_back(i);new_ids.push_back(ids[i]);new_probabilities.push_back(probabilities[i]);new_scores.push_back(scores[i]);}
    out.masks=out.masks.index_select(0,at::tensor(selected,at::kLong).to(device));ids=std::move(new_ids);probabilities=std::move(new_probabilities);scores=std::move(new_scores);
  }
  out.ids=at::tensor(ids,at::kLong);out.probabilities=at::tensor(probabilities,at::kFloat);
  auto boxes=tracking_mask_boxes(out.masks.unsqueeze(1)).squeeze(1).to(at::kFloat);
  out.boxes_xywh=at::stack({boxes.select(1,0)/w,boxes.select(1,1)/h,(boxes.select(1,2)-boxes.select(1,0))/w,(boxes.select(1,3)-boxes.select(1,1))/h},1);
  if(ids.size()>1){
    const auto score=at::stack(scores).view({-1,1,1});const auto pixels=at::where(out.masks,score,0.);
    const auto winner=pixels.argmax(0,true),index=at::arange(ids.size(),opts.dtype(at::kLong)).view({-1,1,1});
    out.masks=out.masks.logical_and(pixels.gt(0)).logical_and(winner.eq(index));
  }
  if(centers){
    const auto y=at::arange(h,opts.dtype(at::kFloat)).view({1,h,1}),x=at::arange(w,opts.dtype(at::kFloat)).view({1,1,w});
    const auto mass=out.masks.sum(at::IntArrayRef{1,2}).clamp_min(1e-6);
    out.centers=at::stack({(out.masks*x).sum(at::IntArrayRef{1,2})/mass/w,(out.masks*y).sum(at::IntArrayRef{1,2})/mass/h},1);
  }
  return out;
}
VideoOutputBuffer::VideoOutputBuffer(const VideoOutputBufferOptions& options):options_(options){TORCH_CHECK(options.height>0 && options.width>0 && options.frame_count>0 && options.end_frame>=0 && options.end_frame<options.frame_count && options.hotstart_delay>=0 && options.confirmation_threshold>0 && options.batch_size>0,"invalid output buffer options");}
void VideoOutputBuffer::reset(){delayed_.clear();ready_.clear();removed_.clear();unconfirmed_.clear();last_.reset();closed_=false;}
void VideoOutputBuffer::cancel(){reset();closed_=true;}
std::vector<VideoEmittedOutput> VideoOutputBuffer::push(const VideoRawOutput& input){
  c10::InferenceMode inference;const auto f=input.frame;TORCH_CHECK(!closed_ && f>=0 && f<options_.frame_count && (!last_ || (options_.reverse?f<*last_:f>*last_)),"output frames must follow propagation direction in an open buffer");
  // Snapshot mutable metadata/tensors before buffering, including the removal
  // set at promotion time, not when a later postprocess batch finally runs.
  auto raw=snapshot(input);last_=f;const bool end=f==options_.end_frame;
  if(options_.hotstart_delay>0){
    removed_.insert(raw.removed.begin(),raw.removed.end());if(raw.unconfirmed)unconfirmed_[f]=*raw.unconfirmed;delayed_.push_back(std::move(raw));
    const size_t count=end?delayed_.size():(delayed_.size()>=size_t(options_.hotstart_delay)?1:0);
    for(size_t i=0;i<count;++i){ready_.push_back({std::move(delayed_.front()),removed_});delayed_.pop_front();}
  }else ready_.push_back({std::move(raw),{}});
  std::vector<VideoEmittedOutput> result;
  const auto count=end?ready_.size():(ready_.size()/options_.batch_size)*options_.batch_size;
  for(size_t i=0;i<count;++i){
    auto& entry=ready_.front();const auto distance=std::min(options_.confirmation_threshold-1,options_.frame_count-1);
    const auto target=options_.reverse?entry.raw.frame-std::min(entry.raw.frame,distance):entry.raw.frame+std::min(options_.frame_count-1-entry.raw.frame,distance);
    std::optional<std::set<int64_t>> unconfirmed;const auto found=unconfirmed_.find(target);if(found!=unconfirmed_.end())unconfirmed=found->second;
    const auto h=options_.height,w=options_.width;
    result.push_back({entry.raw.frame,postprocess_video_output(entry.raw,h,w,entry.removed,unconfirmed,options_.centers),hidden_ids(entry.raw,entry.removed,unconfirmed)});ready_.pop_front();
  }
  if(end)closed_=true;return result;
}
}
