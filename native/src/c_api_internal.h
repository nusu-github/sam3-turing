#pragma once
#include "sam3/c_api.h"
#include "sam3/interactive_image.h"
#include "sam3/grounding.h"
#include "sam3/image_results.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include "sam3/tracking_vision.h"
#include "sam3/multiplex_vision.h"
#include "sam3/preprocess.h"
#include "sam3/video_predictor.h"
#include <c10/core/InferenceMode.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
namespace sam3::api {
struct Failure:std::runtime_error {sam3_status status;Failure(sam3_status s,const char* message):std::runtime_error(message),status(s){}};
inline void require(bool value,const char* message){if(!value)throw Failure(SAM3_INVALID_ARGUMENT,message);}
void error(const char*) noexcept;
template<class F> sam3_status protect(F&& function) noexcept {
  try{error("");c10::InferenceMode inference;function();error("");return SAM3_OK;}
  catch(const Failure& e){error(e.what());return e.status;}
  catch(const c10::OutOfMemoryError& e){error(e.what());return SAM3_OUT_OF_MEMORY;}
  catch(const std::bad_alloc& e){error(e.what());return SAM3_OUT_OF_MEMORY;}
  catch(const std::exception& e){error(e.what());return SAM3_RUNTIME_ERROR;}
  catch(...){error("unknown native exception");return SAM3_RUNTIME_ERROR;}
}
template<class T> void options(const T* p){require(p && p->struct_size>=sizeof(T),"invalid or undersized option struct");}
template<class T> void output(T** p){require(p,"output pointer is required");*p=nullptr;}
class BusyGuard {
 public:
  explicit BusyGuard(std::atomic_flag& flag):flag_(flag){if(flag_.test_and_set(std::memory_order_acquire))throw Failure(SAM3_BUSY,"session is already in use; callback reentry is not allowed");}
  ~BusyGuard(){flag_.clear(std::memory_order_release);}
  BusyGuard(const BusyGuard&)=delete;BusyGuard& operator=(const BusyGuard&)=delete;
 private:std::atomic_flag& flag_;
};
inline BusyGuard lock(std::atomic_flag& flag){return BusyGuard(flag);}

at::Tensor tensor(const sam3_tensor_view*);
at::Tensor rgb(const sam3_rgb_view&);
int32_t dtype(at::ScalarType);
void validate_video_options(const sam3_video_options&,bool require_provider=true);
TrackingSessionOptions tracking_options(const sam3_video_options&);
MultiplexSessionOptions multiplex_options(const sam3_video_options&);
VideoPredictor::FrameProvider frame_provider(const sam3_video_options&);
struct Context {
  WeightStore store;std::string model,mode;at::Device device;std::filesystem::path vocabulary;
  std::mutex mutex;std::shared_ptr<const VisionEncoder> vision_cache;
  std::shared_ptr<const TextEncoder> text_cache;std::shared_ptr<const GroundingDetector> detector_cache;
  std::shared_ptr<const InteractiveImageSession> interactive_cache;
  std::shared_ptr<const Sam3TrackingFrame> tracking_cache;
  std::shared_ptr<const Sam31TrackingFrame> multiplex_cache;
  std::unique_ptr<Tokenizer> tokenizer;
  explicit Context(const sam3_context_options&);
  std::shared_ptr<const VisionEncoder> vision();
  std::shared_ptr<const TextEncoder> text();
  std::shared_ptr<const GroundingDetector> detector();
  std::shared_ptr<const InteractiveImageSession> interactive();
  std::shared_ptr<const Sam3TrackingFrame> tracking();
  std::shared_ptr<const Sam31TrackingFrame> multiplex();
  at::Tensor tokenize(const char* const*,int64_t);
  at::Tensor tokenize(const std::vector<std::string>&);
  void trim();
};
void result(std::map<std::string,at::Tensor>,sam3_result**);
void video_result(const TrackingSessionOutput&,sam3_result**);
bool emit(const TrackingSessionOutput&,sam3_output_callback,void*);
}
struct sam3_context {std::shared_ptr<sam3::api::Context> value;};
struct sam3_result {
  mutable std::atomic<uint64_t> refs{1};std::map<std::string,at::Tensor> fields;
  mutable std::map<std::string,at::Tensor> host;mutable std::mutex mutex;
};
struct sam3_image {
  std::shared_ptr<sam3::api::Context> context;uint32_t flags;std::atomic_flag mutex=ATOMIC_FLAG_INIT;
  std::map<std::string,std::vector<at::Tensor>> pyramids;at::Tensor position;
  std::vector<int64_t> heights,widths;
  std::shared_ptr<const sam3::InteractiveImageSession> prototype;
  std::unique_ptr<sam3::InteractiveImageSession> interactive;
};
struct sam3_predictor {
  std::shared_ptr<sam3::api::Context> context;std::atomic_flag mutex=ATOMIC_FLAG_INIT;
  std::unique_ptr<sam3::VideoPredictor> value;
};
struct sam3_video {
  std::shared_ptr<sam3::api::Context> context;std::atomic_flag mutex=ATOMIC_FLAG_INIT;
  std::shared_ptr<const sam3::Sam3TrackingVision> vision3;
  std::shared_ptr<const sam3::Sam31TrackingVision> vision31;
  std::unique_ptr<sam3::Sam3TrackingSession> tracker;
  std::unique_ptr<sam3::Sam31TrackingSession> multiplex;
};
namespace sam3::api {
template<class F> auto video(sam3_video* handle,F&& function){
  require(handle,"video handle is required");auto guard=lock(handle->mutex);
  if(handle->multiplex)return function(*handle->multiplex);return function(*handle->tracker);
}
}
