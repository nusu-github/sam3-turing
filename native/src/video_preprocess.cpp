#include "sam3/preprocess.h"
#include "sam3/autocast.h"
#include <c10/core/InferenceMode.h>
#include <climits>
#ifdef SAM3_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif
namespace sam3 {
bool video_preprocess_available(VideoPreprocess policy)noexcept{
  switch(policy){
    case VideoPreprocess::ImageFolder:case VideoPreprocess::PilList:case VideoPreprocess::TorchCodecCpu:return true;
    case VideoPreprocess::TorchCodecCuda:return at::hasCUDA();
#ifdef SAM3_WITH_OPENCV
    case VideoPreprocess::Cv2Source:return true;
#endif
    default:return false;
  }
}
at::Tensor preprocess_video_rgb(const at::Tensor& pixels,VideoPreprocess policy,at::Device device){
  c10::InferenceMode inference;
  TORCH_CHECK(pixels.dim()==3 && pixels.size(0)==3 && pixels.size(1)>0 && pixels.size(2)>0 && pixels.scalar_type()==at::kByte,"video preprocessing requires U8 RGB [3,H,W]");
  TORCH_CHECK(video_preprocess_available(policy),"video preprocessing policy unavailable; Cv2Source requires SAM3_WITH_OPENCV=ON");
  if(policy==VideoPreprocess::ImageFolder)return preprocess_video_rgb(pixels);
  if(policy==VideoPreprocess::PilList){
    // PIL.Image.resize defaults to bicubic. The list loader divides in NumPy
    // float64 before storing half, unlike the folder loader's F32 conversion.
    auto image=resize_tracking_rgb(pixels,1008,1008).to(at::kDouble).div_(255.).to(at::kHalf);
    return image.sub_(.5).div_(.5).to(at::kFloat).unsqueeze(0);
  }
  if(policy==VideoPreprocess::TorchCodecCpu || policy==VideoPreprocess::TorchCodecCuda){
    const auto target=policy==VideoPreprocess::TorchCodecCpu?at::Device(at::kCPU):device;
    TORCH_CHECK(policy!=VideoPreprocess::TorchCodecCuda || target.is_cuda(),"TorchCodecCuda preprocessing requires a CUDA device");
    AutocastGuard autocast(target.type(),false,at::kFloat);
    auto image=pixels.to(target,at::kFloat).unsqueeze(0);
    image=at::upsample_bicubic2d(image,{1008,1008},false).to(at::kHalf);
    // Preserve half rounding after each in-place operation. No clamp: source
    // bicubic float interpolation can overshoot the input's byte range.
    return image.div_(255.).sub_(.5).div_(.5).to(at::kFloat);
  }
#ifdef SAM3_WITH_OPENCV
  auto bytes=pixels.cpu().permute({1,2,0}).contiguous();
  TORCH_CHECK(bytes.size(0)<=INT_MAX && bytes.size(1)<=INT_MAX,"OpenCV input dimensions exceed int");
  const cv::Mat input(int(bytes.size(0)),int(bytes.size(1)),CV_8UC3,bytes.data_ptr<uint8_t>());cv::Mat resized;
  cv::resize(input,resized,cv::Size(1008,1008),0,0,cv::INTER_CUBIC);
  auto image=at::from_blob(resized.data,{1008,1008,3},at::TensorOptions().dtype(at::kByte)).to(at::kFloat).permute({2,0,1});
  // Match io_utils.py literally: F32 byte values, F16 mean/std, no /255.
  return image.sub_(.5).div_(.5).unsqueeze(0);
#else
  TORCH_CHECK(false,"OpenCV preprocessing unavailable");
#endif
}
}
