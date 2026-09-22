#pragma once
#include "sam3/weights.h"
#include <array>

namespace sam3 {
struct InteractivePrompt {
  at::Tensor points; // optional [B,N,2], pixel xy in the model input image
  at::Tensor labels; // [B,N]: -1 padding, 0/1 clicks, 2/3 box corners
  at::Tensor boxes;  // optional [B,4], pixel xyxy
  at::Tensor masks;  // optional [B,1,4*grid_h,4*grid_w], unthresholded values
};
struct InteractiveEmbeddings {
  at::Tensor sparse;   // [B,Nsparse,256]
  at::Tensor dense;    // [B,256,grid_h,grid_w]
  at::Tensor position; // [1,256,grid_h,grid_w]
};
class SAM3_NATIVE_EXPORT InteractivePromptEncoder {
 public:
  InteractivePromptEncoder(const WeightStore&,const std::string& model,at::Device device=at::kCPU,
      std::array<int64_t,2> grid={72,72},std::array<int64_t,2> input_size={1008,1008});
  InteractiveEmbeddings forward(const InteractivePrompt&,const std::string& mode="fp32") const;
 private:
  at::Tensor position(const at::Tensor&) const;
  at::Tensor coordinates(const at::Tensor&) const;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
  std::array<int64_t,2> grid_,input_size_;
};
}
