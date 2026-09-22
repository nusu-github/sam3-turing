#pragma once
#include "sam3/weights.h"
#include <tuple>
namespace sam3 {
// Normalized xy points / cxcywh boxes: [N,B,2/4]. Labels [N,B] are 0/1;
// padding [B,N] is bool, true at the right-padded positions. Empty N is valid.
struct GeometryPrompt {
  at::Tensor points, point_labels, point_padding;
  at::Tensor boxes, box_labels, box_padding;
};
class SAM3_NATIVE_EXPORT GeometryEncoder {
 public:
  GeometryEncoder(const WeightStore&, const std::string& model, at::Device device = at::kCPU);
  // Image features and positions [B,256,H,W]. Returns [Npoints+Nboxes+1,B,256]
  // and right-padding [B,Npoints+Nboxes+1]. All three geometry layers are run.
  std::tuple<at::Tensor,at::Tensor> forward(const at::Tensor& image,
      const at::Tensor& positions, const GeometryPrompt&, const std::string& mode = "fp32") const;
 private:
  const at::Tensor& weight(const std::string&) const;
  at::Tensor linear(const at::Tensor&,const std::string&) const;
  at::Tensor norm(const at::Tensor&,const std::string&) const;
  at::Tensor attention(const at::Tensor& query,const at::Tensor& key,
      const at::Tensor& value,const at::Tensor& padding,const std::string& prefix,bool self) const;
  std::map<std::string,at::Tensor> weights_;
  at::Device device_;
};
}
