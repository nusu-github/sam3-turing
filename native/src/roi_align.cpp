#include "sam3/ops.h"
#include "sam3/autocast.h"
#include "roi_align_impl.h"
#include <ATen/Parallel.h>
#include <climits>

namespace sam3 {
#ifdef SAM3_WITH_CUDA
at::Tensor roi_align_cuda(const at::Tensor&, const at::Tensor&, double, int64_t, int64_t, int64_t, bool);
#endif
at::Tensor roi_align(const at::Tensor& input, const at::Tensor& rois, double scale,
                    int64_t pooled_h, int64_t pooled_w, int64_t sampling_ratio, bool aligned) {
  TORCH_CHECK(input.dim() == 4 && rois.dim() == 2 && rois.size(1) == 5,
              "ROIAlign expects input [B,C,H,W], rois [K,5]");
  TORCH_CHECK(input.device() == rois.device() && (input.is_cpu() || input.is_cuda()), "ROIAlign device mismatch");
  TORCH_CHECK(input.is_floating_point() && rois.is_floating_point(), "ROIAlign requires floating tensors");
  TORCH_CHECK(input.size(2) > 0 && input.size(3) > 0 && input.size(2) <= INT_MAX && input.size(3) <= INT_MAX,
              "invalid ROIAlign spatial dimensions");
  TORCH_CHECK(pooled_h > 0 && pooled_w > 0 && pooled_h <= INT_MAX && pooled_w <= INT_MAX && sampling_ratio <= INT_MAX,
              "invalid ROIAlign pooling dimensions");
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "ROIAlign scale must be finite and positive");
  if (sampling_ratio <= 0) sampling_ratio = -1;
  const bool autocast = at::autocast::is_autocast_enabled(input.device().type());
  AutocastGuard guard(input.device().type(), false, at::kFloat);
  const auto values = (autocast && input.scalar_type() != at::kDouble ? input.to(at::kFloat) : input).contiguous();
  const auto boxes = (autocast && rois.scalar_type() != at::kDouble ? rois.to(at::kFloat) : rois).contiguous();
  TORCH_CHECK(values.scalar_type() == boxes.scalar_type(), "ROIAlign input/ROI dtypes must match without autocast");
  TORCH_CHECK(at::isfinite(boxes).all().item<bool>(), "ROIAlign ROIs must be finite");
  const auto ids = boxes.select(1,0);
  TORCH_CHECK(((ids >= 0) & (ids < input.size(0)) & (ids == ids.floor())).all().item<bool>(),
              "ROIAlign batch indices must be integers within the batch");
  // Reject unrepresentable sample grids before conversion to the kernel's ints.
  const auto xy = boxes.slice(1,1) * scale;
  TORCH_CHECK(at::isfinite(xy).all().item<bool>() && xy.abs().le(INT_MAX / 4).all().item<bool>(),
              "ROIAlign scaled coordinates exceed index range");
  at::Tensor result;
  if (values.is_cuda()) {
#ifdef SAM3_WITH_CUDA
    result = roi_align_cuda(values,boxes,scale,pooled_h,pooled_w,sampling_ratio,aligned);
#else
    TORCH_CHECK(false,"build with SAM3_WITH_CUDA for CUDA ROIAlign");
#endif
  } else {
    result = at::empty({boxes.size(0),values.size(1),pooled_h,pooled_w},values.options());
    AT_DISPATCH_FLOATING_TYPES_AND(at::kHalf,values.scalar_type(),"roi_align_cpu",[&] {
      at::parallel_for(0,result.numel(),256,[&](int64_t begin,int64_t end) {
        for (auto i = begin; i < end; ++i)
          result.mutable_data_ptr<scalar_t>()[i] = roi_sample(i,values.const_data_ptr<scalar_t>(),
              boxes.const_data_ptr<scalar_t>(),scalar_t(scale),values.size(1),values.size(2),values.size(3),
              pooled_h,pooled_w,sampling_ratio,aligned);
      });
    });
  }
  return autocast ? result.to(input.scalar_type()) : result;
}
}
