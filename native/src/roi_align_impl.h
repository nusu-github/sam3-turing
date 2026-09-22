#pragma once
// Adapted from torchvision ROIAlign. See third_party/torchvision/LICENSE.
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define SAM3_ROI_HD __host__ __device__
#else
#define SAM3_ROI_HD
#endif
namespace sam3 {
template <typename T>
SAM3_ROI_HD T roi_bilinear(const T* input, int height, int width, T y, T x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) return T(0);
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  int yl = static_cast<int>(y), xl = static_cast<int>(x), yh, xh;
  if (yl >= height - 1) { yh = yl = height - 1; y = T(yl); }
  else yh = yl + 1;
  if (xl >= width - 1) { xh = xl = width - 1; x = T(xl); }
  else xh = xl + 1;
  const T ly = y - yl, lx = x - xl, hy = 1. - ly, hx = 1. - lx;
  const T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  return w1 * input[static_cast<int64_t>(yl)*width+xl] +
         w2 * input[static_cast<int64_t>(yl)*width+xh] +
         w3 * input[static_cast<int64_t>(yh)*width+xl] +
         w4 * input[static_cast<int64_t>(yh)*width+xh];
}
template <typename T>
SAM3_ROI_HD T roi_sample(int64_t index, const T* input, const T* rois, T scale,
    int64_t channels, int height, int width, int pooled_h, int pooled_w,
    int sampling_ratio, bool aligned) {
  const int pw = index % pooled_w, ph = (index / pooled_w) % pooled_h;
  const int64_t c = (index / pooled_w / pooled_h) % channels;
  const int64_t n = index / pooled_w / pooled_h / channels;
  const T* roi = rois + n*5;
  const auto batch = static_cast<int64_t>(roi[0]);
  const T offset = aligned ? T(.5) : T(0);
  const T start_w = roi[1]*scale-offset, start_h = roi[2]*scale-offset;
  T roi_w = roi[3]*scale-offset-start_w, roi_h = roi[4]*scale-offset-start_h;
  if (!aligned) { if (roi_w < 1.0) roi_w = T(1); if (roi_h < 1.0) roi_h = T(1); }
  const T bin_h = roi_h / T(pooled_h), bin_w = roi_w / T(pooled_w);
  const int grid_h = sampling_ratio > 0 ? sampling_ratio : static_cast<int>(ceil(roi_h / pooled_h));
  const int grid_w = sampling_ratio > 0 ? sampling_ratio : static_cast<int>(ceil(roi_w / pooled_w));
  const T count = T(static_cast<int64_t>(grid_h) * grid_w > 0 ? static_cast<int64_t>(grid_h) * grid_w : 1);
  const T* channel = input + (batch*channels+c)*height*width;
  T result = 0;
  for (int iy = 0; iy < grid_h; ++iy) {
    const T y = start_h + ph*bin_h + T(iy+.5f)*bin_h/T(grid_h);
    for (int ix = 0; ix < grid_w; ++ix) {
      const T x = start_w + pw*bin_w + T(ix+.5f)*bin_w/T(grid_w);
      result += roi_bilinear(channel,height,width,y,x);
    }
  }
  return result / count;
}
}
#undef SAM3_ROI_HD
