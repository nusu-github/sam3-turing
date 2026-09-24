#include "sam3/image_results.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
namespace sam3 {
std::vector<ImageResult> postprocess_image(const DetectionOutput& input,
    const std::vector<int64_t>& heights,const std::vector<int64_t>& widths,
    double threshold,bool combine_presence,int64_t chunk_size,const std::string& mode) {
  c10::InferenceMode inference;
  detail::check_mode(mode);
  AutocastGuard autocast(input.logits.device().type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(input.logits.dim()==3 && input.logits.size(2)==1,"image logits must be [B,Q,1]");
  const auto batch=input.logits.size(0),queries=input.logits.size(1);
  TORCH_CHECK(input.boxes.sizes()==at::IntArrayRef({batch,queries,4}) && input.boxes.device()==input.logits.device(),"invalid image boxes");
  TORCH_CHECK(input.masks.dim()==4 && input.masks.size(0)==batch && input.masks.size(1)==queries && input.masks.device()==input.logits.device(),"invalid image masks");
  TORCH_CHECK(heights.size()==static_cast<size_t>(batch) && widths.size()==heights.size(),"one original size is required per prompt batch item");
  TORCH_CHECK(chunk_size>0 && std::isfinite(threshold),"invalid image postprocessing options");
  auto scores=input.logits.sigmoid();
  if (combine_presence) {
    TORCH_CHECK(input.presence_logits.sizes()==at::IntArrayRef({batch,1}) && input.presence_logits.device()==input.logits.device(),"invalid image presence logits");
    scores=scores*input.presence_logits.sigmoid().unsqueeze(1);
  }
  scores=scores.squeeze(-1);
  std::vector<ImageResult> results;
  for (int64_t b=0;b<batch;++b) {
    const auto h=heights[b],w=widths[b];
    TORCH_CHECK(h>0 && w>0,"original image dimensions must be positive");
    const auto keep=scores[b]>threshold;
    const auto selected=scores[b].index({keep});
    const auto boxes=input.boxes[b].index({keep});
    const auto masks=input.masks[b].index({keep});
    const auto cx=boxes.select(1,0),cy=boxes.select(1,1),bw=boxes.select(1,2),bh=boxes.select(1,3);
    const auto xyxy=at::stack({cx-.5*bw,cy-.5*bh,cx+.5*bw,cy+.5*bh},1);
    const auto scaled=xyxy*at::tensor({w,h,w,h},boxes.options().dtype(at::kLong)).unsqueeze(0);
    // CUDA autocast promotes bilinear interpolation to float32 in the source
    // processor. Infer output dtype from the operation, including empty results.
    const auto first_end=std::min(chunk_size,selected.numel());
    auto probabilities=at::upsample_bilinear2d(masks.slice(0,0,first_end).unsqueeze(1),{h,w},false);
    if (selected.numel()>chunk_size) {
      auto all=at::empty({selected.numel(),1,h,w},probabilities.options());
      all.slice(0,0,first_end).copy_(probabilities);
      probabilities=std::move(all);
      for (int64_t start=first_end;start<selected.numel();start+=chunk_size) {
        const auto end=std::min(start+chunk_size,selected.numel());
        auto destination=probabilities.slice(0,start,end);
        // An out= operation bypasses autocast. Match the dtype inferred by the
        // first ordinary interpolation explicitly, then write the same ATen
        // interpolation directly into the final result. This avoids a full
        // resized temporary and its device-to-device copy for every later chunk.
        const auto source=masks.slice(0,start,end).unsqueeze(1).to(probabilities.scalar_type());
        at::upsample_bilinear2d_out(destination,source,{h,w},false);
      }
    }
    // Apply sigmoid once across the assembled result. Per-chunk CPU SIMD
    // tails can otherwise differ by an ULP from the source's single sigmoid.
    probabilities.sigmoid_();
    results.push_back({scaled,selected,probabilities,probabilities>.5,at::nonzero(keep).squeeze(1)});
  }
  return results;
}
}
