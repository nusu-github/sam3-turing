#pragma once
#include "sam3/detection_heads.h"
namespace sam3 {
struct ImageResult {
  at::Tensor boxes;              // [N,4] xyxy in original image pixels
  at::Tensor scores;             // [N]
  at::Tensor mask_probabilities; // [N,1,H,W]; upstream calls this masks_logits
  at::Tensor masks;              // [N,1,H,W] bool, probability > 0.5
  at::Tensor query_indices;      // [N], original query ordering
};
// No detection cap. Chunking bounds resize temporaries, not result count.
SAM3_NATIVE_EXPORT std::vector<ImageResult> postprocess_image(const DetectionOutput&,
    const std::vector<int64_t>& heights,const std::vector<int64_t>& widths,
    double confidence_threshold=.5,bool combine_presence=true,int64_t chunk_size=8,const std::string& mode="fp32");
}
