#pragma once
#include "sam3/interactive_decoder.h"
#include "sam3/vision_encoder.h"
namespace sam3 {
struct InteractiveImagePrompt {
  at::Tensor points; // [N,2] or [B,N,2], xy
  at::Tensor labels; // [N] or [B,N]
  at::Tensor boxes;  // [4] or [B,4], xyxy, prepended as labels 2/3
  at::Tensor masks;  // [1,288,288] or [B,1,288,288], previous low-res logits
  bool pixel_coordinates=true; // false means normalized [0,1] coordinates
};
struct InteractiveImageOptions {
  bool multimask=true;
  bool return_logits=false;
  double mask_threshold=0.;
  double max_hole_area=256.;
  double max_sprinkle_area=0.;
};
struct InteractiveImageResult {
  at::Tensor masks;          // [B,1|3,Horiginal,Woriginal], bool or float logits
  at::Tensor iou;            // [B,1|3], source execution dtype
  at::Tensor low_res_logits; // [B,1|3,288,288], clamped [-32,32] for re-prompting
};
SAM3_NATIVE_EXPORT at::Tensor postprocess_interactive_masks(const at::Tensor& logits,
    int64_t height,int64_t width,const InteractiveImageOptions& options={});

// Retains projected visual features and small prompt/decoder modules. Vision
// remains separately owned so hosts can release it or share a trunk computation
// with the text-grounding image API. No image/video weight variants are needed.
class SAM3_NATIVE_EXPORT InteractiveImageSession {
 public:
  InteractiveImageSession(const WeightStore&,const std::string& model,at::Device device=at::kCPU);
  void set_image(const at::Tensor& rgb,const VisionEncoder&,const std::string& mode="fp32");
  void set_images(const std::vector<at::Tensor>& rgb,const VisionEncoder&,const std::string& mode="fp32");
  void set_features(const std::vector<at::Tensor>& pyramid,const std::vector<int64_t>& heights,
      const std::vector<int64_t>& widths,const std::string& mode="fp32",bool projected_high=false);
  InteractiveImageResult predict(int64_t image_index,const InteractiveImagePrompt&,
      const InteractiveImageOptions& options={}) const;
  std::vector<InteractiveImageResult> predict_batch(const std::vector<InteractiveImagePrompt>&,
      const InteractiveImageOptions& options={}) const;
  at::Tensor image_embedding() const;
  void reset();
  int64_t image_count() const { return static_cast<int64_t>(heights_.size()); }
 private:
  InteractivePromptEncoder encoder_;
  InteractiveMaskDecoder decoder_;
  at::Tensor no_memory_,image_;
  std::vector<at::Tensor> high_;
  std::vector<int64_t> heights_,widths_;
  at::Device device_;
  std::string model_,mode_;
};
}
