#include "c_api_internal.h"
#include <cmath>
using namespace sam3::api;
namespace {
void set_images(sam3_image& image,const sam3_rgb_view* inputs,int64_t count,bool batch){
  require(inputs && count>0,"nonempty RGB image batch is required");auto& c=*image.context;
  std::vector<int64_t> heights,widths;std::vector<at::Tensor> normalized;
  for(int64_t i=0;i<count;++i){const auto pixels=rgb(inputs[i]);heights.push_back(pixels.size(1));widths.push_back(pixels.size(2));normalized.push_back(sam3::preprocess_rgb(pixels.to(c.device)));}
  const auto input=batch?at::stack(normalized).squeeze(1):normalized.front();std::vector<std::string> heads;
  if(image.flags&SAM3_IMAGE_GROUNDING)heads.push_back("convs");if(image.flags&SAM3_IMAGE_INTERACTIVE)heads.push_back(c.model=="sam3"?"sam2_convs":"interactive_convs");
  const std::vector<int64_t> positions=(image.flags&SAM3_IMAGE_GROUNDING)?std::vector<int64_t>{2}:std::vector<int64_t>{};
  auto features=c.vision()->forward(input,c.mode,heads,positions);
  if(c.model=="sam3")for(auto& [name,pyramid]:features.pyramid)pyramid.pop_back();
  image.pyramids=std::move(features.pyramid);image.position=features.positions[2];image.heights=std::move(heights);image.widths=std::move(widths);image.interactive.reset();
}
void prepare_interactive(sam3_image& image){
  require((image.flags&SAM3_IMAGE_INTERACTIVE) && !image.heights.empty(),"interactive features have not been set");if(image.interactive)return;
  auto& c=*image.context;auto prototype=c.interactive();auto instance=std::make_unique<sam3::InteractiveImageSession>(*prototype);
  instance->set_features(image.pyramids.at(c.model=="sam3"?"sam2_convs":"interactive_convs"),image.heights,image.widths,c.mode);
  image.prototype=std::move(prototype);image.interactive=std::move(instance);
}
}
extern "C" {
sam3_status sam3_image_create(sam3_context* context,uint32_t flags,sam3_image** out) noexcept{return protect([&]{output(out);require(context && flags>0 && (flags&~SAM3_IMAGE_ALL)==0,"context and valid image feature flags are required");auto image=std::make_unique<sam3_image>();image->context=context->value;image->flags=flags;*out=image.release();});}
void sam3_image_release(sam3_image* image) noexcept{delete image;}
sam3_status sam3_image_set_rgb(sam3_image* image,const sam3_rgb_view* pixels) noexcept{return protect([&]{require(image && pixels,"image and RGB view are required");auto guard=lock(image->mutex);set_images(*image,pixels,1,false);});}
sam3_status sam3_image_set_rgb_batch(sam3_image* image,const sam3_rgb_view* pixels,int64_t count) noexcept{return protect([&]{require(image,"image is required");auto guard=lock(image->mutex);set_images(*image,pixels,count,true);});}
sam3_status sam3_image_reset(sam3_image* image) noexcept{return protect([&]{require(image,"image is required");auto guard=lock(image->mutex);image->pyramids.clear();image->position=at::Tensor();image->heights.clear();image->widths.clear();image->interactive.reset();image->prototype.reset();});}
sam3_status sam3_interactive_request_init(sam3_interactive_request* o) noexcept{return protect([&]{require(o,"request is required");*o={};o->struct_size=sizeof(*o);o->pixel_coordinates=1;o->multimask=1;o->max_hole_area=256.;});}
sam3_status sam3_image_predict(sam3_image* image,const sam3_interactive_request* o,sam3_result** out) noexcept{return protect([&]{output(out);require(image,"image is required");options(o);auto guard=lock(image->mutex);prepare_interactive(*image);
  const sam3::InteractiveImagePrompt prompt{tensor(o->points),tensor(o->labels),tensor(o->boxes),tensor(o->masks),bool(o->pixel_coordinates)};
  require(std::isfinite(o->mask_threshold) && std::isfinite(o->max_hole_area) && std::isfinite(o->max_sprinkle_area) && o->max_hole_area>=0 && o->max_sprinkle_area>=0,"invalid interactive postprocessing options");
  const auto value=image->interactive->predict(o->image_index,prompt,{bool(o->multimask),bool(o->return_logits),o->mask_threshold,o->max_hole_area,o->max_sprinkle_area});
  result({{"masks",value.masks},{"iou",value.iou},{"low_res_logits",value.low_res_logits}},out);
});}
sam3_status sam3_image_embedding(sam3_image* image,sam3_result** out) noexcept{return protect([&]{output(out);require(image,"image is required");auto guard=lock(image->mutex);prepare_interactive(*image);result({{"embedding",image->interactive->image_embedding()}},out);});}
sam3_status sam3_grounding_request_init(sam3_grounding_request* o) noexcept{return protect([&]{require(o,"request is required");*o={};o->struct_size=sizeof(*o);o->use_text=1;o->confidence_threshold=.5;o->resize_chunk=8;});}
sam3_status sam3_image_ground(sam3_image* image,const sam3_grounding_request* o,sam3_result** out) noexcept{return protect([&]{
  output(out);require(image,"image is required");options(o);auto guard=lock(image->mutex);require((image->flags&SAM3_IMAGE_GROUNDING) && !image->heights.empty(),"grounding features have not been set");auto& c=*image->context;
  auto ids=o->image_ids?tensor(o->image_ids):at::zeros({1},at::kLong);require(ids.dim()==1 && ids.scalar_type()==at::kLong && ids.numel()>0,"image_ids must be nonempty int64[B]");const auto batch=ids.numel();
  std::vector<int64_t> heights,widths;for(int64_t i=0;i<batch;++i){const auto index=ids[i].item<int64_t>();require(index>=0 && uint64_t(index)<image->heights.size(),"image ID is out of range");heights.push_back(image->heights[index]);widths.push_back(image->widths[index]);}
  sam3::GroundingPrompt prompt;prompt.image_ids=ids.to(c.device);prompt.text_ids=(o->text_ids?tensor(o->text_ids):at::zeros({batch},at::kLong)).to(c.device);
  prompt.text_features=tensor(o->text_features);prompt.text_padding=tensor(o->text_padding);prompt.use_text=o->use_text!=0;
  const auto geometry=[&](const sam3_tensor_view* coords,const sam3_tensor_view* labels,const sam3_tensor_view* padding,int64_t channels){
    require(!coords || labels,"geometry coordinates require labels");auto x=coords?tensor(coords):at::empty({0,batch,channels},at::kFloat);
    auto y=labels?tensor(labels):at::empty({0,batch},at::kLong);auto mask=padding?tensor(padding):at::zeros({batch,x.size(0)},at::kBool);
    return std::make_tuple(x.to(c.device),y.to(c.device),mask.to(c.device));
  };
  std::tie(prompt.geometry.points,prompt.geometry.point_labels,prompt.geometry.point_padding)=geometry(o->points,o->point_labels,o->point_padding,2);
  std::tie(prompt.geometry.boxes,prompt.geometry.box_labels,prompt.geometry.box_padding)=geometry(o->boxes,o->box_labels,o->box_padding,4);
  prompt.visual_features=tensor(o->visual_features);prompt.visual_padding=tensor(o->visual_padding);prompt.previous_mask=tensor(o->previous_mask);
  const bool joint=c.model=="sam3.1";const auto raw=c.detector()->forward(image->pyramids.at("convs"),image->position,prompt,joint,c.mode);
  const auto processed=sam3::postprocess_image(raw.detection,heights,widths,o->confidence_threshold,!joint,o->resize_chunk,c.mode);
  std::map<std::string,at::Tensor> fields{{"batch_count",at::scalar_tensor(batch,at::kLong)},{"raw/logits",raw.detection.logits},{"raw/boxes",raw.detection.boxes},{"raw/masks",raw.detection.masks},{"raw/semantic",raw.detection.semantic},{"raw/presence_logits",raw.detection.presence_logits},{"raw/queries",raw.detection.queries},{"raw/presence",raw.detection.presence}};
  for(size_t i=0;i<processed.size();++i){const auto prefix=std::to_string(i)+"/";const auto& value=processed[i];fields.emplace(prefix+"boxes",value.boxes);fields.emplace(prefix+"scores",value.scores);fields.emplace(prefix+"mask_probabilities",value.mask_probabilities);fields.emplace(prefix+"masks",value.masks);fields.emplace(prefix+"query_indices",value.query_indices);}
  result(std::move(fields),out);
});}
}
