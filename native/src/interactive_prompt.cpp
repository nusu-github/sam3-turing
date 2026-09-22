#include "sam3/interactive_prompt.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
InteractivePromptEncoder::InteractivePromptEncoder(const WeightStore& store,const std::string& model,
    at::Device device,std::array<int64_t,2> grid,std::array<int64_t,2> input_size)
    :device_(device),grid_(grid),input_size_(input_size) {
  TORCH_CHECK(model=="sam3" || model=="sam3.1","unknown interactive model");
  TORCH_CHECK(grid[0]>0 && grid[1]>0 && input_size[0]>0 && input_size[1]>0,"invalid interactive image sizes");
  weights_=detail::load(store,model+(model=="sam3"?"/tracker.sam_prompt_encoder.":"/tracker.model.interactive_sam_prompt_encoder."),device);
  device_=detail::weight(weights_,"no_mask_embed.weight").device();
}
at::Tensor InteractivePromptEncoder::position(const at::Tensor& coords) const {
  const auto phase=at::matmul(2*coords-1,detail::weight(weights_,"pe_layer.positional_encoding_gaussian_matrix"))*(2*std::acos(-1.));
  return at::cat({phase.sin(),phase.cos()},-1);
}
at::Tensor InteractivePromptEncoder::coordinates(const at::Tensor& input) const {
  auto coords=input.clone();
  coords.select(2,0).copy_(coords.select(2,0)/input_size_[1]);
  coords.select(2,1).copy_(coords.select(2,1)/input_size_[0]);
  return position(coords.to(at::kFloat));
}
InteractiveEmbeddings InteractivePromptEncoder::forward(const InteractivePrompt& prompt,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  const auto options=at::TensorOptions().dtype(at::kFloat).device(device_);
  int64_t batch=1;
  if (prompt.points.defined()) batch=prompt.points.size(0);
  else if (prompt.boxes.defined()) batch=prompt.boxes.size(0);
  else if (prompt.masks.defined()) batch=prompt.masks.size(0);
  auto sparse=at::empty({batch,0,256},options);
  if (prompt.points.defined()) {
    TORCH_CHECK(prompt.points.dim()==3 && prompt.points.size(2)==2,"interactive points must be [B,N,2]");
    TORCH_CHECK(prompt.labels.defined() && prompt.labels.sizes()==prompt.points.sizes().slice(0,2),"interactive point labels must be [B,N]");
    auto points=prompt.points.to(device_)+.5,labels=prompt.labels.to(device_);
    if (!prompt.boxes.defined()) {
      points=at::cat({points,at::zeros({batch,1,2},options)},1);
      labels=at::cat({labels,-at::ones({batch,1},options)},1);
    }
    auto embedding=coordinates(points);
    embedding=at::where((labels==-1).unsqueeze(-1),at::zeros_like(embedding)+detail::weight(weights_,"not_a_point_embed.weight"),embedding);
    for (int label=0;label<4;++label)
      embedding=at::where((labels==label).unsqueeze(-1),embedding+detail::weight(weights_,"point_embeddings."+std::to_string(label)+".weight"),embedding);
    sparse=at::cat({sparse,embedding},1);
  } else TORCH_CHECK(!prompt.labels.defined(),"point labels require coordinates");
  if (prompt.boxes.defined()) {
    TORCH_CHECK(prompt.boxes.sizes()==at::IntArrayRef({batch,4}),"interactive boxes must be [B,4]");
    auto embedding=coordinates((prompt.boxes.to(device_)+.5).reshape({-1,2,2}));
    embedding.select(1,0).add_(detail::weight(weights_,"point_embeddings.2.weight"));
    embedding.select(1,1).add_(detail::weight(weights_,"point_embeddings.3.weight"));
    sparse=at::cat({sparse,embedding},1);
  }
  at::Tensor dense;
  if (prompt.masks.defined()) {
    TORCH_CHECK(prompt.masks.sizes()==at::IntArrayRef({batch,1,4*grid_[0],4*grid_[1]}),"mask prompts require [B,1,4*grid_h,4*grid_w]; resize upstream masks before encoding");
    dense=prompt.masks.to(device_);
    for (int layer : {0,3,6}) {
      const auto prefix="mask_downscaling."+std::to_string(layer);
      dense=at::conv2d(dense,detail::weight(weights_,prefix+".weight"),detail::weight(weights_,prefix+".bias"),layer==6?at::IntArrayRef({1,1}):at::IntArrayRef({2,2}));
      if (layer!=6) {
        const auto norm="mask_downscaling."+std::to_string(layer+1);
        const auto mean=dense.mean(1,true),variance=(dense-mean).pow(2).mean(1,true);
        dense=(dense-mean)/(variance+1e-6).sqrt();
        dense=detail::weight(weights_,norm+".weight").unsqueeze(-1).unsqueeze(-1)*dense+detail::weight(weights_,norm+".bias").unsqueeze(-1).unsqueeze(-1);
        dense=at::gelu(dense);
      }
    }
  } else dense=detail::weight(weights_,"no_mask_embed.weight").reshape({1,256,1,1}).expand({batch,256,grid_[0],grid_[1]});
  const auto grid=at::ones({grid_[0],grid_[1]},options);
  const auto y=(grid.cumsum(0)-.5)/grid_[0],x=(grid.cumsum(1)-.5)/grid_[1];
  auto pos=position(at::stack({x,y},-1)).permute({2,0,1}).unsqueeze(0);
  return {std::move(sparse),std::move(dense),std::move(pos)};
}
}
