#include "sam3/interactive_decoder.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
namespace sam3 {
InteractiveMaskDecoder::InteractiveMaskDecoder(const WeightStore& store,const std::string& model,at::Device device)
    :device_(device),sigmoid_iou_(model=="sam3") {
  TORCH_CHECK(model=="sam3" || model=="sam3.1","unknown interactive model");
  weights_=detail::load(store,model+(model=="sam3"?"/tracker.sam_mask_decoder.":"/tracker.model.interactive_sam_mask_decoder."),device);
  device_=detail::weight(weights_,"mask_tokens.weight").device();
}
at::Tensor InteractiveMaskDecoder::attention(const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,const std::string& name) const {
  auto q=detail::linear(weights_,query,name+".q_proj"),k=detail::linear(weights_,key,name+".k_proj"),v=detail::linear(weights_,value,name+".v_proj");
  const auto b=q.size(0),n=q.size(1),c=q.size(2);
  q=q.reshape({b,n,8,c/8}).transpose(1,2);
  k=k.reshape({b,k.size(1),8,c/8}).transpose(1,2);
  v=v.reshape({b,v.size(1),8,c/8}).transpose(1,2);
  auto out=at::scaled_dot_product_attention(q,k,v).transpose(1,2).reshape({b,n,c});
  return detail::linear(weights_,out,name+".out_proj");
}
std::vector<at::Tensor> InteractiveMaskDecoder::project_pyramid(const std::vector<at::Tensor>& features,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(features.size()==2,"expected two high-resolution feature maps");
  std::vector<at::Tensor> result;
  for (int i=0;i<2;++i) {
    TORCH_CHECK(features[i].dim()==4 && features[i].size(1)==256 && features[i].device()==device_,"invalid high-resolution visual features");
    const auto prefix="conv_s"+std::to_string(i);
    result.push_back(at::conv2d(features[i],detail::weight(weights_,prefix+".weight"),detail::weight(weights_,prefix+".bias")));
  }
  return result;
}
InteractiveMaskOutput InteractiveMaskDecoder::forward(const at::Tensor& image,const InteractiveEmbeddings& prompt,
    const std::vector<at::Tensor>& high,bool multimask,bool repeat_image,const std::string& mode,
    bool dynamic_stability,double delta,double threshold) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(image.dim()==4 && image.size(1)==256 && image.device()==device_,"interactive image features must be [B,256,H,W]");
  TORCH_CHECK(prompt.sparse.dim()==3 && prompt.sparse.size(2)==256 && prompt.sparse.device()==device_,"invalid sparse interactive embeddings");
  const auto batch=prompt.sparse.size(0),h=image.size(2),w=image.size(3);
  TORCH_CHECK(h>0 && w>0 && batch>0,"empty interactive batch/grid");
  TORCH_CHECK(image.size(0)==(repeat_image?1:batch),"repeat_image requires one source image, otherwise one per prompt");
  TORCH_CHECK(prompt.dense.sizes()==at::IntArrayRef({batch,256,h,w}) && prompt.dense.device()==device_,"invalid dense interactive embeddings");
  TORCH_CHECK(prompt.position.sizes()==at::IntArrayRef({1,256,h,w}) && prompt.position.device()==device_,"invalid interactive position grid");
  TORCH_CHECK(high.size()==2,"interactive decoder needs both projected high-resolution maps");
  for (int i=0;i<2;++i) {
    const auto scale=i==0?4:2,channels=i==0?32:64;
    TORCH_CHECK(high[i].dim()==4 && (high[i].size(0)==1 || high[i].size(0)==batch) && high[i].size(1)==channels && high[i].size(2)==h*scale && high[i].size(3)==w*scale && high[i].device()==device_,"invalid projected high-resolution map");
  }
  auto output_tokens=at::cat({detail::weight(weights_,"obj_score_token.weight"),detail::weight(weights_,"iou_token.weight"),detail::weight(weights_,"mask_tokens.weight")},0).unsqueeze(0).expand({batch,-1,-1});
  const auto tokens=at::cat({output_tokens,prompt.sparse},1);
  auto source=(repeat_image?at::repeat_interleave(image,batch,0):image)+prompt.dense;
  const auto pos=at::repeat_interleave(prompt.position,batch,0).flatten(2).permute({0,2,1});
  auto keys=source.flatten(2).permute({0,2,1}),queries=tokens;
  for (int layer=0;layer<2;++layer) {
    const auto name="transformer.layers."+std::to_string(layer);
    if (layer==0) queries=attention(queries,queries,queries,name+".self_attn");
    else { const auto q=queries+tokens;queries=queries+attention(q,q,queries,name+".self_attn"); }
    queries=detail::norm(weights_,queries,name+".norm1");
    queries=detail::norm(weights_,queries+attention(queries+tokens,keys+pos,keys,name+".cross_attn_token_to_image"),name+".norm2");
    const auto mlp=detail::linear(weights_,at::relu(detail::linear(weights_,queries,name+".mlp.lin1")),name+".mlp.lin2");
    queries=detail::norm(weights_,queries+mlp,name+".norm3");
    keys=detail::norm(weights_,keys+attention(keys+pos,queries+tokens,queries,name+".cross_attn_image_to_token"),name+".norm4");
  }
  queries=detail::norm(weights_,queries+attention(queries+tokens,keys+pos,keys,"transformer.final_attn_token_to_image"),"transformer.norm_final_attn");
  const auto all_tokens=queries.slice(1,2,6);
  auto pixel=keys.transpose(1,2).view({batch,256,h,w});
  pixel=at::conv_transpose2d(pixel,detail::weight(weights_,"output_upscaling.0.weight"),detail::weight(weights_,"output_upscaling.0.bias"),{2,2})+high[1];
  const auto mean=pixel.mean(1,true),variance=(pixel-mean).pow(2).mean(1,true);
  pixel=(pixel-mean)/(variance+1e-6).sqrt();
  pixel=at::gelu(detail::weight(weights_,"output_upscaling.1.weight").unsqueeze(-1).unsqueeze(-1)*pixel+detail::weight(weights_,"output_upscaling.1.bias").unsqueeze(-1).unsqueeze(-1));
  pixel=at::gelu(at::conv_transpose2d(pixel,detail::weight(weights_,"output_upscaling.3.weight"),detail::weight(weights_,"output_upscaling.3.bias"),{2,2})+high[0]);
  std::vector<at::Tensor> hyper;
  for (int i=0;i<4;++i) hyper.push_back(detail::mlp(weights_,all_tokens.select(1,i),"output_hypernetworks_mlps."+std::to_string(i),3));
  const auto all_masks=at::matmul(at::stack(hyper,1),pixel.view({batch,32,h*w*16})).view({batch,4,h*4,w*4});
  auto all_iou=detail::mlp(weights_,queries.select(1,1),"iou_prediction_head",3);
  if (sigmoid_iou_) all_iou=all_iou.sigmoid();
  const auto objects=detail::mlp(weights_,queries.select(1,0),"pred_obj_score_head",3);
  at::Tensor masks,iou,selected_tokens;
  if (multimask) {
    masks=all_masks.slice(1,1);iou=all_iou.slice(1,1);selected_tokens=all_tokens.slice(1,1);
  } else {
    masks=all_masks.slice(1,0,1);iou=all_iou.slice(1,0,1);selected_tokens=all_tokens.slice(1,0,1);
    if (dynamic_stability) {
      const auto best=all_iou.slice(1,1).argmax(-1)+1,ids=at::arange(batch,best.options());
      const auto best_masks=all_masks.index({ids,best}).unsqueeze(1),best_iou=all_iou.index({ids,best}).unsqueeze(1);
      const auto flat=masks.flatten(-2),area_i=(flat>delta).sum(-1).to(at::kFloat),area_u=(flat>-delta).sum(-1).to(at::kFloat);
      const auto stable=at::where(area_u>0,area_i/area_u,1.)>=threshold;
      masks=at::where(stable.unsqueeze(-1).unsqueeze(-1).expand_as(masks),masks,best_masks);
      iou=at::where(stable.expand_as(iou),iou,best_iou);
      // Upstream keeps token zero even when stability selects another mask.
    }
  }
  return {masks,iou,selected_tokens,objects,all_masks,all_iou,all_tokens};
}
}
