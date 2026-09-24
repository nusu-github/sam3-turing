#include "sam3/multiplex_decoder.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
namespace sam3 {
MultiplexMaskDecoder::MultiplexMaskDecoder(const WeightStore& store,at::Device device):device_(device) {
  weights_=detail::load(store,"sam3.1/tracker.model.sam_mask_decoder.",device);
  device_=detail::weight(weights_,"mask_tokens.weight").device();
  TORCH_CHECK(detail::weight(weights_,"mask_tokens.weight").sizes()==at::IntArrayRef({48,256}),"expected shipped SAM3.1 propagation weights");
}
at::Tensor MultiplexMaskDecoder::attention(const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,const std::string& name) const {
  auto q=detail::linear(weights_,query,name+".q_proj"),k=detail::linear(weights_,key,name+".k_proj"),v=detail::linear(weights_,value,name+".v_proj");
  const auto b=q.size(0),n=q.size(1),c=q.size(2);
  q=q.reshape({b,n,8,c/8}).transpose(1,2);
  k=k.reshape({b,k.size(1),8,c/8}).transpose(1,2);
  v=v.reshape({b,v.size(1),8,c/8}).transpose(1,2);
  auto out=at::scaled_dot_product_attention(q,k,v).transpose(1,2).reshape({b,n,c});
  return detail::linear(weights_,out,name+".out_proj");
}
std::vector<at::Tensor> MultiplexMaskDecoder::project_pyramid(const std::vector<at::Tensor>& features,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(features.size()==2,"expected two propagation high-resolution maps");
  std::vector<at::Tensor> result;
  for (int i=0;i<2;++i) {
    TORCH_CHECK(features[i].dim()==4 && features[i].size(1)==256 && features[i].device()==device_,"invalid propagation high-resolution features");
    const auto name="conv_s"+std::to_string(i);
    result.push_back(at::conv2d(features[i],detail::weight(weights_,name+".weight"),detail::weight(weights_,name+".bias")));
  }
  return result;
}
MultiplexMaskOutput MultiplexMaskDecoder::forward(const at::Tensor& image,const at::Tensor& position,
    const std::vector<at::Tensor>& high,const at::Tensor& extra,const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(image.dim()==4 && image.size(1)==256 && image.device()==device_,"propagation image features must be [B,256,H,W]");
  const auto batch=image.size(0),h=image.size(2),w=image.size(3);
  TORCH_CHECK(batch>0 && h>0 && w>0,"empty propagation batch/grid");
  TORCH_CHECK(position.sizes()==at::IntArrayRef({1,256,h,w}) && position.device()==device_,"invalid propagation position grid");
  TORCH_CHECK(high.size()==2,"propagation decoder needs both projected high-resolution maps");
  for (int i=0;i<2;++i) {
    const auto scale=i==0?4:2,channels=i==0?32:64;
    TORCH_CHECK(high[i].dim()==4 && (high[i].size(0)==1 || high[i].size(0)==batch) && high[i].size(1)==channels && high[i].size(2)==h*scale && high[i].size(3)==w*scale && high[i].device()==device_,"invalid projected propagation map");
  }
  auto header=at::cat({detail::weight(weights_,"obj_score_token.weight"),detail::weight(weights_,"iou_token.weight")},0).unsqueeze(0).expand({batch,-1,-1});
  auto masks=detail::weight(weights_,"mask_tokens.weight").unsqueeze(0).expand({batch,-1,-1});
  if (extra.defined()) {
    TORCH_CHECK(extra.dim()==3 && (extra.size(0)==1 || extra.size(0)==batch) && extra.size(1)==16 && extra.size(2)==256 && extra.device()==device_,"invalid per-object propagation embeddings");
    masks=(masks.view({batch,16,3,256})+extra.unsqueeze(2)).flatten(1,2);
  }
  const auto tokens=at::cat({header,masks},1);
  const auto pos=at::repeat_interleave(position,batch,0).flatten(2).permute({0,2,1});
  auto keys=image.flatten(2).permute({0,2,1}),queries=tokens;
  for (int layer=0;layer<2;++layer) {
    const auto name="transformer.layers."+std::to_string(layer);
    if (layer==0) queries=attention(queries,queries,queries,name+".self_attn");
    else {const auto q=queries+tokens;queries=queries+attention(q,q,queries,name+".self_attn");}
    queries=detail::norm(weights_,queries,name+".norm1");
    queries=detail::norm(weights_,queries+attention(queries+tokens,keys+pos,keys,name+".cross_attn_token_to_image"),name+".norm2");
    const auto mlp=detail::linear(weights_,at::relu(detail::linear(weights_,queries,name+".mlp.lin1")),name+".mlp.lin2");
    queries=detail::norm(weights_,queries+mlp,name+".norm3");
    keys=detail::norm(weights_,keys+attention(keys+pos,queries+tokens,queries,name+".cross_attn_image_to_token"),name+".norm4");
  }
  queries=detail::norm(weights_,queries+attention(queries+tokens,keys+pos,keys,"transformer.final_attn_token_to_image"),"transformer.norm_final_attn");
  const auto mask_tokens=queries.slice(1,32,80).view({batch,16,3,256});
  auto pixel=keys.transpose(1,2).view({batch,256,h,w});
  pixel=at::conv_transpose2d(pixel,detail::weight(weights_,"output_upscaling.0.weight"),detail::weight(weights_,"output_upscaling.0.bias"),{2,2})+high[1];
  const auto mean=pixel.mean(1,true),variance=(pixel-mean).pow(2).mean(1,true);
  pixel=(pixel-mean)/(variance+1e-6).sqrt();
  pixel=at::gelu(detail::weight(weights_,"output_upscaling.1.weight").unsqueeze(-1).unsqueeze(-1)*pixel+detail::weight(weights_,"output_upscaling.1.bias").unsqueeze(-1).unsqueeze(-1));
  pixel=at::gelu(at::conv_transpose2d(pixel,detail::weight(weights_,"output_upscaling.3.weight"),detail::weight(weights_,"output_upscaling.3.bias"),{2,2})+high[0]);
  std::vector<at::Tensor> hyper;
  for (int i=0;i<3;++i) hyper.push_back(detail::mlp(weights_,mask_tokens.select(2,i),"output_hypernetworks_mlps."+std::to_string(i),3));
  const auto all_masks=at::bmm(at::stack(hyper,2).flatten(1,2),pixel.view({batch,32,h*w*16})).view({batch,16,3,h*4,w*4});
  const auto iou=detail::mlp(weights_,queries.slice(1,16,32),"iou_prediction_head",3).view({batch,16,3});
  const auto objects=detail::mlp(weights_,queries.slice(1,0,16),"pred_obj_score_head",3);
  return {all_masks,iou,mask_tokens,objects};
}
MultiplexPropagationHeads::MultiplexPropagationHeads(const WeightStore& store,at::Device device)
    :decoder_(store,device),device_(device) {
  const std::string root="sam3.1/tracker.model.";
  for (const auto name:{"obj_ptr_proj","no_obj_ptr_linear"})
    for (auto& [key,value]:detail::load(store,root+name+".",device)) weights_.emplace(std::string(name)+"."+key,std::move(value));
  for (const auto name:{"output_valid_embed","output_invalid_embed","image_pe_layer.positional_encoding_gaussian_matrix"})
    weights_.emplace(name,store.read(root+name,device));
  device_=detail::weight(weights_,"output_valid_embed").device();
}
std::vector<at::Tensor> MultiplexPropagationHeads::project_pyramid(const std::vector<at::Tensor>& features,const std::string& mode) const {
  return decoder_.project_pyramid(features,mode);
}
at::Tensor MultiplexPropagationHeads::dense_position(const std::string& mode) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  const auto grid=at::ones({72,72},at::TensorOptions().device(device_).dtype(at::kFloat));
  const auto y=(grid.cumsum(0)-.5)/72,x=(grid.cumsum(1)-.5)/72;
  const auto phase=at::matmul(2*at::stack({x,y},-1)-1,detail::weight(weights_,"image_pe_layer.positional_encoding_gaussian_matrix"))*(2*std::acos(-1.));
  return at::cat({phase.sin(),phase.cos()},-1).permute({2,0,1}).unsqueeze(0);
}
VideoMaskOutput MultiplexPropagationHeads::forward(const MultiplexState& state,const at::Tensor& image,
    const std::vector<at::Tensor>& high,const std::string& mode,double threshold,bool attenuate) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(state.valid() && state.width()==16 && state.object_count()>0,"propagation requires live objects in 16-slot buckets");
  TORCH_CHECK(image.sizes()==at::IntArrayRef({state.bucket_count(),256,72,72}) && image.device()==device_,"propagation requires one full-grid feature per bucket");
  const auto valid=state.valid_object_mask().unsqueeze(-1).to(at::kFloat);
  const auto extra=valid*detail::weight(weights_,"output_valid_embed").unsqueeze(0)+(1-valid)*detail::weight(weights_,"output_invalid_embed").unsqueeze(0);
  const auto decoded=decoder_.forward(image,dense_position(mode),high,extra,mode);
  auto iou=state.demux(decoded.iou);
  const auto objects=state.demux(decoded.object_logits),tokens=state.demux(decoded.tokens),present=objects>threshold;
  const auto low=at::where(present.unsqueeze(-1).unsqueeze(-1),state.demux(decoded.masks),-1024.).to(at::kFloat);
  const auto full=at::upsample_bilinear2d(low,{1008,1008},false);
  if (attenuate) {
    const auto flat=low.flatten(-2),area_i=(flat>.05).sum(-1).to(at::kFloat),area_u=(flat>-.05).sum(-1).to(at::kFloat);
    iou=iou*at::where(area_u>0,area_i/area_u,1.);
  }
  const auto best=iou.argmax(-1),ids=at::arange(state.object_count(),best.options());
  const auto selected_low=low.index({ids,best}).unsqueeze(1),selected_full=full.index({ids,best}).unsqueeze(1);
  const auto pointer=detail::mlp(weights_,tokens.index({ids,best}),"obj_ptr_proj",3),probability=present.to(at::kFloat);
  const auto gated=probability*pointer+(1-probability)*detail::linear(weights_,pointer,"no_obj_ptr_linear");
  return {low,full,iou,selected_low,selected_full,gated,objects};
}
}
