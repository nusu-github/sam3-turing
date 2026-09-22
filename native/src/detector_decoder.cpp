#include "sam3/detector.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
namespace {
at::Tensor inverse_sigmoid(const at::Tensor& input) {
  const auto x=input.clamp(0,1);
  return (x.clamp_min(1e-3)/(1-x).clamp_min(1e-3)).log();
}
at::Tensor sine_box_embedding(const at::Tensor& boxes) {
  const auto index=at::arange(128,boxes.options().dtype(at::kFloat));
  const auto divisor=at::pow(10000.,2*at::floor_divide(index,2)/128);
  std::vector<at::Tensor> channels;
  for (int i : {1,0,2,3}) {
    const auto phase=(boxes.select(2,i)*6.28318530717958647692).unsqueeze(2)/divisor;
    channels.push_back(at::stack({phase.slice(2,0,128,2).sin(),phase.slice(2,1,128,2).cos()},3).flatten(2));
  }
  return at::cat(channels,2);
}
}
DetectorDecoder::DetectorDecoder(const WeightStore& store,const std::string& model,at::Device device)
    :weights_(detail::load(store,model+"/detector.transformer.decoder.",device)),device_(device) {
  device_=detail::weight(weights_,"query_embed.weight").device();
  TORCH_CHECK(detail::weight(weights_,"query_embed.weight").sizes()==at::IntArrayRef({200,256}),"expected all 200 detector queries");
  for (int i=0;i<6;++i) detail::weight(weights_,"layers."+std::to_string(i)+".self_attn.in_proj_weight");
}
at::Tensor DetectorDecoder::relative_position_bias(const at::Tensor& boxes,int64_t h,int64_t w) const {
  const auto cx=boxes.select(2,0),cy=boxes.select(2,1),bw=boxes.select(2,2),bh=boxes.select(2,3);
  const auto xyxy=at::stack({cx-.5*bw,cy-.5*bh,cx+.5*bw,cy+.5*bh},-1).transpose(0,1);
  const auto batch=xyxy.size(0),queries=xyxy.size(1);
  // The constructor precomputes the standard 72x72 grid with host integers.
  // Other grids are generated from spatial_shapes device scalars in upstream.
  // CUDA scalar division uses reciprocal multiplication; tensor division has
  // different rounding, which can propagate through the half-precision RPB MLP.
  auto coords_h=at::arange(h,boxes.options().dtype(at::kFloat));
  auto coords_w=at::arange(w,boxes.options().dtype(at::kFloat));
  if (h==72 && w==72) { coords_h=coords_h/h;coords_w=coords_w/w; }
  else {
    coords_h=coords_h/at::scalar_tensor(h,boxes.options().dtype(at::kLong));
    coords_w=coords_w/at::scalar_tensor(w,boxes.options().dtype(at::kLong));
  }
  const auto flat=xyxy.reshape({-1,1,4});
  auto dy=(coords_h.view({1,h,1})-flat.slice(2,1,4,2)).view({batch,queries,h,2});
  auto dx=(coords_w.view({1,w,1})-flat.slice(2,0,3,2)).view({batch,queries,w,2});
  dx=dx*8;dy=dy*8;
  dx=dx.sign()*(dx.abs()+1).log2()/3.;dy=dy.sign()*(dy.abs()+1).log2()/3.;
  dx=detail::mlp(weights_,dx,"boxRPB_embed_x",2);dy=detail::mlp(weights_,dy,"boxRPB_embed_y",2);
  return (dy.unsqueeze(3)+dx.unsqueeze(2)).flatten(2,3).permute({0,3,1,2}).contiguous();
}
DecoderFeatures DetectorDecoder::forward(const FusionFeatures& encoded,const at::Tensor& prompt_padding,const std::string& mode,std::map<std::string,at::Tensor>* trace) const {
  c10::InferenceMode inference;
  detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16" ? at::kHalf : at::kBFloat16);
  const auto& memory=encoded.memory;
  TORCH_CHECK(memory.dim()==3 && memory.size(1)>0 && memory.size(2)==256 && memory.device()==device_,"decoder memory must be [H*W,B,256]");
  const auto batch=memory.size(1);
  TORCH_CHECK(encoded.positions.sizes()==memory.sizes() && encoded.positions.device()==device_,"decoder position mismatch");
  TORCH_CHECK(encoded.prompt.dim()==3 && encoded.prompt.size(1)==batch && encoded.prompt.size(2)==256 && encoded.prompt.device()==device_,"decoder prompt mismatch");
  TORCH_CHECK(prompt_padding.sizes()==at::IntArrayRef({batch,encoded.prompt.size(0)}) && prompt_padding.scalar_type()==at::kBool && prompt_padding.device()==device_,"decoder prompt padding mismatch");
  TORCH_CHECK(encoded.spatial_shapes.sizes()==at::IntArrayRef({1,2}) && encoded.spatial_shapes.scalar_type()==at::kLong,"decoder expects one original feature level");
  const auto shape=encoded.spatial_shapes.cpu();
  const auto h=shape[0][0].item<int64_t>(),w=shape[0][1].item<int64_t>();
  TORCH_CHECK(h>0 && w>0 && memory.size(0)/h==w && memory.size(0)%h==0,"decoder memory spatial dimensions mismatch");
  TORCH_CHECK(encoded.valid_ratios.sizes()==at::IntArrayRef({batch,1,2}) && encoded.valid_ratios.device()==device_,"decoder valid ratios mismatch");
  at::Tensor image_padding;
  if (encoded.padding.defined()) {
    TORCH_CHECK(encoded.padding.sizes()==at::IntArrayRef({memory.size(0),batch}) && encoded.padding.scalar_type()==at::kBool && encoded.padding.device()==device_,"decoder image padding mismatch");
    image_padding=encoded.padding.transpose(0,1);
  }
  auto output=detail::weight(weights_,"query_embed.weight").unsqueeze(1).repeat({1,batch,1});
  auto references=detail::weight(weights_,"reference_points.weight").unsqueeze(1).repeat({1,batch,1}).sigmoid();
  auto presence=detail::weight(weights_,"presence_token.weight").unsqueeze(0).expand({1,batch,256});
  const auto ratio=at::cat({encoded.valid_ratios,encoded.valid_ratios},-1).unsqueeze(0);
  std::vector<at::Tensor> outputs,anchors,presence_logits;
  for (int i=0;i<6;++i) {
    anchors.push_back(references);
    const auto reference_input=(references.unsqueeze(2)*ratio).select(2,0);
    auto query_pos=detail::mlp(weights_,sine_box_embedding(reference_input),"ref_point_head",2);
    auto bias=relative_position_bias(references,h,w);
    const auto prefix="layers."+std::to_string(i);
    const auto record=[&](const std::string& name,const at::Tensor& value) { if (trace) (*trace)[prefix+"."+name]=value; };
    record("query_pos",query_pos);record("bias",bias);
    auto tgt=at::cat({presence,output},0);
    query_pos=at::cat({at::zeros_like(presence),query_pos},0);
    const auto q=tgt+query_pos;
    auto attended=detail::attention(weights_,q,q,tgt,at::Tensor(),prefix+".self_attn",detail::Projection::Separate,true);
    record("self_attn",attended);
    tgt=detail::norm(weights_,tgt+attended,prefix+".norm2");record("norm2",tgt);
    attended=detail::attention(weights_,tgt+query_pos,encoded.prompt,encoded.prompt,prompt_padding,
        prefix+".ca_text",detail::Projection::SharedKV,true);record("ca_text",attended);
    tgt=detail::norm(weights_,tgt+attended,prefix+".catext_norm");record("catext_norm",tgt);
    bias=at::cat({at::zeros_like(bias.slice(2,0,1)),bias},2);
    attended=detail::attention(weights_,tgt+query_pos,memory+encoded.positions,memory,image_padding,
        prefix+".cross_attn",detail::Projection::Separate,false,bias);record("cross_attn",attended);
    tgt=detail::norm(weights_,tgt+attended,prefix+".norm1");record("norm1",tgt);
    at::Tensor ffn;
    {
      // The source disables CUDA autocast specifically for the decoder FFN.
      std::optional<AutocastGuard> full_precision;
      if (device_.is_cuda()) full_precision.emplace(at::kCUDA,false,at::kFloat);
      ffn=detail::linear(weights_,at::relu(detail::linear(weights_,tgt,prefix+".linear1")),prefix+".linear2");
    }
    record("ffn",ffn);tgt=detail::norm(weights_,tgt+ffn,prefix+".norm3");record("norm3",tgt);
    presence=tgt.slice(0,0,1);output=tgt.slice(0,1);
    const auto normalized=detail::norm(weights_,output,"norm");
    references=(detail::mlp(weights_,normalized,"bbox_embed",3)+inverse_sigmoid(references)).sigmoid();
    outputs.push_back(normalized);
    // Upstream calls clamp() without assigning the return, so these logits are
    // intentionally not clamped here. Preserve observable model behavior.
    presence_logits.push_back(detail::mlp(weights_,detail::norm(weights_,presence,"presence_token_out_norm"),"presence_token_head",3).squeeze(-1));
  }
  return {at::stack(outputs),at::stack(anchors),at::stack(presence_logits),presence.clone()};
}
}
