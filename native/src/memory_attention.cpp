#include "sam3/memory_attention.h"
#include "sam3/autocast.h"
#include "detector_layers.h"
#include <c10/core/InferenceMode.h>
namespace sam3 {
namespace {
at::Tensor axial_frequencies(int64_t size,int64_t dim,at::Device device) {
  const auto options=at::TensorOptions().device(device).dtype(at::kFloat);
  const auto freq=1./at::pow(10000.,at::arange(0,dim,4,options)/dim);
  const auto t=at::arange(size*size,options);
  const auto x=t.remainder(size)*1.+0.,y=t.floor_divide(size)*1.+0.;
  const auto fx=at::outer(x,freq),fy=at::outer(y,freq);
  return at::cat({at::polar(at::ones_like(fx),fx),at::polar(at::ones_like(fy),fy)},-1);
}
at::Tensor rotate(const at::Tensor& value,const at::Tensor& frequencies) {
  const auto complex=at::view_as_complex(value.to(at::kFloat).reshape({value.size(0),value.size(1),value.size(2),-1,2}));
  return at::view_as_real(complex*frequencies).flatten(3).to(value.scalar_type());
}
}
MemoryAttention::MemoryAttention(const WeightStore& store,const std::string& model,at::Device device)
    :device_(device),multiplex_(model=="sam3.1") {
  TORCH_CHECK(model=="sam3" || multiplex_,"unknown memory attention model");
  weights_=detail::load(store,model+(multiplex_?"/tracker.model.transformer.encoder.":"/tracker.transformer.encoder."),device);
  device_=detail::weight(weights_,"norm.weight").device();
  c10::InferenceMode inference;
  frequencies_=axial_frequencies(72,multiplex_?32:256,device_);
}
at::Tensor MemoryAttention::attention(const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,
    const at::Tensor& frequencies,int64_t pointer_tokens,bool repeat) const {
  const int64_t heads=multiplex_?8:1;
  const auto b=query.size(0),n=query.size(1),m=key.size(1),dim=query.size(2)/heads;
  TORCH_CHECK(key.size(0)==b && value.size(0)==b && value.size(1)==m,"invalid memory attention batch");
  TORCH_CHECK(pointer_tokens>=0 && pointer_tokens<=m,"invalid pointer token count");
  const auto spatial=m-pointer_tokens;
  TORCH_CHECK(spatial==0 || (repeat?spatial%n==0:spatial==n),"spatial memory must contain full query grids");
  auto q=query.reshape({b,n,heads,dim}).transpose(1,2);
  auto k=key.reshape({b,m,heads,dim}).transpose(1,2);
  const auto v=value.reshape({b,m,heads,dim}).transpose(1,2);
  const auto freq=frequencies.view({1,1,n,dim/2});
  q=rotate(q,freq);
  if (spatial>0) k.slice(2,0,spatial).copy_(rotate(k.slice(2,0,spatial),repeat?freq.repeat({1,1,spatial/n,1}):freq));
  // Let LibTorch choose its available precompiled backend. Do not force the
  // source SAM3.1 Flash-only context: Turing needs an available fallback.
  return at::scaled_dot_product_attention(q,k,v).transpose(1,2).reshape({b,n,heads*dim});
}
at::Tensor MemoryAttention::forward(const at::Tensor& source,const at::Tensor& source_position,
    const at::Tensor& memory,const at::Tensor& memory_position,int64_t pointers,const std::string& mode,
    const at::Tensor& image,const at::Tensor& memory_image,const at::Tensor& memory_image_position,std::vector<at::Tensor>* trace) const {
  c10::InferenceMode inference;detail::check_mode(mode);
  AutocastGuard autocast(device_.type(),mode!="fp32",mode=="fp16"?at::kHalf:at::kBFloat16);
  TORCH_CHECK(source.dim()==3 && source.size(0)>0 && source.size(1)>0 && source.size(2)==256 && source.device()==device_,"source requires [grid_tokens,batch,256]");
  const auto check_position=[&](const at::Tensor& position,int64_t tokens,int64_t channels) {
    return position.dim()==3 && position.size(0)==tokens && (position.size(1)==1 || position.size(1)==source.size(1)) && position.size(2)==channels && position.device()==device_;
  };
  TORCH_CHECK(check_position(source_position,source.size(0),256),"invalid source positions");
  TORCH_CHECK(memory.dim()==3 && memory.size(0)>0 && memory.size(1)==source.size(1) && memory.size(2)==(multiplex_?256:64) && memory.device()==device_,"invalid memory shape/device");
  TORCH_CHECK(check_position(memory_position,memory.size(0),memory.size(2)),"invalid memory positions");
  TORCH_CHECK(pointers>=0 && pointers<=memory.size(0),"invalid object pointer count");
  const auto side=static_cast<int64_t>(std::sqrt(source.size(0)));
  TORCH_CHECK(side*side==source.size(0),"memory attention requires a square spatial query grid");
  const auto frequencies=side==72?frequencies_:axial_frequencies(side,multiplex_?32:256,device_);
  auto output=(source+.1*source_position).transpose(0,1);
  const auto mem=memory.transpose(0,1),pos=memory_position.transpose(0,1);
  at::Tensor image_batch,memory_image_batch,memory_image_pos_batch;
  if (multiplex_) {
    TORCH_CHECK(image.dim()==3 && image.size(0)==source.size(0) && (image.size(1)==1 || image.size(1)==source.size(1)) && image.size(2)==256 && image.device()==device_,"invalid SAM3.1 image stream");
    TORCH_CHECK(memory_image.dim()==3 && memory_image.size(1)==image.size(1) && memory_image.size(2)==256 && memory_image.device()==device_,"invalid SAM3.1 memory image stream");
    TORCH_CHECK(check_position(memory_image_position,memory_image.size(0),256),"invalid memory image positions");
    image_batch=image.transpose(0,1);memory_image_batch=memory_image.transpose(0,1);memory_image_pos_batch=memory_image_position.transpose(0,1);
    if (memory_image.size(0)!=memory.size(0)) {
      TORCH_CHECK(memory.size(0)-memory_image.size(0)==pointers,"memory image length must omit exactly the pointer tokens");
      memory_image_batch=at::cat({memory_image_batch,at::zeros({image.size(1),pointers,256},memory_image.options())},1);
      // Source uses shared image positions when appending pointer positions.
      TORCH_CHECK(memory_image_position.size(1)==1,"source pointer-position padding requires shared memory image positions");
      memory_image_pos_batch=at::cat({memory_image_pos_batch,pos.slice(0,0,1).slice(1,pos.size(1)-pointers)},1);
    }
  }
  if (trace) trace->clear();
  for (int layer=0;layer<4;++layer) {
    const auto name="layers."+std::to_string(layer);
    const auto normalized=detail::norm(weights_,output,name+".norm1");
    const auto prefix=name+(multiplex_?".self_attn_":".self_attn.");
    const auto q=detail::linear(weights_,normalized,prefix+"q_proj"),k=detail::linear(weights_,normalized,prefix+"k_proj"),v=detail::linear(weights_,normalized,prefix+"v_proj");
    output=output+detail::linear(weights_,attention(q,k,v,frequencies,0,false),prefix+"out_proj");
    const auto cross_norm=detail::norm(weights_,output,name+".norm2");
    at::Tensor cross_q,cross_k,cross_v;std::string cross_prefix;
    if (multiplex_) {
      cross_prefix=name+".cross_attn_";
      cross_q=detail::linear(weights_,image_batch,name+".image_cross_attn_q_proj")+detail::linear(weights_,cross_norm,cross_prefix+"q_proj");
      cross_k=detail::linear(weights_,memory_image_batch,name+".image_cross_attn_k_proj")+detail::linear(weights_,mem,cross_prefix+"k_proj");
      cross_k=cross_k+memory_image_pos_batch;
      cross_v=detail::linear(weights_,mem,cross_prefix+"v_proj");
    } else {
      cross_prefix=name+".cross_attn_image.";
      cross_q=detail::linear(weights_,cross_norm,cross_prefix+"q_proj");
      cross_k=detail::linear(weights_,mem+pos,cross_prefix+"k_proj");
      cross_v=detail::linear(weights_,mem,cross_prefix+"v_proj");
    }
    output=output+detail::linear(weights_,attention(cross_q,cross_k,cross_v,frequencies,pointers,true),cross_prefix+"out_proj");
    auto mlp=detail::linear(weights_,detail::norm(weights_,output,name+".norm3"),name+".linear1");
    mlp=multiplex_?at::gelu(mlp):at::relu(mlp);
    output=output+detail::linear(weights_,mlp,name+".linear2");
    if (trace) trace->push_back(output.transpose(0,1));
  }
  return detail::norm(weights_,output,"norm").transpose(0,1);
}
}
