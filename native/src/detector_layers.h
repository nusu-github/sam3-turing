#pragma once
#include "sam3/weights.h"
#include <limits>
#include <cmath>
namespace sam3::detail {
using Weights = std::map<std::string,at::Tensor>;
inline Weights load(const WeightStore& store,const std::string& prefix,at::Device device) {
  Weights result;
  for (auto& [name,value] : store.read_prefix(prefix,device)) result.emplace(name.substr(prefix.size()),std::move(value));
  return result;
}
inline const at::Tensor& weight(const Weights& weights,const std::string& name) {
  const auto it = weights.find(name);
  TORCH_CHECK(it != weights.end(),"missing detector weight: ",name);
  return it->second;
}
inline at::Tensor linear(const Weights& w,const at::Tensor& x,const std::string& name) {
  return at::linear(x,weight(w,name+".weight"),weight(w,name+".bias"));
}
inline at::Tensor norm(const Weights& w,const at::Tensor& x,const std::string& name) {
  return at::layer_norm(x,{256},weight(w,name+".weight"),weight(w,name+".bias"),1e-5);
}
inline at::Tensor mlp(const Weights& w,at::Tensor x,const std::string& name,int layers) {
  for (int i = 0; i < layers; ++i) {
    x = linear(w,x,name+".layers."+std::to_string(i));
    if (i+1 < layers) x = at::relu(x);
  }
  return x;
}
enum class Projection { Separate, SharedKV, SharedQKV };
// Sequence-first MHA. Match packed projection choices and the source's choice
// of SDPA vs explicit bmm/softmax (nn.MultiheadAttention need_weights=True).
inline at::Tensor attention(const Weights& weights,const at::Tensor& query,const at::Tensor& key,
    const at::Tensor& value,const at::Tensor& padding,const std::string& prefix,
    Projection projection, bool explicit_softmax = false,const at::Tensor& bias = {}) {
  const auto length=query.size(0), batch=query.size(1), source=key.size(0);
  const auto& w=weight(weights,prefix+".in_proj_weight");
  const auto& b=weight(weights,prefix+".in_proj_bias");
  at::Tensor q,k,v;
  if (projection == Projection::SharedQKV) {
    const auto packed=at::linear(query,w,b);
    q=packed.slice(2,0,256);k=packed.slice(2,256,512);v=packed.slice(2,512,768);
  } else {
    q=at::linear(query,w.slice(0,0,256),b.slice(0,0,256));
    if (projection == Projection::SharedKV) {
      const auto kv=at::linear(key,w.slice(0,256,768),b.slice(0,256,768));
      k=kv.slice(2,0,256);v=kv.slice(2,256,512);
    } else {
      k=at::linear(key,w.slice(0,256,512),b.slice(0,256,512));
      v=at::linear(value,w.slice(0,512,768),b.slice(0,512,768));
    }
  }
  q=q.contiguous().view({length,batch*8,32}).transpose(0,1);
  k=k.contiguous().view({source,batch*8,32}).transpose(0,1);
  v=v.contiguous().view({source,batch*8,32}).transpose(0,1);
  std::optional<at::Tensor> mask;
  if (bias.defined()) mask=bias.view({batch*8,length,source});
  if (padding.defined()) {
    const auto expanded=padding.view({batch,1,1,source}).expand({batch,8,1,source}).reshape({batch*8,1,source});
    const auto pm=at::zeros({batch*8,1,source},explicit_softmax ? query.options() : q.options())
        .masked_fill_(expanded,-std::numeric_limits<float>::infinity());
    mask=mask ? *mask+pm : pm;
  }
  at::Tensor output;
  if (explicit_softmax) {
    const auto scaled=q*std::sqrt(1.0/32);
    const auto scores=mask ? at::baddbmm(*mask,scaled,k.transpose(1,2)) : at::bmm(scaled,k.transpose(1,2));
    output=at::bmm(at::softmax(scores,-1),v).transpose(0,1).contiguous().view({length*batch,256});
  } else {
    if (mask) mask=mask->view({batch,8,-1,source});
    output=at::scaled_dot_product_attention(q.view({batch,8,length,32}),k.view({batch,8,source,32}),
        v.view({batch,8,source,32}),mask).permute({2,0,1,3}).contiguous().view({length*batch,256});
  }
  return linear(weights,output,prefix+".out_proj").view({length,batch,256});
}
inline void check_mode(const std::string& mode) {
  TORCH_CHECK(mode=="fp32" || mode=="fp16" || mode=="bf16_reference","unknown detector precision mode");
}
}
