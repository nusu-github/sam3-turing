#include "sam3/geometry_encoder.h"
#include "sam3/ops.h"
#include "sam3/autocast.h"
#include <c10/core/InferenceMode.h>
#include <limits>
namespace sam3 {
namespace {
std::tuple<at::Tensor,at::Tensor> concat_padded(const at::Tensor& a,const at::Tensor& am,
                                             const at::Tensor& b,const at::Tensor& bm) {
  const auto batch = a.size(1), width = a.size(2), length = a.size(0)+b.size(0);
  const auto na = am.logical_not().sum(1), nb = bm.logical_not().sum(1);
  const auto mask = at::arange(length,bm.options().dtype(at::kLong)).unsqueeze(0) >= (na+nb).unsqueeze(1);
  auto result = at::zeros({length,batch,width},b.options());
  result.slice(0,0,a.size(0)).copy_(a);
  const auto index = at::arange(b.size(0),na.options()).unsqueeze(1)+na.unsqueeze(0);
  result.scatter_(0,index.unsqueeze(2).expand({-1,-1,width}),b);
  return {result,mask};
}
at::Tensor coordinate_embedding(const at::Tensor& coordinate) {
  const auto index = at::arange(128,coordinate.options().dtype(at::kFloat));
  const auto denominator = at::pow(10000.,2*at::floor_divide(index,2)/128);
  const auto phase = (coordinate.flatten()*6.28318530717958647692).unsqueeze(1)/denominator;
  return at::stack({phase.slice(1,0,128,2).sin(),phase.slice(1,1,128,2).cos()},2).flatten(1);
}
void check_prompt(const at::Tensor& coords,const at::Tensor& labels,const at::Tensor& padding,
                  int64_t width,int64_t batch,at::Device device) {
  TORCH_CHECK(coords.dim() == 3 && coords.size(1) == batch && coords.size(2) == width && coords.scalar_type() == at::kFloat,
              "geometry coordinates must be float32 [N,B,2/4]");
  const auto n = coords.size(0);
  TORCH_CHECK(labels.sizes() == at::IntArrayRef({n,batch}) &&
              (labels.scalar_type() == at::kLong || labels.scalar_type() == at::kInt || labels.scalar_type() == at::kBool),
              "geometry labels must be bool/int32/int64 [N,B]");
  TORCH_CHECK(padding.sizes() == at::IntArrayRef({batch,n}) && padding.scalar_type() == at::kBool,
              "geometry padding must be bool [B,N]");
  TORCH_CHECK(coords.device() == device && labels.device() == device && padding.device() == device,
              "geometry prompt device mismatch");
  TORCH_CHECK(at::isfinite(coords).all().item<bool>() && ((labels == 0) | (labels == 1)).all().item<bool>(),
              "geometry requires finite coordinates and binary labels");
  if (n > 1) TORCH_CHECK((padding.slice(1,0,n-1) & padding.slice(1,1,n).logical_not()).any().item<bool>() == false,
                         "geometry prompts must be right-padded");
}
}
GeometryEncoder::GeometryEncoder(const WeightStore& store,const std::string& model,at::Device device):device_(device) {
  const auto prefix = model+"/detector.geometry_encoder.";
  for (auto& [name,value] : store.read_prefix(prefix,device)) weights_.emplace(name.substr(prefix.size()),std::move(value));
  TORCH_CHECK(weight("boxes_pool_project.weight").sizes() == at::IntArrayRef({256,256,7,7}),"unexpected geometry pooling weights");
  device_ = weight("boxes_pool_project.weight").device();
  for (int i = 0; i < 3; ++i) weight("encode."+std::to_string(i)+".self_attn.in_proj_weight");
}
const at::Tensor& GeometryEncoder::weight(const std::string& name) const {
  const auto it = weights_.find(name);
  TORCH_CHECK(it != weights_.end(),"missing geometry weight: ",name);
  return it->second;
}
at::Tensor GeometryEncoder::linear(const at::Tensor& x,const std::string& prefix) const {
  return at::linear(x,weight(prefix+".weight"),weight(prefix+".bias"));
}
at::Tensor GeometryEncoder::norm(const at::Tensor& x,const std::string& prefix) const {
  return at::layer_norm(x,{256},weight(prefix+".weight"),weight(prefix+".bias"),1e-5);
}
at::Tensor GeometryEncoder::attention(const at::Tensor& query,const at::Tensor& key,
    const at::Tensor& value,const at::Tensor& padding,const std::string& prefix,bool self) const {
  const auto length = query.size(0), batch = query.size(1), source = key.size(0);
  const auto& w = weight(prefix+".in_proj_weight");
  const auto& b = weight(prefix+".in_proj_bias");
  at::Tensor q,k,v;
  if (self) {
    const auto packed = at::linear(query,w,b);
    q = packed.slice(2,0,256); k = packed.slice(2,256,512); v = packed.slice(2,512,768);
  } else {
    q = at::linear(query,w.slice(0,0,256),b.slice(0,0,256));
    k = at::linear(key,w.slice(0,256,512),b.slice(0,256,512));
    v = at::linear(value,w.slice(0,512,768),b.slice(0,512,768));
  }
  q = q.contiguous().view({length,batch*8,32}).transpose(0,1).view({batch,8,length,32});
  k = k.contiguous().view({source,batch*8,32}).transpose(0,1).view({batch,8,source,32});
  v = v.contiguous().view({source,batch*8,32}).transpose(0,1).view({batch,8,source,32});
  std::optional<at::Tensor> mask;
  if (padding.defined()) {
    const auto expanded = padding.view({batch,1,1,source}).expand({batch,8,1,source});
    mask = at::zeros({batch,8,1,source},q.options()).masked_fill_(expanded,-std::numeric_limits<float>::infinity());
  }
  const auto output = at::scaled_dot_product_attention(q,k,v,mask).permute({2,0,1,3}).contiguous().view({length*batch,256});
  return linear(output,prefix+".out_proj").view({length,batch,256});
}
std::tuple<at::Tensor,at::Tensor> GeometryEncoder::forward(const at::Tensor& image,
    const at::Tensor& positions,const GeometryPrompt& prompt,const std::string& mode) const {
  c10::InferenceMode inference;
  TORCH_CHECK(mode == "fp32" || mode == "fp16" || mode == "bf16_reference","unknown geometry precision mode");
  AutocastGuard autocast(device_.type(),mode != "fp32",mode == "fp16" ? at::kHalf : at::kBFloat16);
  TORCH_CHECK(image.dim() == 4 && image.size(1) == 256 && image.size(0) > 0 && image.size(2) > 0 && image.size(3) > 0,
              "geometry image must be [B,256,H,W]");
  TORCH_CHECK(image.device() == device_ && positions.device() == device_ && positions.sizes() == image.sizes(),
              "geometry image/position shape or device mismatch");
  const auto batch = image.size(0), h = image.size(2), w = image.size(3);
  check_prompt(prompt.points,prompt.point_labels,prompt.point_padding,2,batch,device_);
  check_prompt(prompt.boxes,prompt.box_labels,prompt.box_padding,4,batch,device_);
  const auto memory = image.flatten(2).permute({2,0,1});
  const auto pos = positions.flatten(2).permute({2,0,1});
  const auto normalized = norm(memory,"img_pre_norm").permute({1,2,0}).view({batch,256,h,w});
  const auto np = prompt.points.size(0), nb = prompt.boxes.size(0);
  const auto& points = prompt.points;
  auto pe = linear(points,"points_direct_project");
  const auto grid = points.transpose(0,1).unsqueeze(2)*2-1;
  const auto sampled_points = at::grid_sampler(normalized,grid,0,0,false).squeeze(-1).permute({2,0,1});
  pe = pe+linear(sampled_points,"points_pool_project");
  const auto px = coordinate_embedding(points.select(2,0)).view({np,batch,128});
  const auto py = coordinate_embedding(points.select(2,1)).view({np,batch,128});
  pe = pe+linear(at::cat({px,py},-1),"points_pos_enc_project");
  pe = at::embedding(weight("label_embed.weight"),prompt.point_labels.to(at::kLong))+pe;
  const auto& boxes = prompt.boxes;
  auto be = linear(boxes,"boxes_direct_project");
  const auto cx = boxes.select(2,0), cy = boxes.select(2,1), bw = boxes.select(2,2), bh = boxes.select(2,3);
  auto xyxy = at::stack({cx-.5*bw,cy-.5*bh,cx+.5*bw,cy+.5*bh},-1);
  xyxy = xyxy*at::tensor({double(w),double(h),double(w),double(h)},boxes.options()).view({1,1,4});
  const auto ids = at::arange(batch,boxes.options()).unsqueeze(1).expand({batch,nb}).reshape({batch*nb,1});
  const auto rois = at::cat({ids,xyxy.transpose(0,1).reshape({batch*nb,4})},1);
  const auto sampled_boxes = roi_align(normalized,rois);
  const auto pooled_boxes = at::conv2d(sampled_boxes,weight("boxes_pool_project.weight"),weight("boxes_pool_project.bias"));
  be = be+pooled_boxes.view({batch,nb,256}).transpose(0,1);
  const auto bx = coordinate_embedding(cx), by = coordinate_embedding(cy);
  const auto boxpos = at::cat({by,bx,bh.flatten().unsqueeze(1),bw.flatten().unsqueeze(1)},1).view({nb,batch,258});
  be = be+linear(boxpos,"boxes_pos_enc_project");
  be = at::embedding(weight("label_embed.weight"),prompt.box_labels.to(at::kLong))+be;
  auto [encoded,padding] = concat_padded(pe,prompt.point_padding,be,prompt.box_padding);
  const auto cls = weight("cls_embed.weight").view({1,1,256}).repeat({1,batch,1});
  std::tie(encoded,padding) = concat_padded(encoded,padding,cls,at::zeros({batch,1},padding.options()));
  encoded = norm(linear(encoded,"final_proj"),"norm");
  for (int i = 0; i < 3; ++i) {
    const auto prefix = "encode."+std::to_string(i);
    const auto normalized = norm(encoded,prefix+".norm1");
    encoded = encoded+attention(normalized,normalized,normalized,padding,prefix+".self_attn",true);
    encoded = encoded+attention(norm(encoded,prefix+".norm2"),memory+pos,memory,at::Tensor(),prefix+".cross_attn_image",false);
    encoded = encoded+linear(at::relu(linear(norm(encoded,prefix+".norm3"),prefix+".linear1")),prefix+".linear2");
  }
  return {norm(encoded,"encode_norm"),padding};
}
}
