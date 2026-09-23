#include "sam3/video_frame.h"
#include <torch/library.h>
namespace {
sam3::VideoDetectionOptions options(bool mux,int64_t nms,double score,double overlap,bool iom,bool boundary,bool allow,const std::string& mode){sam3::VideoDetectionOptions o;o.model=mux?sam3::AssociationPolicy::Sam31:sam3::AssociationPolicy::Sam3;o.nms=static_cast<sam3::VideoNmsMode>(nms);o.score_threshold=score;o.nms_threshold=overlap;o.use_iom=iom;o.boundary_filter=boundary;o.allow_new_detections=allow;o.policy_mode=mode;return o;}
}
TORCH_LIBRARY_FRAGMENT(sam3_native,m){
  m.def("video_mask_nms(Tensor scores, Tensor masks, int nms, float score, float overlap, bool iom, str mode='fp32') -> Tensor",[](const at::Tensor& scores,const at::Tensor& masks,int64_t nms,double score,double overlap,bool iom,const std::string& mode){return sam3::video_mask_nms(scores,masks,options(nms!=0,nms,score,overlap,iom,false,true,mode));});
  m.def("video_detections(Tensor logits, Tensor masks, Tensor boxes, bool multiplex, int nms, float score, float overlap, bool iom, bool boundary, bool allow, str mode='fp32') -> Tensor[][]",[](const at::Tensor& logits,const at::Tensor& masks,const at::Tensor& boxes,bool mux,int64_t nms,double score,double overlap,bool iom,bool boundary,bool allow,const std::string& mode){sam3::DetectionOutput raw;raw.logits=logits;raw.masks=masks;raw.boxes_xyxy=boxes;std::vector<std::vector<at::Tensor>> out;for(const auto& d:sam3::postprocess_video_detections(raw,options(mux,nms,score,overlap,iom,boundary,allow,mode)))out.push_back({d.masks,d.scores,d.boxes,d.keep});return out;});
  m.def("video_frame_features(str directory, str model, Tensor image, Tensor[] prompt, str mode) -> Dict(str, Tensor)",[](const std::string& directory,const std::string& model,const at::Tensor& image,const std::vector<at::Tensor>& fields,const std::string& mode){
    TORCH_CHECK(model=="sam3" || model=="sam3.1","invalid model");const auto device=image.device();sam3::WeightStore store(std::filesystem::u8path(directory));
    const auto vision=std::make_shared<sam3::VisionEncoder>(store,model,device);const auto detector=std::make_shared<sam3::GroundingDetector>(store,model,device);
    std::unique_ptr<sam3::VideoFrameEncoder> encoder;
    if(model=="sam3")encoder=std::make_unique<sam3::VideoFrameEncoder>(vision,detector,std::make_shared<sam3::Sam3TrackingFrame>(store,device),device);
    else encoder=std::make_unique<sam3::VideoFrameEncoder>(vision,detector,std::make_shared<sam3::Sam31TrackingFrame>(store,device),device);
    const auto features=image.dim()==3?encoder->encode_rgb(image,mode):encoder->encode_preprocessed(image,mode);c10::Dict<std::string,at::Tensor> out;
    for(size_t i=0;i<features.detection_pyramid.size();++i)out.insert("detection/"+std::to_string(i),features.detection_pyramid[i]);out.insert("position",features.detection_position);
    for(const auto& [name,p]:std::vector<std::pair<std::string,sam3::TrackingFeatures>>{{"interactive",features.tracking.interactive},{"propagation",features.tracking.propagation}}){out.insert(name+"/image",p.image);out.insert(name+"/position",p.position);out.insert(name+"/high0",p.high[0]);out.insert(name+"/high1",p.high[1]);}
    if(!fields.empty()){TORCH_CHECK(fields.size()==10,"grounding test needs ten prompt tensors");sam3::GroundingPrompt prompt{fields[0],fields[1],fields[2],fields[3],{fields[4],fields[5],fields[6],fields[7],fields[8],fields[9]}};const auto result=encoder->detect(features,prompt,mode);const auto& d=result.detection;out.insert("pred_logits",d.logits);out.insert("pred_masks",d.masks);out.insert("pred_boxes",d.boxes);out.insert("pred_boxes_xyxy",d.boxes_xyxy);out.insert("presence_logit_dec",d.presence_logits);out.insert("semantic_seg",d.semantic);out.insert("encoder_hidden_states",result.encoded.memory);}
    return out;
  });
}
