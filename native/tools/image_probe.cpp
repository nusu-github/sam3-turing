#include "sam3/vision_encoder.h"
#include "sam3/text_encoder.h"
#include "sam3/grounding.h"
#include "sam3/preprocess.h"
#include "sam3/image_results.h"
#include "sam3/ops.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cctype>

namespace {
std::string ppm_token(std::istream& in) {
  std::string token;
  while (in) {
    in>>std::ws;
    if (in.peek()!='#') { in>>token;return token; }
    in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
  }
  TORCH_CHECK(false,"truncated PPM header");
}
at::Tensor read_ppm(const std::filesystem::path& path) {
  std::ifstream in(path,std::ios::binary);
  TORCH_CHECK(in,"cannot open input image");
  TORCH_CHECK(ppm_token(in)=="P6","this development probe accepts binary P6 RGB PPM");
  const auto w=std::stoll(ppm_token(in)),h=std::stoll(ppm_token(in));
  TORCH_CHECK(ppm_token(in)=="255","PPM must have 8-bit RGB samples");
  const int delimiter=in.get();
  TORCH_CHECK(delimiter!=EOF && std::isspace(static_cast<unsigned char>(delimiter)),"invalid PPM separator");
  if (delimiter=='\r' && in.peek()=='\n') in.get();
  TORCH_CHECK(w>0 && h>0 && w<=std::numeric_limits<int64_t>::max()/3/h,"invalid PPM dimensions");
  auto pixels=at::empty({h,w,3},at::TensorOptions().dtype(at::kByte));
  in.read(reinterpret_cast<char*>(pixels.mutable_data_ptr<uint8_t>()),pixels.numel());
  TORCH_CHECK(in.gcount()==pixels.numel(),"truncated PPM pixels");
  return pixels.permute({2,0,1});
}
void write_result(const std::filesystem::path& prefix,const sam3::ImageResult& result,int64_t h,int64_t w) {
  if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
  auto json_path=prefix;json_path+=".json";
  auto masks_path=prefix;masks_path+=".masks.bin";
  auto packed=sam3::pack_masks(result.masks.squeeze(1)).cpu().contiguous();
  std::ofstream masks(masks_path,std::ios::binary);
  TORCH_CHECK(masks,"cannot create mask output");
  masks.write(reinterpret_cast<const char*>(packed.const_data_ptr<uint8_t>()),packed.numel());
  TORCH_CHECK(masks,"cannot write masks");
  const auto boxes=result.boxes.cpu().to(at::kFloat).contiguous();
  const auto scores=result.scores.cpu().to(at::kFloat).contiguous();
  const auto indices=result.query_indices.cpu().contiguous();
  std::ofstream out(json_path);TORCH_CHECK(out,"cannot create JSON output");
  out<<std::setprecision(9)<<"{\n  \"height\":"<<h<<",\"width\":"<<w<<",\"count\":"<<scores.numel()
     <<",\"mask_row_bytes\":"<<packed.size(1)<<",\n  \"mask_format\":\"per-mask little-endian bits, row-major pixels\",\n  \"detections\":[";
  for (int64_t i=0;i<scores.numel();++i) {
    if (i) out<<',';
    out<<"\n    {\"query\":"<<indices[i].item<int64_t>()<<",\"score\":"<<scores[i].item<float>()<<",\"box\":[";
    for (int j=0;j<4;++j) { if (j) out<<',';out<<boxes[i][j].item<float>(); }
    out<<"]}";
  }
  out<<"\n  ]\n}\n";TORCH_CHECK(out,"cannot write JSON");
}
}
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc>=9,"usage: sam3_image_probe STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference IMAGE.ppm OUTPUT_PREFIX THRESHOLD TOKEN_ID...");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const std::string model=argv[2],mode=argv[4];
    TORCH_CHECK(model=="sam3" || model=="sam3.1","unknown model");
    const at::Device device(argv[3]);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));
    auto pixels=read_ppm(std::filesystem::u8path(argv[5]));
    const auto h=pixels.size(1),w=pixels.size(2);
    std::vector<at::Tensor> pyramid;at::Tensor positions;
    {
      const sam3::VisionEncoder vision(store,model,device);
      auto features=vision.forward(sam3::preprocess_rgb(pixels.to(device)),mode,{"convs"});
      pyramid=std::move(features.pyramid.at("convs"));
      if (model=="sam3") pyramid.pop_back(); // Original SAM3 backbone scalp=1.
      positions=features.positions[pyramid.size()-1];
    }
    auto tokens=at::zeros({1,32},at::TensorOptions().dtype(at::kLong));
    TORCH_CHECK(argc-8<=32,"token sequence exceeds upstream context length");
    for (int i=8;i<argc;++i) tokens[0][i-8]=std::stoll(argv[i]);
    at::Tensor text,text_padding;
    {
      const sam3::TextEncoder encoder(store,model,device);
      const auto features=encoder.forward(tokens.to(device),mode);
      text_padding=std::get<0>(features);text=std::get<1>(features);
    }
    const auto options=at::TensorOptions().dtype(at::kFloat).device(device);
    const auto ids=at::zeros({1},options.dtype(at::kLong));
    const auto labels=at::empty({0,1},options.dtype(at::kLong));
    const auto padding=at::empty({1,0},options.dtype(at::kBool));
    sam3::GroundingPrompt prompt{ids,ids,text,text_padding,
        {at::empty({0,1,2},options),labels,padding,at::empty({0,1,4},options),labels,padding}};
    const sam3::GroundingDetector detector(store,model,device);
    const bool joint_scores=model=="sam3.1";
    const auto output=detector.forward(pyramid,positions,prompt,joint_scores,mode);
    const auto results=sam3::postprocess_image(output.detection,{h},{w},std::stod(argv[7]),!joint_scores,8,mode);
    write_result(std::filesystem::u8path(argv[6]),results[0],h,w);
    std::cout<<"queries="<<output.detection.logits.size(1)<<" masks="<<output.detection.masks.sizes()
             <<" detections="<<results[0].scores.numel()<<" original="<<h<<'x'<<w<<'\n';
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
