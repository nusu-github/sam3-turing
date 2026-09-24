#include "sam3/interactive_image.h"
#include "sam3/ops.h"
#include "ppm.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace {
void save(const std::filesystem::path& prefix,const sam3::InteractiveImageResult& result) {
  if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
  const auto packed=sam3::pack_masks(result.masks.flatten(0,1)).cpu().contiguous();
  auto path=prefix;path+=".masks.bin";std::ofstream masks(path,std::ios::binary);
  masks.write(reinterpret_cast<const char*>(packed.const_data_ptr<uint8_t>()),packed.numel());TORCH_CHECK(masks,"cannot write packed masks");
  const auto low=result.low_res_logits.cpu().to(at::kFloat).contiguous();
  path=prefix;path+=".lowres.f32.bin";std::ofstream logits(path,std::ios::binary);
  logits.write(reinterpret_cast<const char*>(low.const_data_ptr<float>()),low.numel()*sizeof(float));TORCH_CHECK(logits,"cannot write low-res logits");
  const auto scores=result.iou.cpu().to(at::kFloat).flatten();
  path=prefix;path+=".json";std::ofstream out(path);out<<std::setprecision(9);
  out<<"{\"height\":"<<result.masks.size(2)<<",\"width\":"<<result.masks.size(3)<<",\"count\":"<<scores.numel()<<",\"mask_row_bytes\":"<<packed.size(1)<<",\"iou\":[";
  for (int64_t i=0;i<scores.numel();++i) { if (i) out<<',';out<<scores[i].item<float>(); }
  out<<"]}\n";TORCH_CHECK(out,"cannot write interactive metadata");
}
}
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc>=7 && (argc-7)%3==0,"usage: sam3_interactive_image STORE sam3|sam3.1 cpu|cuda fp32|fp16|bf16_reference IMAGE.ppm OUTPUT_PREFIX [X Y LABEL]...");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);
    const auto pixels=sam3::cli::read_ppm(std::filesystem::u8path(argv[5]));
    sam3::InteractiveImageSession session(store,model,device);
    { const sam3::VisionEncoder vision(store,model,device);session.set_image(pixels,vision,mode); }
    sam3::InteractiveImagePrompt request;
    if (argc>7) {
      const auto count=(argc-7)/3;request.points=at::empty({count,2});request.labels=at::empty({count},at::TensorOptions().dtype(at::kInt));
      for (int i=0;i<count;++i) { request.points[i][0]=std::stod(argv[7+i*3]);request.points[i][1]=std::stod(argv[8+i*3]);request.labels[i]=std::stoi(argv[9+i*3]); }
    }
    const auto initial=session.predict(0,request);auto prefix=std::filesystem::u8path(argv[6]);auto initial_path=prefix;initial_path+=".initial";save(initial_path,initial);
    const auto best=initial.iou[0].argmax().item<int64_t>();request.masks=initial.low_res_logits[0][best].unsqueeze(0);
    sam3::InteractiveImageOptions options;options.multimask=false;
    const auto refined=session.predict(0,request,options);auto refined_path=prefix;refined_path+=".refined";save(refined_path,refined);
    std::cout<<"images="<<session.image_count()<<" initial="<<initial.masks.sizes()<<" refined="<<refined.masks.sizes()<<" reused_image_features=true\n";
    session.reset();TORCH_CHECK(session.image_count()==0,"reset did not clear images");
    bool rejected=false;try { session.predict(0,request); } catch (const c10::Error&) { rejected=true; }
    TORCH_CHECK(rejected,"prediction after reset should fail");return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
