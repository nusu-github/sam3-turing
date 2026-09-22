#include "sam3/tracking_session.h"
#include "sam3/tracking_vision.h"
#include "sam3/ops.h"
#include "ppm.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace {
void binary(std::filesystem::path path,const at::Tensor& value) {
  const auto data=value.cpu().contiguous();std::ofstream out(path,std::ios::binary);
  out.write(static_cast<const char*>(data.const_data_ptr()),data.nbytes());TORCH_CHECK(out,"cannot write output: ",path.u8string());
}
void save(const std::filesystem::path& root,int64_t operation,int64_t number,const sam3::TrackingSessionOutput& result) {
  const auto name=std::to_string(operation)+"-"+std::to_string(number);
  const auto masks=result.masks.to(at::kFloat),packed=sam3::pack_masks((masks>0).squeeze(1));
  binary(root/(name+".masks.bin"),packed);binary(root/(name+".video.f32.bin"),masks);
  if(result.low_masks.defined())binary(root/(name+".low.f32.bin"),result.low_masks.to(at::kFloat));
  std::ofstream out(root/(name+".json"));out<<std::setprecision(9);
  out<<"{\"frame\":"<<result.index<<",\"height\":"<<masks.size(2)<<",\"width\":"<<masks.size(3)<<",\"mask_row_bytes\":"<<packed.size(1)<<",\"ids\":[";
  for(size_t i=0;i<result.object_ids.size();++i){if(i)out<<',';out<<result.object_ids[i];}
  out<<"],\"low_masks\":"<<(result.low_masks.defined()?"true":"false")<<",\"object_logits\":[";
  if(result.object_logits.defined()) {const auto scores=result.object_logits.cpu().to(at::kFloat).flatten();for(int64_t i=0;i<scores.numel();++i){if(i)out<<',';out<<scores[i].item<float>();}}
  out<<"]}\n";TORCH_CHECK(out,"cannot write metadata");
}
}
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==7,"usage: sam3_tracking_video STORE cpu|cuda fp32|fp16|bf16_reference FRAMES.txt COMMANDS.txt OUTPUT_DIRECTORY");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const at::Device device(argv[2]);const std::string mode=argv[3];
    const auto manifest=std::filesystem::u8path(argv[4]),commands=std::filesystem::u8path(argv[5]),root=std::filesystem::u8path(argv[6]);
    std::ifstream input(manifest);TORCH_CHECK(input,"cannot read frame manifest");std::vector<std::filesystem::path> frames;std::string line;
    while(std::getline(input,line)){if(!line.empty() && line.back()=='\r')line.pop_back();if(line.empty() || line[0]=='#')continue;auto path=std::filesystem::u8path(line);frames.push_back(path.is_absolute()?path:manifest.parent_path()/path);}
    TORCH_CHECK(!frames.empty(),"frame manifest is empty");
    const auto first=sam3::cli::read_ppm(frames.front());const auto height=first.size(1),width=first.size(2);
    const auto core=std::make_shared<sam3::Sam3TrackingFrame>(store,device);
    const auto vision=std::make_shared<sam3::VisionEncoder>(store,"sam3",device);
    const sam3::Sam3TrackingVision encoder(vision,core,device);int64_t encodes=0,outputs=0;
    sam3::TrackingSessionOptions options;options.offload_state=true;
    sam3::Sam3TrackingSession session(core,[&](int64_t index){
      auto pixels=sam3::cli::read_ppm(frames.at(index));TORCH_CHECK(pixels.size(1)==height && pixels.size(2)==width,"video frame dimensions changed");
      ++encodes;return encoder.encode_rgb(pixels,mode);
    },frames.size(),height,width,device,mode,options);
    std::filesystem::create_directories(root);std::ifstream script(commands);TORCH_CHECK(script,"cannot read commands");int64_t operation=0;
    while(std::getline(script,line)) {
      std::istringstream parser(line);std::string command;if(!(parser>>command) || command[0]=='#')continue;
      int64_t number=0;const auto emit=[&](const sam3::TrackingSessionOutput& out){save(root,operation,number++,out);++outputs;};
      if(command=="points") {
        int64_t frame,id,clear,normalized,previous;TORCH_CHECK(bool(parser>>frame>>id>>clear>>normalized>>previous),"points requires FRAME ID CLEAR NORMALIZED PREVIOUS [X Y LABEL]...");
        std::vector<float> coords;std::vector<int64_t> labels;float x,y;int64_t label;
        while(parser>>x){TORCH_CHECK(bool(parser>>y>>label),"incomplete point");coords.insert(coords.end(),{x,y});labels.push_back(label);}
        sam3::TrackingPoints prompt{at::tensor(coords).reshape({-1,2}),at::tensor(labels,at::kLong),{},bool(normalized)};
        emit(session.add_points(frame,id,prompt,clear,previous));
      }else if(command=="box") {
        int64_t frame,id,normalized;float x0,y0,x1,y1;TORCH_CHECK(bool(parser>>frame>>id>>normalized>>x0>>y0>>x1>>y1),"box requires FRAME ID NORMALIZED X0 Y0 X1 Y1");
        sam3::TrackingPoints prompt;prompt.box=at::tensor({x0,y0,x1,y1});prompt.normalized=normalized;emit(session.add_points(frame,id,prompt));
      }else if(command=="mask") {
        int64_t frame,id;std::string filename;TORCH_CHECK(bool(parser>>frame>>id>>std::quoted(filename)),"mask requires FRAME ID PPM_PATH");
        auto path=std::filesystem::u8path(filename);if(path.is_relative())path=commands.parent_path()/path;
        emit(session.add_mask(frame,id,sam3::cli::read_ppm(path)[0].to(at::kFloat)/255.));
      }else if(command=="preflight") {
        int64_t encode;TORCH_CHECK(bool(parser>>encode),"preflight requires ENCODE_MEMORY");session.preflight(encode);
      }else if(command=="propagate") {
        int64_t start,steps,reverse,encode,preflight,stop,cancel;TORCH_CHECK(bool(parser>>start>>steps>>reverse>>encode>>preflight>>stop>>cancel),"propagate requires START STEPS REVERSE ENCODE PREFLIGHT STOP_AFTER USE_CANCEL");
        sam3::TrackingPropagation request;if(start>=0)request.start=start;if(steps>=0)request.max_steps=steps;request.reverse=reverse;request.encode_memory=encode;request.preflight=preflight;
        session.propagate(request,[&](const auto& out){emit(out);if(stop>0 && number>=stop){if(cancel)session.cancel();else return false;}return true;});
      }else if(command=="clear") {
        int64_t frame,id;TORCH_CHECK(bool(parser>>frame>>id),"clear requires FRAME ID");emit(session.clear_input(frame,id));
      }else if(command=="remove") {
        int64_t id;TORCH_CHECK(bool(parser>>id),"remove requires ID");for(const auto& out:session.remove_object(id,true))emit(out);
      }else if(command=="reset")session.reset();
      else TORCH_CHECK(false,"unknown command: ",command);
      std::cout<<"operation="<<operation++<<" command="<<command<<" outputs="<<number<<" backbone_calls="<<encodes<<'\n';
    }
    std::cout<<"completed frames="<<frames.size()<<" outputs="<<outputs<<" backbone_calls="<<encodes<<" Python=none\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
