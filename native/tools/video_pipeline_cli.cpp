#include "sam3/video_frame.h"
#include "sam3/multiplex_storage.h"
#include "sam3/text_encoder.h"
#include "sam3/tokenizer.h"
#include "sam3/ops.h"
#include "ppm.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
namespace {
void binary(const std::filesystem::path& path,const at::Tensor& tensor){const auto x=tensor.cpu().contiguous();std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(x.const_data_ptr()),x.nbytes());TORCH_CHECK(f,"cannot write tensor output");}
template<bool Mux> void run(const sam3::WeightStore& store,at::Device device,const std::string& mode,const std::vector<std::filesystem::path>& files,const sam3::GroundingPrompt& prompt,const std::filesystem::path& root,bool trace){
  const std::string model=Mux?"sam3.1":"sam3";const auto vision=std::make_shared<sam3::VisionEncoder>(store,model,device);const auto detector=std::make_shared<sam3::GroundingDetector>(store,model,device);
  using Core=std::conditional_t<Mux,sam3::Sam31TrackingFrame,sam3::Sam3TrackingFrame>;using Session=std::conditional_t<Mux,sam3::Sam31TrackingSession,sam3::Sam3TrackingSession>;using Options=std::conditional_t<Mux,sam3::MultiplexSessionOptions,sam3::TrackingSessionOptions>;
  const auto core=std::make_shared<Core>(store,device);const sam3::VideoFrameEncoder encoder(vision,detector,core,device);
  auto pixels=sam3::cli::read_ppm(files.front());const auto h=pixels.size(1),w=pixels.size(2);int64_t cached_index=-1,encodes=0;sam3::VideoFrameFeatures cached;
  const auto features=[&](int64_t index)->const sam3::VideoFrameFeatures&{if(cached_index!=index){auto rgb=sam3::cli::read_ppm(files.at(index));TORCH_CHECK(rgb.size(1)==h && rgb.size(2)==w,"video dimensions changed");cached=encoder.encode_rgb(rgb,mode);cached_index=index;++encodes;}return cached;};
  Options session_options;session_options.offload_state=true;
  if constexpr(!Mux)session_options.frame.temporal.select_by_score=true;
  if constexpr(Mux)session_options.history_directory=root/"history";
  std::vector<std::unique_ptr<Session>> sessions;
  std::function<std::unique_ptr<Session>()> factory=[&]{return std::make_unique<Session>(core,[&](int64_t index){if constexpr(Mux)return features(index).tracking;else return features(index).tracking.propagation;},files.size(),h,w,device,mode,session_options);};
  sam3::VideoDetectionOptions detection_options;detection_options.model=Mux?sam3::AssociationPolicy::Sam31:sam3::AssociationPolicy::Sam3;detection_options.nms=Mux?sam3::VideoNmsMode::Sam31Batched:sam3::VideoNmsMode::Greedy;detection_options.score_threshold=Mux?.4:.5;detection_options.use_iom=Mux;detection_options.boundary_filter=Mux;
  sam3::VideoUpdateOptions update;update.association.policy=update.recondition.policy=detection_options.model;update.association.new_detection_threshold=Mux?.65:.7;update.association.use_iom=Mux;update.association.iom_recondition_threshold=Mux?.5:.8;
  update.hotstart.delay=15;update.hotstart.unmatched_threshold=8;update.hotstart.duplicate_threshold=8;update.hotstart.suppress_only_within_hotstart=!Mux;
  if constexpr(!Mux){update.hotstart.min_keep_alive=-1;update.hotstart.max_keep_alive=30;update.hotstart.initial_keep_alive=30;}
  update.recondition.period=16;update.occlusion_threshold=.7;update.cleanup_area=Mux?0:16;update.confirmation_enabled=Mux;
  auto metadata=sam3::initialize_video_metadata(1,device);const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);std::filesystem::create_directories(root);
  for(size_t index=0;index<files.size();++index){
    const auto& visual=features(index);auto raw=encoder.detect(visual,prompt,mode);auto detected=sam3::postprocess_video_detections(raw.detection,detection_options);TORCH_CHECK(detected.size()==1,"this probe accepts one runtime text prompt");auto& detections=detected[0];
    std::vector<at::Tensor> masks,scores;std::vector<int64_t> ids;sam3::TrackingPropagation request;request.start=index;request.max_steps=0;request.encode_memory=false;
    for(auto& session:sessions)session->propagate(request,[&](const auto& value){ids.insert(ids.end(),value.object_ids.begin(),value.object_ids.end());masks.push_back(value.low_masks.squeeze(1));scores.push_back(value.object_logits.flatten());return true;});
    auto low=at::empty({0,288,288},opts),logits=at::empty({0},opts);
    if(!ids.empty()){const auto rows=sam3::video_memory_rows(ids,{metadata.object_ids()})[0];const auto order=at::tensor(rows,opts.dtype(at::kLong));low=sam3::clean_video_mask_scores(at::cat(masks).index_select(0,order).unsqueeze(1),update.cleanup_area).squeeze(1);logits=at::cat(scores).index_select(0,order);}
    auto plan=sam3::plan_video_update(index,false,detections,low,logits,metadata,update);sam3::execute_video_update(index,0,plan,detections,sessions,factory,update);auto output=sam3::build_video_outputs(plan,detections,h,w,update);sam3::finalize_video_scores(plan.metadata,index,plan.previous_ids,logits);metadata=std::move(plan.metadata);
    if constexpr(Mux){if(trace)for(size_t si=0;si<sessions.size();++si)for(const auto* history:{&sessions[si]->state().history.conditioning,&sessions[si]->state().history.tracked})for(const auto& stored:*history)if(stored.index==int64_t(index)){
      const auto frame=sam3::load_multiplex_frame(stored);const auto prefix=std::to_string(index)+".state"+std::to_string(si)+".";
      std::cout<<"trace frame="<<index<<" conditions="<<frame.conditioning_objects.size()<<" image_dtype="<<frame.image.scalar_type()<<" mask_dtype="<<(frame.memory_masks.defined()?frame.memory_masks.scalar_type():at::kFloat)<<"\n";
      for(const auto& [name,value]:std::vector<std::pair<std::string,at::Tensor>>{{"memory",frame.memory},{"position",frame.memory_position},{"image",frame.image},{"image_position",frame.image_position},{"pointer",frame.pointer},{"low",frame.masks.low_res_mask},{"high_input",frame.memory_masks},{"proxy",frame.memory_object_logits}})if(value.defined())binary(root/(prefix+name+".f32.bin"),value.to(at::kFloat));
    }}
    std::vector<at::Tensor> result_masks;std::vector<int64_t> output_ids;for(const auto& [id,mask]:output){output_ids.push_back(id);result_masks.push_back(mask.squeeze(0));}
    const auto stacked=result_masks.empty()?at::empty({0,h,w},opts.dtype(at::kBool)):at::stack(result_masks);const auto stem=std::to_string(index);binary(root/(stem+".masks.bin"),sam3::pack_masks(stacked));binary(root/(stem+".detector_logits.f32.bin"),raw.detection.logits.to(at::kFloat));binary(root/(stem+".tracker_low.f32.bin"),low.to(at::kFloat));
    std::ofstream json(root/(stem+".json"));json<<std::setprecision(9)<<"{\"frame\":"<<index<<",\"height\":"<<h<<",\"width\":"<<w<<",\"queries\":"<<raw.detection.logits.size(1)<<",\"tracked\":"<<metadata.object_ids().size()<<",\"ids\":[";
    for(size_t i=0;i<output_ids.size();++i){if(i)json<<',';json<<output_ids[i];}json<<"],\"scores\":[";for(size_t i=0;i<output_ids.size();++i){if(i)json<<',';json<<metadata.object_scores.at(output_ids[i]);}json<<"]}\n";TORCH_CHECK(json,"cannot write frame metadata");
    std::cout<<"frame="<<index<<" queries="<<raw.detection.logits.size(1)<<" candidates="<<detections.keep.sum().template item<int64_t>()<<" tracked="<<metadata.object_ids().size()<<" masks="<<output.size()<<" encodes="<<encodes<<"\n";
  }
  TORCH_CHECK(encodes==int64_t(files.size()),"sequential pipeline repeated a visual trunk evaluation");std::cout<<"shared-trunk detector/tracker pipeline completed; raw outputs, no Python\n";
}
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==9 || argc==10,"usage: sam3_video_pipeline_probe STORE sam3|sam3.1 DEVICE MODE FRAMES.txt BPE.gz PROMPT.txt OUTPUT [--trace]");TORCH_CHECK(argc!=10 || std::string(argv[9])=="--trace","unknown probe option");at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);TORCH_CHECK(model=="sam3" || model=="sam3.1","invalid model");
  const auto manifest=std::filesystem::u8path(argv[5]);std::ifstream input(manifest);TORCH_CHECK(input,"cannot read frame manifest");std::vector<std::filesystem::path> files;std::string line;
  while(std::getline(input,line)){if(!line.empty() && line.back()=='\r')line.pop_back();if(line.empty() || line[0]=='#')continue;auto p=std::filesystem::u8path(line);files.push_back(p.is_absolute()?p:manifest.parent_path()/p);}TORCH_CHECK(!files.empty(),"empty frame manifest");
  std::ifstream textfile(std::filesystem::u8path(argv[7]),std::ios::binary);TORCH_CHECK(textfile,"cannot read prompt");const std::string text((std::istreambuf_iterator<char>(textfile)),{});sam3::Tokenizer tokenizer(std::filesystem::u8path(argv[6]));
  sam3::GroundingPrompt prompt;const auto opts=at::TensorOptions().device(device);prompt.image_ids=prompt.text_ids=at::zeros({1},opts.dtype(at::kLong));
  {sam3::TextEncoder encoder(store,model,device);const auto encoded=encoder.forward([&]{std::vector<at::Tensor> rows;for(const auto& ids:tokenizer.tokenize(model=="sam3.1"?std::vector<std::string>{text,"visual","geometric"}:std::vector<std::string>{text,"visual"}))rows.push_back(at::tensor(ids,opts.dtype(at::kLong)));return at::stack(rows);}(),mode);prompt.text_padding=std::get<0>(encoded);prompt.text_features=std::get<1>(encoded);}
  const auto labels=at::empty({0,1},opts.dtype(at::kLong)),padding=at::empty({1,0},opts.dtype(at::kBool));prompt.geometry={at::empty({0,1,2},opts),labels,padding,at::empty({0,1,4},opts),labels,padding};
  if(model=="sam3")run<false>(store,device,mode,files,prompt,std::filesystem::u8path(argv[8]),argc==10);else run<true>(store,device,mode,files,prompt,std::filesystem::u8path(argv[8]),argc==10);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
