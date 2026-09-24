#include "sam3/video_frame.h"
#include "sam3/video_output.h"
#include "sam3/video_interaction.h"
#include "sam3/video_edit.h"
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
template<bool Mux> void run(const sam3::WeightStore& store,at::Device device,const std::string& mode,const std::vector<std::filesystem::path>& files,const sam3::GroundingPrompt& prompt,const std::filesystem::path& root,bool trace,bool partial_probe,bool edit_probe,bool edit_sequence,bool edit_trace){
  const std::string model=Mux?"sam3.1":"sam3";const auto vision=std::make_shared<sam3::VisionEncoder>(store,model,device);const auto detector=std::make_shared<sam3::GroundingDetector>(store,model,device);
  using Core=std::conditional_t<Mux,sam3::Sam31TrackingFrame,sam3::Sam3TrackingFrame>;using Session=std::conditional_t<Mux,sam3::Sam31TrackingSession,sam3::Sam3TrackingSession>;using Options=std::conditional_t<Mux,sam3::MultiplexSessionOptions,sam3::TrackingSessionOptions>;
  const auto core=std::make_shared<Core>(store,device);const sam3::VideoFrameEncoder encoder(vision,detector,core,device);
  auto pixels=sam3::cli::read_ppm(files.front());const auto h=pixels.size(1),w=pixels.size(2);int64_t cached_index=-1,encodes=0;sam3::VideoFrameFeatures cached;
  const auto features=[&](int64_t index)->const sam3::VideoFrameFeatures&{if(cached_index!=index){auto rgb=sam3::cli::read_ppm(files.at(index));TORCH_CHECK(rgb.size(1)==h && rgb.size(2)==w,"video dimensions changed");cached=encoder.encode_rgb(rgb,mode);cached_index=index;++encodes;}return cached;};
  Options session_options;session_options.offload_state=true;
  // SAM3.1 keeps corrections to tracked frames in non-conditioning history;
  // SAM3 explicitly promotes them. This affects the next frame's memories.
  if constexpr(Mux)session_options.all_edits_conditioning=false;
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
  sam3::VideoOutputBufferOptions output_options;output_options.frame_count=files.size();output_options.end_frame=files.size()-1;output_options.height=h;output_options.width=w;output_options.batch_size=Mux?16:1;
  sam3::VideoOutputBuffer output_buffer(output_options);
  sam3::VideoInteractionState interaction(detection_options.model,files.size(),h,w);std::map<int64_t,std::set<int64_t>> suppressed_per_frame;
  for(size_t index=0;index<files.size();++index){
    const auto& visual=features(index);auto raw=encoder.detect(visual,prompt,mode);auto detected=sam3::postprocess_video_detections(raw.detection,detection_options);TORCH_CHECK(detected.size()==1,"this probe accepts one runtime text prompt");auto& detections=detected[0];
    std::vector<at::Tensor> masks,scores;std::vector<int64_t> ids;sam3::TrackingPropagation request;request.start=index;request.max_steps=0;request.encode_memory=false;
    for(auto& session:sessions)session->propagate(request,[&](const auto& value){ids.insert(ids.end(),value.object_ids.begin(),value.object_ids.end());masks.push_back(value.low_masks.squeeze(1));scores.push_back(value.object_logits.flatten());return true;});
    auto low=at::empty({0,288,288},opts),logits=at::empty({0},opts);
    if(!ids.empty()){const auto rows=sam3::video_memory_rows(ids,{metadata.object_ids()})[0];const auto order=at::tensor(rows,opts.dtype(at::kLong));low=sam3::clean_video_mask_scores(at::cat(masks).index_select(0,order).unsqueeze(1),update.cleanup_area).squeeze(1);logits=at::cat(scores).index_select(0,order);}
    auto plan=sam3::plan_video_update(index,false,detections,low,logits,metadata,update);sam3::execute_video_update(index,0,plan,detections,sessions,factory,update);auto output=sam3::build_video_outputs(plan,detections,h,w,update);sam3::finalize_video_scores(plan.metadata,index,plan.previous_ids,logits);metadata=std::move(plan.metadata);
    sam3::VideoRawOutput final_raw;final_raw.frame=index;final_raw.masks=output;final_raw.scores=metadata.object_scores;final_raw.tracker_scores=metadata.frame_scores[index];final_raw.removed=plan.removed;
    final_raw.frame_stats={{"num_obj_tracked",int64_t(metadata.object_ids().size())},{"num_obj_dropped",0}};
    // GPU hotstart candidates are not source predictor output hide lists.
    if(metadata.host.suppressed.count(index))final_raw.suppressed=metadata.host.suppressed.at(index);
    final_raw.unconfirmed=std::set<int64_t>{};if(update.confirmation_enabled){const auto all_ids=metadata.object_ids();for(size_t i=0;i<all_ids.size();++i)if(metadata.confirmation.status[i]==1)final_raw.unconfirmed->insert(all_ids[i]);}
    suppressed_per_frame[index]=final_raw.suppressed;
    for(const auto& emitted:output_buffer.push(final_raw)){
      interaction.record(emitted.frame,emitted.output);
      const auto prefix=std::to_string(emitted.frame)+".final.";const auto& value=emitted.output;
      binary(root/(prefix+"ids.i64.bin"),value.ids);binary(root/(prefix+"scores.f32.bin"),value.probabilities);binary(root/(prefix+"boxes.f32.bin"),value.boxes_xywh);binary(root/(prefix+"masks.bin"),sam3::pack_masks(value.masks));
      std::ofstream timing(root/(prefix+"json"));timing<<"{\"emitted_at\":"<<index<<",\"count\":"<<value.ids.numel()<<"}\n";TORCH_CHECK(timing,"cannot write final metadata");
    }
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
  TORCH_CHECK(output_buffer.pending()==0,"final output buffer was not drained");
  TORCH_CHECK(encodes==int64_t(files.size()),"sequential pipeline repeated a visual trunk evaluation");
  if(partial_probe){
    TORCH_CHECK(files.size()>18 && !metadata.object_ids().empty(),"partial regression probe requires at least 19 frames and an existing object");const std::vector<int64_t> selected{metadata.object_ids().front()};
    interaction.append({sam3::VideoActionType::Refine,18,selected});const auto route=interaction.route();TORCH_CHECK(route.type==sam3::VideoActionType::Partial && route.ids,"partial route missing");interaction.append({route.type,18,route.ids});
    const auto range=sam3::video_processing_range(files.size(),{},18,3,true);std::vector<Session*> local;for(auto& session:sessions)local.push_back(session.get());
    for(auto frame=range.first;!range.empty && frame>=range.end;frame+=range.step){
      const auto refined=sam3::propagate_video_refinements(frame,true,*route.ids,local,update.cleanup_area);const auto output=interaction.merge_refined(frame,refined,metadata,suppressed_per_frame[frame]);
      const auto prefix=std::to_string(frame)+".partial.";binary(root/(prefix+"ids.i64.bin"),output.ids);binary(root/(prefix+"scores.f32.bin"),output.probabilities);binary(root/(prefix+"boxes.f32.bin"),output.boxes_xywh);binary(root/(prefix+"masks.bin"),sam3::pack_masks(output.masks));
      const auto fetched=interaction.fetch(frame,metadata,suppressed_per_frame[frame]);TORCH_CHECK(at::equal(fetched.masks,output.masks) && at::equal(fetched.ids,output.ids),"cache fetch changed partial output");std::cout<<"partial frame="<<frame<<" selected="<<selected.front()<<" masks="<<output.ids.numel()<<" encodes="<<encodes<<"\n";
    }
  }
  if(edit_probe){
    if constexpr(Mux){
      TORCH_CHECK(files.size()>20 && metadata.object_ids().size()>1,"SAM3.1 edit regression requires21frames and two existing objects");const auto selected=metadata.object_ids()[0];
      const auto save=[&](const std::string& tag,const sam3::VideoOutput& value){binary(root/(tag+".ids.i64.bin"),value.ids);binary(root/(tag+".scores.f32.bin"),value.probabilities);binary(root/(tag+".boxes.f32.bin"),value.boxes_xywh);binary(root/(tag+".masks.bin"),sam3::pack_masks(value.masks));if(edit_sequence && (tag=="19.repeat1" || tag=="19.repeat2"))for(const auto& session:sessions)if(session->object_ids()==std::vector<int64_t>{selected})for(const bool cond:{true,false})for(const auto& stored:cond?session->state().history.conditioning:session->state().history.tracked)if(stored.index==0 || stored.index==18 || stored.index==19){const auto f=sam3::load_multiplex_frame(stored);const auto stem=tag+"."+(cond?"cond":"tracked")+std::to_string(f.index);binary(root/(stem+".low.f32.bin"),f.masks.low_res_mask.to(at::kFloat));if(f.memory.defined())binary(root/(stem+".memory.f32.bin"),f.memory.to(at::kFloat));}};
      sam3::Sam31VideoEditOptions edit;sam3::TrackingPoints points;points.points=at::tensor({.45,.55,.8,.15},opts).view({2,2});points.labels=at::tensor({1,0},opts.dtype(at::kInt));
      save("18.point",sam3::edit_video_points(18,selected,points,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
      const auto route=interaction.route();TORCH_CHECK(route.type==sam3::VideoActionType::Partial && route.ids,"edited IDs did not select partial propagation");interaction.append({route.type,18,route.ids});std::vector<Session*> local;for(auto& session:sessions)local.push_back(session.get());
      for(int64_t frame=18;frame<=20;++frame){const auto refined=sam3::propagate_video_refinements(frame,false,*route.ids,local,update.cleanup_area);save(std::to_string(frame)+".track",interaction.merge_refined(frame,refined,metadata,suppressed_per_frame[frame]));}
      if(edit_sequence){
        TORCH_CHECK(files.size()>22,"extended edit regression requires23frames");
        const auto propagate=[&](int64_t start,int64_t steps,bool reverse,const std::string& tag){const auto route=interaction.route();TORCH_CHECK(route.type==sam3::VideoActionType::Partial && route.ids,"expected partial action");interaction.append({route.type,start,route.ids});std::vector<Session*> owners;for(auto& session:sessions)owners.push_back(session.get());const auto range=sam3::video_processing_range(files.size(),{},start,steps,reverse);for(int64_t f=range.first;!range.empty && (reverse?f>=range.end:f<=range.end);f+=range.step){if(edit_trace && reverse && f==17){
          const auto dir=root/"reverse-input";std::filesystem::create_directories(dir);std::ofstream manifest(dir/"tensors.tsv");
          const auto dump=[&](const std::string& name,const at::Tensor& x){if(!x.defined())return;manifest<<name<<"\t"<<x.scalar_type()<<"\t";for(auto d:x.sizes())manifest<<d<<",";manifest<<"\t";for(auto d:x.strides())manifest<<d<<",";manifest<<"\n";binary(dir/(name+".bin"),x.to(at::kFloat));};
          for(const auto& owner:sessions)if(owner->object_ids()==std::vector<int64_t>{selected}){
            sam3::MultiplexTemporalState temporal;
            const auto history=sam3::load_selected_multiplex_history(owner->state().history,f,files.size(),reverse,session_options.frame.temporal);
            for(const bool cond:{true,false})for(const auto& x:cond?history.conditioning:history.tracked){
              const auto key=std::string(cond?"cond":"tracked")+std::to_string(x.index)+".";
              dump(key+"memory",x.memory);dump(key+"position",x.memory_position);dump(key+"pointer",x.pointer);dump(key+"image",x.image);dump(key+"image_position",x.image_position);
              (cond?temporal.conditioning:temporal.tracked).push_back({x.index,x.memory,x.memory_position,x.pointer,x.confidence,x.image,x.image_position});
            }
            const auto& features_now=features(f).tracking.propagation;
            const auto source=features_now.image.flatten(2).permute({2,0,1}),position=features_now.position.flatten(2).permute({2,0,1});dump("source",source);dump("source_position",position);
            sam3::MultiplexMemoryConditioner conditioner(store,device);sam3::MultiplexTemporalAssembly assembly;
            dump("conditioned",conditioner.forward(source,position,72,72,f,files.size(),false,reverse,true,temporal,*owner->state().buckets,session_options.frame.temporal,mode,&assembly));
            dump("assembled_memory",assembly.memory);dump("assembled_position",assembly.position);dump("assembled_image",assembly.image);dump("assembled_image_position",assembly.image_position);
            std::ofstream plan(dir/"plan.tsv");for(auto r:assembly.plan.spatial)plan<<"spatial\t"<<r.frame<<"\t"<<r.position<<"\t"<<r.conditioning<<"\n";for(auto r:assembly.plan.pointers)plan<<"pointer\t"<<r.frame<<"\t"<<r.position<<"\t"<<r.conditioning<<"\n";
          }
        }
        const auto refined=sam3::propagate_video_refinements(f,reverse,*route.ids,owners,update.cleanup_area);save(std::to_string(f)+tag,interaction.merge_refined(f,refined,metadata,suppressed_per_frame[f]));}};
        save("19.repeat1",sam3::edit_video_points(19,selected,points,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
        sam3::TrackingPoints extra;extra.points=at::tensor({.5,.65},opts).view({1,2});extra.labels=at::tensor({1},opts.dtype(at::kInt));edit.clear_old_points=false;
        save("19.repeat2",sam3::edit_video_points(19,selected,extra,sessions,factory,metadata,interaction,suppressed_per_frame,edit));edit.clear_old_points=true;
        propagate(19,2,true,".reverse");
        save("20.new",sam3::edit_video_points(20,9000,extra,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
        propagate(20,2,false,".multi");
        sam3::remove_video_user_object(9000,sessions,metadata,interaction);TORCH_CHECK(interaction.route().type==sam3::VideoActionType::Fetch,"removal should select fetch");save("20.remove",interaction.fetch(20,metadata,suppressed_per_frame[20]));
        sam3::remove_video_user_object(9000,sessions,metadata,interaction);save("20.remove_again",interaction.fetch(20,metadata,suppressed_per_frame[20]));
        edit.stateless_refinement=true;const auto other=metadata.object_ids()[1];save("22.stateless",sam3::edit_video_points(22,other,points,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
      }
      std::cout<<"SAM3.1 singleton point edit and partial propagation completed; buckets="<<metadata.buckets_per_rank[0]<<"\n";
    }
    else {
      TORCH_CHECK(files.size()>22 && metadata.object_ids().size()>1,"edit regression requires23frames and two existing objects");const auto selected=metadata.object_ids()[0],other=metadata.object_ids()[1];
      const auto save=[&](const std::string& tag,const sam3::VideoOutput& value){binary(root/(tag+".ids.i64.bin"),value.ids);binary(root/(tag+".scores.f32.bin"),value.probabilities);binary(root/(tag+".boxes.f32.bin"),value.boxes_xywh);binary(root/(tag+".masks.bin"),sam3::pack_masks(value.masks));};
      sam3::VideoEditOptions edit;edit.cleanup_area=update.cleanup_area;
      sam3::TrackingPoints points;points.points=at::tensor({.45,.55,.8,.15},opts).view({2,2});points.labels=at::tensor({1,0},opts.dtype(at::kInt));
      save("18.edit_point",sam3::edit_video_points(18,selected,points,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
      auto mask=at::zeros({h,w},opts);mask.slice(0,h/4,3*h/4).slice(1,w/3,2*w/3).fill_(1);
      save("20.edit_mask_new",sam3::edit_video_mask(20,9000,mask,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
      save("21.edit_mask_existing",sam3::edit_video_mask(21,selected,mask.roll({h/10,w/12},{0,1}),sessions,factory,metadata,interaction,suppressed_per_frame,edit));
      const auto route=interaction.route();TORCH_CHECK(route.type==sam3::VideoActionType::Partial && route.ids,"edited IDs did not select partial propagation");interaction.append({route.type,18,route.ids});std::vector<Session*> local;for(auto& session:sessions)local.push_back(session.get());
      for(int64_t frame=18;frame<=22;++frame){const auto refined=sam3::propagate_video_refinements(frame,false,*route.ids,local,update.cleanup_area);save(std::to_string(frame)+".edit_track",interaction.merge_refined(frame,refined,metadata,suppressed_per_frame[frame]));}
      sam3::remove_video_user_object(9000,sessions,metadata,interaction);TORCH_CHECK(interaction.route().type==sam3::VideoActionType::Fetch,"remove-only action should fetch");save("20.edit_remove",interaction.fetch(20,metadata,suppressed_per_frame[20]));
      edit.stateless_refinement=true;const auto before_ids=metadata.object_ids();const auto before_actions=interaction.actions().size();bool rejected=false;
      try{sam3::edit_video_points(files.size(),other,points,sessions,factory,metadata,interaction,suppressed_per_frame,edit);}catch(const c10::Error&){rejected=true;}
      TORCH_CHECK(rejected && metadata.object_ids()==before_ids && interaction.actions().size()==before_actions,"invalid stateless edit changed state");
      save("22.edit_stateless",sam3::edit_video_points(22,other,points,sessions,factory,metadata,interaction,suppressed_per_frame,edit));
      std::cout<<"point, new/existing mask, partial propagation, removal and stateless point edit completed; max_id="<<metadata.max_id<<"\n";
    }
  }
std::cout<<"shared-trunk detector/tracker pipeline completed; raw outputs, no Python\n";
}
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==9 || argc==10,"usage: sam3_video_pipeline_probe STORE sam3|sam3.1 DEVICE MODE FRAMES.txt BPE.gz PROMPT.txt OUTPUT [--trace|--partial-probe|--edit-probe|--edit-sequence-probe|--edit-sequence-trace]");const std::string option=argc==10?argv[9]:"";TORCH_CHECK(option.empty() || option=="--trace" || option=="--partial-probe" || option=="--edit-probe" || option=="--edit-sequence-probe" || option=="--edit-sequence-trace","unknown probe option");at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  const sam3::WeightStore store(std::filesystem::u8path(argv[1]));const std::string model=argv[2],mode=argv[4];const at::Device device(argv[3]);TORCH_CHECK(model=="sam3" || model=="sam3.1","invalid model");
  const auto manifest=std::filesystem::u8path(argv[5]);std::ifstream input(manifest);TORCH_CHECK(input,"cannot read frame manifest");std::vector<std::filesystem::path> files;std::string line;
  while(std::getline(input,line)){if(!line.empty() && line.back()=='\r')line.pop_back();if(line.empty() || line[0]=='#')continue;auto p=std::filesystem::u8path(line);files.push_back(p.is_absolute()?p:manifest.parent_path()/p);}TORCH_CHECK(!files.empty(),"empty frame manifest");
  std::ifstream textfile(std::filesystem::u8path(argv[7]),std::ios::binary);TORCH_CHECK(textfile,"cannot read prompt");const std::string text((std::istreambuf_iterator<char>(textfile)),{});sam3::Tokenizer tokenizer(std::filesystem::u8path(argv[6]));
  sam3::GroundingPrompt prompt;const auto opts=at::TensorOptions().device(device);prompt.image_ids=prompt.text_ids=at::zeros({1},opts.dtype(at::kLong));
  {sam3::TextEncoder encoder(store,model,device);const auto encoded=encoder.forward([&]{std::vector<at::Tensor> rows;for(const auto& ids:tokenizer.tokenize(model=="sam3.1"?std::vector<std::string>{text,"visual","geometric"}:std::vector<std::string>{text,"visual"}))rows.push_back(at::tensor(ids,opts.dtype(at::kLong)));return at::stack(rows);}(),mode);prompt.text_padding=std::get<0>(encoded);prompt.text_features=std::get<1>(encoded);}
  const auto labels=at::empty({0,1},opts.dtype(at::kLong)),padding=at::empty({1,0},opts.dtype(at::kBool));prompt.geometry={at::empty({0,1,2},opts),labels,padding,at::empty({0,1,4},opts),labels,padding};
  if(model=="sam3")run<false>(store,device,mode,files,prompt,std::filesystem::u8path(argv[8]),option=="--trace",option=="--partial-probe",option=="--edit-probe" || option=="--edit-sequence-probe" || option=="--edit-sequence-trace",option=="--edit-sequence-probe" || option=="--edit-sequence-trace",option=="--edit-sequence-trace");else run<true>(store,device,mode,files,prompt,std::filesystem::u8path(argv[8]),option=="--trace",option=="--partial-probe",option=="--edit-probe" || option=="--edit-sequence-probe" || option=="--edit-sequence-trace",option=="--edit-sequence-probe" || option=="--edit-sequence-trace",option=="--edit-sequence-trace");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
