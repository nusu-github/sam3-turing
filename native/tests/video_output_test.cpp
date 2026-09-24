#include "sam3/video_output.h"
#include <ATen/Parallel.h>
#include <iostream>
int main(int argc,char** argv){try{
 at::set_num_threads(1);const at::Device device(argc>1?argv[1]:"cpu");const auto opts=at::TensorOptions().device(device);
 sam3::VideoOutputBufferOptions o;o.frame_count=3;o.end_frame=2;o.height=3;o.width=5;o.hotstart_delay=2;o.batch_size=2;o.centers=true;sam3::VideoOutputBuffer buffer(o);
 sam3::VideoRawOutput raw;raw.frame=0;raw.masks[9]=at::ones({1,3,5},opts.dtype(at::kBool));raw.scores[9]=.75;raw.frame_stats["num_obj_tracked"]=1;
 TORCH_CHECK(buffer.push(raw).empty(),"early emission");raw.masks[9].zero_();raw.frame=1;TORCH_CHECK(buffer.push(raw).empty(),"postprocess batch ignored");raw.frame=2;auto result=buffer.push(raw);
 TORCH_CHECK(result.size()==3 && result[0].output.masks.all().item<bool>() && result[1].output.masks.size(0)==0 && result[1].output.cached_masks.count(9),"snapshot/empty cache mismatch");
 TORCH_CHECK(result[0].output.frame_stats.at("num_obj_tracked")==1 && result[0].output.centers.sizes()==at::IntArrayRef({1,2}),"stats/centers missing");
 bool rejected=false;try{buffer.push(raw);}catch(const c10::Error&){rejected=true;}TORCH_CHECK(rejected,"closed buffer accepted output");buffer.reset();raw.frame=0;buffer.push(raw);buffer.cancel();TORCH_CHECK(buffer.pending()==0,"cancel retained output");buffer.reset();
 // No detection limit: 257 tied overlapping objects survive row filtering,
 // first sorted ID wins, and boxes remain those computed before suppression.
 sam3::VideoRawOutput crowded;for(int64_t id=0;id<257;++id){crowded.masks[id]=at::ones({1,3,5},opts.dtype(at::kBool));crowded.scores[id]=.5;crowded.tracker_scores[id]=at::scalar_tensor(.5,opts);}
 const auto output=sam3::postprocess_video_output(crowded,3,5);TORCH_CHECK(output.ids.numel()==257 && output.masks[0].all().item<bool>() && !output.masks.slice(0,1).any().item<bool>() && output.boxes_xywh.select(1,2).gt(0).all().item<bool>(),"object cap, tie or post-overlap re-filtering");
 std::cout<<"video output snapshot/cancel/reset/uncapped overlap passed "<<device<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
