#include "sam3/multiplex_frame.h"
#include "sam3/multiplex_storage.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <numeric>
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==5 || argc==6,"usage: sam3_multiplex_update STORE cpu|cuda fp32|fp16|bf16_reference ADDED_OBJECTS [HISTORY_DIRECTORY]");
    at::set_num_threads(4);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
    const at::Device device(argv[2]);const std::string mode=argv[3];const auto added=std::stoll(argv[4]);TORCH_CHECK(added>0,"positive added count required");
    const sam3::Sam31TrackingFrame core(sam3::WeightStore(std::filesystem::u8path(argv[1])),device);
    auto buckets=sam3::MultiplexController().get_state(3,device,at::kFloat,true,std::vector<int64_t>{101,-77,10000000001LL});
    const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
    const auto features=[&]{return sam3::TrackingFeatures{at::randn({1,256,72,72},opts),at::randn({1,256,72,72},opts),{at::randn({1,32,288,288},opts),at::randn({1,64,144,144},opts)}};};
    const auto interactive=features(),propagation=features();sam3::MultiplexFrameHistory history;sam3::MultiplexFrameOptions policy;
    sam3::MultiplexFrameRequest request;request.frame_count=2;request.initial=true;request.points=at::rand({3,2,2},opts)*1008;request.labels=at::ones({3,2},opts.dtype(at::kInt));
    auto frame=core.forward(interactive,propagation,request,history,buckets,policy,mode);
    // A capacity failure must not partially append IDs or alter the frame.
    const auto original=frame.masks.low_res_mask.clone();const auto assignments=buckets.assignments();
    sam3::MultiplexMaskUpdate update;update.append=true;
    const auto update_frame=[&](const at::Tensor& masks,const std::vector<int64_t>& indices,const std::optional<std::vector<int64_t>>& ids){
      auto expected=frame;auto expected_buckets=buckets;
      if(argc==6){
        core.update_masks(interactive,propagation,masks,indices,ids,expected,expected_buckets,update,policy,mode);
        frame.masks.low_res_mask=frame.masks.low_res_mask.cpu();frame.masks.high_res_mask=frame.masks.high_res_mask.cpu();
        sam3::archive_multiplex_frame(frame,std::filesystem::u8path(argv[5]));
      }
      const auto result=core.update_masks(interactive,propagation,masks,indices,ids,frame,buckets,update,policy,mode);
      if(argc==6){
        TORCH_CHECK(buckets.assignments()==expected_buckets.assignments(),"paging changed update assignments");
        for(const auto& pair:std::vector<std::pair<at::Tensor,at::Tensor>>{{frame.masks.low_res_mask,expected.masks.low_res_mask},{frame.masks.high_res_mask,expected.masks.high_res_mask},{frame.masks.object_logits,expected.masks.object_logits},{frame.pointer,expected.pointer},{frame.memory,expected.memory},{frame.memory_position,expected.memory_position}})
          TORCH_CHECK(at::equal(pair.first,pair.second),"paging changed updated masks/pointers/memory");
      }
      return result;
    };
    bool rejected=false;try {update_frame(at::zeros({17,1,4,4},opts),std::vector<int64_t>(17),std::vector<int64_t>(17));}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected && buckets.assignments()==assignments && at::equal(original,frame.masks.low_res_mask),"failed update changed caller state");
    std::vector<int64_t> indices(added),ids(added);std::iota(indices.begin(),indices.end(),3);std::iota(ids.begin(),ids.end(),9001);
    auto masks=at::rand({added,1,1008,1008},opts)>.5;masks.select(0,0).zero_();update.allow_new_buckets=true;update.prefer_new_buckets=true;
    TORCH_CHECK(update_frame(masks,indices,ids)==indices,"incorrect allocated indices");
    TORCH_CHECK(buckets.object_count()==3+added && buckets.bucket_count()>1 && buckets.object_ids()->back()==9000+added,"incorrect IDs or bucket growth");
    const auto untouched=frame.masks.low_res_mask.select(0,1).clone();
    update.append=false;update_frame(at::rand({2,1,1008,1008},opts)>.5,{added+2,0},std::vector<int64_t>{9000+added,101});
    TORCH_CHECK(at::equal(untouched,frame.masks.low_res_mask.select(0,1)),"reconditioning changed an unrelated object");
    TORCH_CHECK(frame.pointer.size(0)==buckets.bucket_count() && frame.memory.size(0)==buckets.bucket_count(),"updated memory is not bucket-aligned");
    history.conditioning.push_back(std::move(frame));request.index=1;request.initial=false;request.points=at::Tensor();request.labels=at::Tensor();
    const auto tracked=core.forward(features(),features(),request,history,buckets,policy,mode);
    TORCH_CHECK(tracked.masks.low_res_mask.sizes()==at::IntArrayRef({3+added,1,288,288}) && at::isfinite(tracked.masks.low_res_mask).all().item<bool>(),"updated frame failed propagation");
    std::cout<<"native SAM3.1 mask update passed: 3 -> "<<buckets.object_count()<<" objects, "<<buckets.bucket_count()<<" buckets; capacity rollback, recondition, then propagation; no Python\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
