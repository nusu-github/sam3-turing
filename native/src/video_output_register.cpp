#include "sam3/video_output.h"
#include <torch/library.h>
namespace {
using Dict=c10::Dict<std::string,at::Tensor>;
void save(Dict& out,const std::string& root,const sam3::VideoOutput& value){
  for(const auto& [key,x]:std::vector<std::pair<std::string,at::Tensor>>{{"ids",value.ids},{"scores",value.probabilities},{"boxes",value.boxes_xywh},{"masks",value.masks},{"centers",value.centers}})if(x.defined())out.insert(root+key,x.cpu());
  for(const auto& [id,x]:value.cached_masks)out.insert(root+"cache/"+std::to_string(id),x.cpu());
}
}
TORCH_LIBRARY_FRAGMENT(sam3_native,m){
 m.def("video_output_sequence(Tensor[][] inputs, int[][] ids, int[][] removed, int[][] suppressed, int[][] unconfirmed, int[] frames, int[] options, bool reverse=False, bool centers=False) -> Dict(str, Tensor)",
 [](const std::vector<std::vector<at::Tensor>>& inputs,const std::vector<std::vector<int64_t>>& ids,const std::vector<std::vector<int64_t>>& removed,const std::vector<std::vector<int64_t>>& suppressed,const std::vector<std::vector<int64_t>>& unconfirmed,const std::vector<int64_t>& frames,const std::vector<int64_t>& options,bool reverse,bool centers){
  TORCH_CHECK(options.size()==7 && inputs.size()==frames.size() && ids.size()==frames.size() && removed.size()==frames.size() && suppressed.size()==frames.size() && unconfirmed.size()==frames.size(),"invalid output sequence");
  sam3::VideoOutputBufferOptions o;o.frame_count=options[0];o.end_frame=options[1];o.height=options[2];o.width=options[3];o.hotstart_delay=options[4];o.confirmation_threshold=options[5];o.batch_size=options[6];o.reverse=reverse;o.centers=centers;sam3::VideoOutputBuffer buffer(o);Dict result;
  for(size_t i=0;i<frames.size();++i){const auto& x=inputs[i];TORCH_CHECK(x.size()==3 && x[0].size(0)==int64_t(ids[i].size()) && x[1].numel()==int64_t(ids[i].size()) && x[2].numel()==int64_t(ids[i].size()),"output fixture masks/scores/trackers required");sam3::VideoRawOutput raw;raw.frame=frames[i];raw.removed={removed[i].begin(),removed[i].end()};raw.suppressed={suppressed[i].begin(),suppressed[i].end()};raw.unconfirmed=std::set<int64_t>(unconfirmed[i].begin(),unconfirmed[i].end());
   for(size_t j=0;j<ids[i].size();++j){raw.masks.emplace(ids[i][j],x[0].slice(0,j,j+1));raw.scores.emplace(ids[i][j],x[1][j].item<double>());raw.tracker_scores.emplace(ids[i][j],x[2][j]);}
   for(const auto& output:buffer.push(raw)){const auto root=std::to_string(output.frame)+"/";save(result,root,output.output);result.insert(root+"emitted_at",at::scalar_tensor(frames[i],at::kLong));result.insert(root+"hidden",at::tensor(std::vector<int64_t>(output.hidden.begin(),output.hidden.end()),at::kLong));}
  }result.insert("pending",at::scalar_tensor(int64_t(buffer.pending()),at::kLong));return result;
 });
}
