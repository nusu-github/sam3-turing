#include "sam3/video_memory.h"
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(sam3_native,m){
  m.def("prepare_video_memory(Tensor low, bool multiplex, bool warmup) -> Tensor[]",[](const at::Tensor& low,bool mux,bool warmup){const auto value=sam3::prepare_video_memory(low,mux?sam3::AssociationPolicy::Sam31:sam3::AssociationPolicy::Sam3,warmup);return std::vector<at::Tensor>{value.high_masks,value.object_logits};});
  m.def("video_memory_rows(int[] global_ids, int[][] state_ids) -> int[][]",&sam3::video_memory_rows);
}
