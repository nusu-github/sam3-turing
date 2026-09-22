#include "sam3/c_api.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed line %d: %s; %s\n",__LINE__,#x,sam3_last_error());return 1;}}while(0)
int main(void){
  sam3_context_options context;sam3_context* handle=(sam3_context*)1;sam3_result* result=(sam3_result*)1;sam3_tensor_view view;
  CHECK(sam3_abi_version()==SAM3_ABI_VERSION);
  CHECK(sam3_context_options_init(&context)==SAM3_OK && context.struct_size==sizeof(context));
  CHECK(sam3_context_create(&context,&handle)==SAM3_INVALID_ARGUMENT && handle==NULL && strlen(sam3_last_error())>0);
  context.struct_size=1;CHECK(sam3_context_create(&context,&handle)==SAM3_INVALID_ARGUMENT);
  CHECK(sam3_context_options_init(&context)==SAM3_OK);context.weight_directory="sam3-deliberately-missing-weight-directory";
  context.precision=999;CHECK(sam3_context_create(&context,&handle)==SAM3_INVALID_ARGUMENT && handle==NULL);
  context.precision=SAM3_FP32;CHECK(sam3_context_create(&context,&handle)==SAM3_RUNTIME_ERROR && handle==NULL);
  CHECK(sam3_result_retain(NULL,&result)==SAM3_INVALID_ARGUMENT && result==NULL);
  CHECK(sam3_result_get(NULL,"masks",&view)==SAM3_INVALID_ARGUMENT && view.data==NULL);
  CHECK(sam3_video_cancel(NULL)==SAM3_INVALID_ARGUMENT);
  CHECK(sam3_runtime_configure(0,0)==SAM3_INVALID_ARGUMENT);
  sam3_video_options video;CHECK(sam3_video_options_init(&video,SAM3_MODEL_31)==SAM3_OK && video.non_overlap_output==1 && video.memory_slots==7);
  CHECK(sam3_video_options_init(&video,SAM3_MODEL_3)==SAM3_OK && video.non_overlap_output==0 && video.clear_near_input==1);
  sam3_interactive_request interactive;CHECK(sam3_interactive_request_init(&interactive)==SAM3_OK && interactive.multimask==1);
  sam3_grounding_request grounding;CHECK(sam3_grounding_request_init(&grounding)==SAM3_OK && grounding.resize_chunk==8);
  sam3_context_release(NULL);sam3_image_release(NULL);sam3_video_release(NULL);sam3_result_release(NULL);
  puts("pure C API: ABI/defaults/NULL/invalid options/exception translation passed");return 0;
}
