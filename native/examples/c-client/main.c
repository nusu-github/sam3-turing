#include <sam3/c_api.h>
#include <stdio.h>
#include <math.h>
#define CHECK(call) do { if((call)!=SAM3_OK) { fprintf(stderr,"%s\n",sam3_last_error()); return 1; } } while(0)
int main(int argc,char** argv) {
  sam3_predictor_options options;
  if(sam3_abi_version()!=SAM3_ABI_VERSION) return 1;
  CHECK(sam3_predictor_options_init(&options,SAM3_MODEL_31));
  printf("SAM3 C ABI %u: SAM3.1 image threshold %.2f, output batch %lld\n",
         sam3_abi_version(),options.image_detection_threshold,(long long)options.output_batch_size);
  if(argc==1) return 0;
  if(argc!=4) { fprintf(stderr,"usage: sam3_c_client [WEIGHTS VOCABULARY TEXT]\n"); return 1; }
  sam3_context_options config;
  CHECK(sam3_context_options_init(&config));
  config.weight_directory=argv[1];config.vocabulary_path=argv[2];
  CHECK(sam3_runtime_configure(4,0));
  sam3_context* context=NULL;sam3_result* result=NULL;
  CHECK(sam3_context_create(&config,&context));
  const char* text=argv[3];CHECK(sam3_encode_text(context,&text,1,&result));
  sam3_context_release(context);
  sam3_tensor_view features,tokens;
  CHECK(sam3_result_get(result,"features",&features));CHECK(sam3_result_get(result,"tokens",&tokens));
  if(features.dtype!=SAM3_F32 || features.rank!=3 || features.shape[1]!=1 || features.shape[2]!=256 || tokens.dtype!=SAM3_I64) return 1;
  for(uint64_t i=0;i<features.bytes/sizeof(float);++i) if(!isfinite(((const float*)features.data)[i])) return 1;
  printf("Encoded text: %lld tokens, %lld x %lld x %lld finite features\n",(long long)tokens.shape[1],(long long)features.shape[0],(long long)features.shape[1],(long long)features.shape[2]);
  sam3_result_release(result);return 0;
}
