/* Development client compiled as C11, with no Torch or C++ headers. */
#include "sam3/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>
#define OK(call) do{sam3_status status_=(call);if(status_!=SAM3_OK){fprintf(stderr,"line %d: status=%d %s\n",__LINE__,status_,sam3_last_error());exit(1);}}while(0)
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed line %d: %s (%s)\n",__LINE__,#x,sam3_last_error());exit(1);}}while(0)
static const char* input_root;static const char* output_root;static int64_t height,width;
static uint8_t* pixels;static sam3_video* running_video;static int provider_calls,fail_provider,callback_count,operation,stop_after,use_cancel;
static sam3_result* retained_output;
static sam3_rgb_view read_rgb(int64_t index){
  char path[4096];snprintf(path,sizeof(path),"%s/%" PRId64 ".rgb",input_root,index);FILE* f=fopen(path,"rb");CHECK(f);
  free(pixels);const int64_t stride=width*3+7;pixels=(uint8_t*)malloc((size_t)(height*stride));CHECK(pixels);
  memset(pixels,0xcd,(size_t)(height*stride));for(int64_t y=0;y<height;++y)CHECK(fread(pixels+y*stride,1,(size_t)(width*3),f)==(size_t)(width*3));CHECK(fgetc(f)==EOF);CHECK(fclose(f)==0);
  sam3_rgb_view view={pixels,(uint64_t)(height*stride),height,width,stride};return view;
}
static void dump(const sam3_result* result,const char* name){
  char path[4096];snprintf(path,sizeof(path),"%s/%s.json",output_root,name);FILE* meta=fopen(path,"w");CHECK(meta);fputs("{",meta);int64_t count;OK(sam3_result_field_count(result,&count));
  for(int64_t i=0;i<count;++i){const char* field;sam3_tensor_view value;OK(sam3_result_field_name(result,i,&field));OK(sam3_result_get(result,field,&value));char safe[256];CHECK(strlen(field)<sizeof(safe));strcpy(safe,field);for(char* p=safe;*p;++p)if(*p=='/')*p='_';
    snprintf(path,sizeof(path),"%s/%s-%s.bin",output_root,name,safe);FILE* data=fopen(path,"wb");CHECK(data);CHECK(fwrite(value.data,1,(size_t)value.bytes,data)==value.bytes);CHECK(fclose(data)==0);
    fprintf(meta,"%s\"%s\":{\"dtype\":%d,\"shape\":[",i?",":"",field,value.dtype);for(int64_t j=0;j<value.rank;++j)fprintf(meta,"%s%" PRId64,j?",":"",value.shape[j]);fprintf(meta,"],\"bytes\":%" PRIu64 "}",value.bytes);
  }
  fputs("}\n",meta);CHECK(fclose(meta)==0);
}
static float number(const sam3_tensor_view* v,int64_t i){
  if(v->dtype==SAM3_F32)return ((const float*)v->data)[i];if(v->dtype==SAM3_F64)return (float)((const double*)v->data)[i];
  uint16_t bits=((const uint16_t*)v->data)[i];if(v->dtype==SAM3_BF16){union{uint32_t u;float f;}x;x.u=(uint32_t)bits<<16;return x.f;}
  CHECK(v->dtype==SAM3_F16);int sign=(bits&0x8000)?-1:1,exponent=(bits>>10)&31,mantissa=bits&1023;
  if(exponent==31)return mantissa?NAN:sign*INFINITY;return sign*(exponent?ldexpf((float)(1024+mantissa),exponent-25):ldexpf((float)mantissa,-24));
}
static int32_t provider(void* user,int64_t index,sam3_rgb_view* out){
  (void)user;if(fail_provider)return -1;if(running_video){sam3_result* invalid=NULL;CHECK(sam3_video_object_ids(running_video,&invalid)==SAM3_BUSY && invalid==NULL);}
  ++provider_calls;*out=read_rgb(index);return 0;
}
static int32_t callback(void* user,const sam3_result* result){
  (void)user;char name[80];snprintf(name,sizeof(name),"%d-%d",operation,callback_count);dump(result,name);++callback_count;
  sam3_result* invalid=NULL;CHECK(sam3_video_object_ids(running_video,&invalid)==SAM3_BUSY && invalid==NULL);
  if(!retained_output)OK(sam3_result_retain(result,&retained_output));
  if(stop_after>0 && callback_count>=stop_after){if(use_cancel){OK(sam3_video_cancel(running_video));return 0;}return 1;}return 0;
}
static void output(sam3_result* result,const char* name){dump(result,name);sam3_result_release(result);}
static void image_test(sam3_context* context){
  sam3_image* image=NULL;OK(sam3_image_create(context,SAM3_IMAGE_ALL,&image));sam3_rgb_view rgb=read_rgb(0);OK(sam3_image_set_rgb(image,&rgb));
  OK(sam3_context_trim(context));sam3_context_release(context); /* child owns context */
  float coords[]={500,600,100,100};int64_t labels[]={1,0},ps[]={2,2},ls[]={2};sam3_tensor_view p={coords,sizeof(coords),SAM3_F32,2,ps},l={labels,sizeof(labels),SAM3_I64,1,ls};
  sam3_interactive_request request;OK(sam3_interactive_request_init(&request));request.points=&p;request.labels=&l;sam3_result *first=NULL,*second=NULL,*held=NULL;
  OK(sam3_image_predict(image,&request,&first));dump(first,"initial");OK(sam3_result_retain(first,&held));sam3_result_release(first);
  sam3_tensor_view iou,low;OK(sam3_result_get(held,"iou",&iou));OK(sam3_result_get(held,"low_res_logits",&low));int64_t best=0;for(int64_t i=1;i<iou.shape[1];++i)if(number(&iou,i)>number(&iou,best))best=i;
  int64_t shape[]={1,288,288};uint64_t row=low.bytes/(uint64_t)low.shape[1];sam3_tensor_view mask={(const uint8_t*)low.data+row*best,row,low.dtype,3,shape};request.masks=&mask;request.multimask=0;
  OK(sam3_image_predict(image,&request,&second));dump(second,"refined");sam3_result_release(second);sam3_result_release(held);
  /* Compare both batch positions to their corresponding reference outputs. */
  sam3_rgb_view batch[]={rgb,rgb};OK(sam3_image_set_rgb_batch(image,batch,2));float boxes[]={.1f,.1f,.9f,.9f,.2f,.2f,.7f,.8f};int64_t bs[]={2,4};sam3_tensor_view box={boxes,sizeof(boxes),SAM3_F32,2,bs};
  OK(sam3_interactive_request_init(&request));request.boxes=&box;request.pixel_coordinates=0;request.max_hole_area=0;
  OK(sam3_image_predict(image,&request,&first));request.image_index=1;OK(sam3_image_predict(image,&request,&second));dump(first,"batch");dump(second,"batch-1");sam3_result_release(first);sam3_result_release(second);
  OK(sam3_image_embedding(image,&first));sam3_tensor_view embedding;OK(sam3_result_get(first,"embedding",&embedding));CHECK(embedding.rank==4 && embedding.shape[0]==2);
  OK(sam3_image_reset(image));CHECK(sam3_image_embedding(image,&second)==SAM3_INVALID_ARGUMENT && second==NULL);sam3_image_release(image);
  OK(sam3_result_get(first,"embedding",&embedding));CHECK(embedding.shape[0]==2);sam3_result_release(first);
}
static void ground_test(sam3_context* context){
  const char* texts[]={"a truck","person"};sam3_result *encoded=NULL,*tokenized=NULL;OK(sam3_encode_text(context,texts,2,&encoded));dump(encoded,"text");
  const char embedded[]={'a',0,' ','t','r','u','c','k'};sam3_utf8_view counted={embedded,sizeof(embedded)};OK(sam3_tokenize_utf8(context,&counted,1,&tokenized));output(tokenized,"nul-tokenize");
  sam3_tensor_view features,padding;OK(sam3_result_get(encoded,"features",&features));OK(sam3_result_get(encoded,"padding",&padding));
  sam3_image* image=NULL;OK(sam3_image_create(context,SAM3_IMAGE_GROUNDING,&image));sam3_rgb_view rgb=read_rgb(0);OK(sam3_image_set_rgb(image,&rgb));
  sam3_grounding_request request;OK(sam3_grounding_request_init(&request));request.text_features=&features;request.text_padding=&padding;request.confidence_threshold=-1.;
  sam3_result* result=NULL;OK(sam3_image_ground(image,&request,&result));sam3_tensor_view scores;OK(sam3_result_get(result,"0/scores",&scores));CHECK(scores.shape[0]==200);output(result,"ground");
  int64_t ids[]={0,0},texts_ids[]={1,0},is[]={2};sam3_tensor_view iv={ids,sizeof(ids),SAM3_I64,1,is},tv={texts_ids,sizeof(texts_ids),SAM3_I64,1,is};
  float boxes[]={.5f,.5f,.6f,.7f,.4f,.4f,.3f,.3f};int64_t labels[]={1,0},bs[]={1,2,4},ls[]={1,2};sam3_tensor_view box={boxes,sizeof(boxes),SAM3_F32,3,bs},label={labels,sizeof(labels),SAM3_I64,2,ls};
  request.image_ids=&iv;request.text_ids=&tv;request.boxes=&box;request.box_labels=&label;request.confidence_threshold=1.;OK(sam3_image_ground(image,&request,&result));output(result,"geometry");
  float visual[3*2*256];for(size_t i=0;i<sizeof(visual)/sizeof(float);++i)visual[i]=.125f;uint8_t pad[]={0,0,1,0,1,1};
  int64_t vs[]={3,2,256},pads[]={2,3},prevs[]={5184,2,256};sam3_tensor_view vv={visual,sizeof(visual),SAM3_F32,3,vs},pv={pad,sizeof(pad),SAM3_BOOL,2,pads};
  const size_t n=5184*2*256;float* previous=(float*)malloc(n*sizeof(float));CHECK(previous);for(size_t i=0;i<n;++i)previous[i]=.01f;
  sam3_tensor_view previous_view={previous,n*sizeof(float),SAM3_F32,3,prevs};request.visual_features=&vv;request.visual_padding=&pv;request.previous_mask=&previous_view;request.use_text=0;
  OK(sam3_image_ground(image,&request,&result));output(result,"visual");free(previous);
  sam3_result_release(encoded);sam3_image_release(image);OK(sam3_context_trim(context));sam3_context_release(context);
}
static void video_test(sam3_context* context,int model,const char* history){
  sam3_video_options options;OK(sam3_video_options_init(&options,model));options.frames=3;options.height=height;options.width=width;options.provider=provider;if(model==SAM3_MODEL_31)options.history_directory=history;
  sam3_video *video=NULL,*other=NULL;OK(sam3_video_create(context,&options,&video));OK(sam3_video_create(context,&options,&other));
  float coords[]={.6f,.68f,.25f,.3f};int64_t labels[]={1,0},ps[]={2,2},ls[]={2};sam3_tensor_view p={coords,sizeof(coords),SAM3_F32,2,ps},l={labels,sizeof(labels),SAM3_I64,1,ls};sam3_result* result=NULL;
  fail_provider=1;CHECK(sam3_video_add_points(other,0,11,&p,&l,NULL,1,1,0,&result)==SAM3_CALLBACK_ERROR && result==NULL);fail_provider=0;sam3_video_release(other);
  OK(sam3_context_trim(context));sam3_context_release(context);running_video=video;
  OK(sam3_video_add_points(video,0,11,&p,&l,NULL,1,1,0,&result));output(result,"0-0");
  if(model==SAM3_MODEL_3){float box[]={.79f,.13f,.99f,.94f};int64_t bs[]={4};sam3_tensor_view b={box,sizeof(box),SAM3_F32,1,bs};OK(sam3_video_add_points(video,0,22,NULL,NULL,&b,1,1,0,&result));output(result,"1-0");}
  OK(sam3_video_preflight(video,1));operation=model==SAM3_MODEL_3?3:2;callback_count=0;OK(sam3_video_propagate(video,0,2,0,1,0,callback,NULL));CHECK(callback_count==3);
  float refine[]={.63f,.7f};int64_t one[]={1},rs[]={1,2},os[]={1};sam3_tensor_view rp={refine,sizeof(refine),SAM3_F32,2,rs},rl={one,sizeof(one),SAM3_I64,1,os};
  OK(sam3_video_add_points(video,1,11,&rp,&rl,NULL,1,1,model==SAM3_MODEL_3,&result));output(result,model==SAM3_MODEL_3?"4-0":"3-0");
  OK(sam3_video_preflight(video,1));operation=5;callback_count=0;OK(sam3_video_propagate(video,2,2,1,1,0,callback,NULL));CHECK(callback_count==3);
  if(model==SAM3_MODEL_3){operation=6;callback_count=0;OK(sam3_video_remove_object(video,22,1,callback,NULL));CHECK(callback_count==1);}
  operation=20;callback_count=0;stop_after=1;use_cancel=1;OK(sam3_video_propagate(video,0,2,0,1,1,callback,NULL));CHECK(callback_count==1);stop_after=0;
  OK(sam3_video_object_ids(video,&result));sam3_tensor_view ids;OK(sam3_result_get(result,"ids",&ids));CHECK(ids.shape[0]==1);sam3_result_release(result);
  OK(sam3_video_clear_input(video,1,11,&result));sam3_result_release(result);OK(sam3_video_reset(video));sam3_video_release(video);running_video=NULL;
  sam3_tensor_view masks;OK(sam3_result_get(retained_output,"masks",&masks));CHECK(masks.rank==4 && masks.shape[2]==height);sam3_result_release(retained_output);retained_output=NULL;
  printf("frame provider calls=%d; callbacks/reentry/cancel/provider error/context lifetime/result retention passed\n",provider_calls);
}
int main(int argc,char** argv){
  CHECK(sam3_abi_version()==SAM3_ABI_VERSION);CHECK(argc==8 || argc==9);input_root=argv[6];output_root=argv[7];char path[4096];snprintf(path,sizeof(path),"%s/size.txt",input_root);FILE* f=fopen(path,"r");CHECK(f && fscanf(f,"%" SCNd64 " %" SCNd64,&height,&width)==2);CHECK(fclose(f)==0);
  sam3_context_options options;OK(sam3_context_options_init(&options));options.weight_directory=argv[1];options.model=atoi(argv[2]);options.device=argv[3];options.precision=atoi(argv[4]);options.vocabulary_path=argc==9?argv[8]:NULL;
  OK(sam3_runtime_configure(4,0));sam3_context* context=NULL;OK(sam3_context_create(&options,&context));
  if(strcmp(argv[5],"image")==0)image_test(context);else if(strcmp(argv[5],"ground")==0)ground_test(context);else if(strcmp(argv[5],"video")==0)video_test(context,options.model,argc==9?argv[8]:NULL);else CHECK(0);
  free(pixels);puts("pure C model client passed; Python=none");return 0;
}
