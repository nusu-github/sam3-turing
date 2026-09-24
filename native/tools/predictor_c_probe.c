/* Pure C11 owning predictor regression; input/output plumbing only. */
#include "sam3/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,sam3_last_error());exit(1);}}while(0)
#define OK(x) CHECK((x)==SAM3_OK)
static const char *frames_root,*output_root;
static int64_t height,width,latest=-1;
static uint8_t* pixels;
static sam3_predictor* running;
static sam3_result* retained;
static int failure,callbacks,callback_mode;
static const char* suffix;
static int32_t provider(void* user,int64_t frame,sam3_rgb_view* out){
  (void)user;sam3_result* invalid=NULL;
  if(running)CHECK(sam3_predictor_info(running,&invalid)==SAM3_BUSY && invalid==NULL);
  if(running)CHECK(sam3_predictor_output_cache_stats(running,&invalid)==SAM3_BUSY && invalid==NULL);
  if(failure)return -1;
  char path[4096];snprintf(path,sizeof(path),"%s/%" PRId64 ".rgb",frames_root,frame);FILE* f=fopen(path,"rb");CHECK(f);
  const int64_t row=width*3+7;free(pixels);pixels=(uint8_t*)malloc((size_t)(height*row));CHECK(pixels);memset(pixels,0xcd,(size_t)(height*row));
  for(int64_t y=0;y<height;++y)CHECK(fread(pixels+y*row,1,(size_t)(width*3),f)==(size_t)(width*3));CHECK(fgetc(f)==EOF);CHECK(fclose(f)==0);
  *out=(sam3_rgb_view){pixels,(uint64_t)(height*row),height,width,row};latest=frame;return 0;
}
static void dump(const sam3_result* result,const char* name){
  char path[4096];snprintf(path,sizeof(path),"%s/%s.json",output_root,name);FILE* meta=fopen(path,"w");CHECK(meta);fputs("{",meta);int64_t count;OK(sam3_result_field_count(result,&count));
  for(int64_t i=0;i<count;++i){const char* field;sam3_tensor_view v;OK(sam3_result_field_name(result,i,&field));OK(sam3_result_get(result,field,&v));char safe[256];CHECK(strlen(field)<sizeof(safe));strcpy(safe,field);for(char* p=safe;*p;++p)if(*p=='/')*p='_';
    snprintf(path,sizeof(path),"%s/%s-%s.bin",output_root,name,safe);FILE* data=fopen(path,"wb");CHECK(data);CHECK(fwrite(v.data,1,(size_t)v.bytes,data)==v.bytes);CHECK(fclose(data)==0);
    fprintf(meta,"%s\"%s\":{\"dtype\":%d,\"shape\":[",i?",":"",field,v.dtype);for(int64_t j=0;j<v.rank;++j)fprintf(meta,"%s%" PRId64,j?",":"",v.shape[j]);fprintf(meta,"],\"bytes\":%" PRIu64 "}",v.bytes);
  }
  fputs("}\n",meta);CHECK(fclose(meta)==0);printf("%s emitted_at=%" PRId64 "\n",name,latest);fflush(stdout);
  snprintf(path,sizeof(path),"%s/%s.emission.txt",output_root,name);meta=fopen(path,"w");CHECK(meta);fprintf(meta,"%" PRId64 "\n",latest);CHECK(fclose(meta)==0);
}
static int64_t scalar(const sam3_result* result,const char* key){sam3_tensor_view v;OK(sam3_result_get(result,key,&v));CHECK(v.dtype==SAM3_I64 && v.bytes==8);return *(const int64_t*)v.data;}
static int64_t first_id(const sam3_result* result){sam3_tensor_view ids;OK(sam3_result_get(result,"ids",&ids));CHECK(ids.dtype==SAM3_I64 && ids.bytes>=8);return *(const int64_t*)ids.data;}
static void output(sam3_result* result,const char* name){dump(result,name);sam3_result_release(result);}
static int32_t callback(void* user,const sam3_result* result){
  (void)user;++callbacks;sam3_result* invalid=NULL;CHECK(sam3_predictor_info(running,&invalid)==SAM3_BUSY && invalid==NULL);
  CHECK(sam3_predictor_output_cache_stats(running,&invalid)==SAM3_BUSY && invalid==NULL);
  if(!retained)OK(sam3_result_retain(result,&retained));
  if(callback_mode==1){OK(sam3_predictor_cancel(running));return 0;} /* cancel inside a buffered batch */
  if(callback_mode==2)return 1;
  if(callback_mode==3)return -1;
  if(suffix){char tag[100];snprintf(tag,sizeof(tag),"%" PRId64 "%s",scalar(result,"frame"),suffix);dump(result,tag);}return 0;
}
static void propagate(int64_t start,int64_t steps,const char* tag){suffix=tag;callbacks=0;OK(sam3_predictor_propagate(running,start,steps,0,0,callback,NULL));}
int main(int argc,char** argv){
  CHECK(argc==12 || argc==13 || argc==14 || argc==16); /* STORE MODEL DEVICE PRECISION RGB_DIR H W BPE PROMPT OUTPUT video|image [TRACKING_DEVICES [PARALLEL [CACHE_STORAGE CACHE_DIRECTORY]]] */
  const int model=atoi(argv[2]),precision=atoi(argv[4]),image=strcmp(argv[11],"image")==0;CHECK(image || strcmp(argv[11],"video")==0);
  frames_root=argv[5];height=strtoll(argv[6],NULL,10);width=strtoll(argv[7],NULL,10);output_root=argv[10];CHECK(height>0 && width>0);
  FILE* prompt_file=fopen(argv[9],"rb");CHECK(prompt_file);CHECK(fseek(prompt_file,0,SEEK_END)==0);long length=ftell(prompt_file);CHECK(length>=0);rewind(prompt_file);char* text=(char*)malloc((size_t)length+1);CHECK(text);CHECK(fread(text,1,(size_t)length,prompt_file)==(size_t)length);fclose(prompt_file);while(length && (text[length-1]=='\n' || text[length-1]=='\r'))--length;
  sam3_utf8_view utf8={text,(uint64_t)length};sam3_semantic_prompt prompt;OK(sam3_semantic_prompt_init(&prompt));prompt.text=&utf8;
  OK(sam3_runtime_configure(4,0));sam3_context_options context_options;OK(sam3_context_options_init(&context_options));context_options.weight_directory=argv[1];context_options.model=model;context_options.precision=precision;context_options.device=argv[3];context_options.vocabulary_path=argv[8];sam3_context* context=NULL;OK(sam3_context_create(&context_options,&context));
  sam3_predictor_options options;OK(sam3_predictor_options_init(&options,model));options.tracking.frames=image?1:34;options.tracking.height=height;options.tracking.width=width;options.tracking.provider=provider;options.image_only=image;options.centers=1;
  sam3_predictor* other=NULL;sam3_predictor_options invalid=options;invalid.output_batch_size=0;CHECK(sam3_predictor_create(context,&invalid,&other)==SAM3_INVALID_ARGUMENT && other==NULL);
  invalid=options;invalid.model=model==3?31:3;CHECK(sam3_predictor_create(context,&invalid,&other)==SAM3_INVALID_ARGUMENT && other==NULL);
  OK(sam3_predictor_create(context,&options,&running));OK(sam3_predictor_create(context,&options,&other));OK(sam3_context_trim(context));sam3_context_release(context); /* modules and children survive */
  CHECK(sam3_predictor_set_parallel_tracking(running,2)==SAM3_INVALID_ARGUMENT);
  if(argc>=14){OK(sam3_predictor_set_parallel_tracking(running,atoi(argv[13])));OK(sam3_predictor_set_parallel_tracking(other,atoi(argv[13])));}
  if(argc>=13){char* names=(char*)malloc(strlen(argv[12])+1);CHECK(names);strcpy(names,argv[12]);const char** devices=(const char**)malloc((strlen(names)+1)*sizeof(char*));CHECK(devices);int64_t n=0;for(char* token=strtok(names,",");token;token=strtok(NULL,","))devices[n++]=token;
    CHECK(sam3_predictor_set_tracking_devices(running,NULL,0)==SAM3_INVALID_ARGUMENT);OK(sam3_predictor_set_tracking_devices(running,devices,n));OK(sam3_predictor_set_tracking_devices(other,devices,n));free(devices);free(names);}
  CHECK(sam3_predictor_set_output_cache(running,3,NULL)==SAM3_INVALID_ARGUMENT);
  CHECK(sam3_predictor_set_output_cache(running,SAM3_OUTPUT_CACHE_PACKED_DISK,NULL)==SAM3_INVALID_ARGUMENT);
  const int cache_storage=argc==16?atoi(argv[14]):0;
  if(argc==16){OK(sam3_predictor_set_output_cache(running,cache_storage,argv[15]));OK(sam3_predictor_set_output_cache(other,cache_storage,argv[15]));}
  sam3_result* result=NULL;failure=1;CHECK(sam3_predictor_add_prompt(running,0,&prompt,&result)==SAM3_CALLBACK_ERROR && result==NULL);failure=0;OK(sam3_predictor_reset(running));
  if(image){
    OK(sam3_predictor_add_prompt(running,0,&prompt,&result));dump(result,"semantic");const int64_t id=first_id(result);OK(sam3_result_retain(result,&retained));sam3_result_release(result);
    int64_t es[]={0,2},ls[]={0};sam3_tensor_view empty={NULL,0,SAM3_F32,2,es},labels={NULL,0,SAM3_I64,1,ls};
    OK(sam3_predictor_add_points(running,0,id,&empty,&labels,NULL,1,1,0,1,&result));output(result,"empty_stateless");
    float xy[]={.45f,.55f};int64_t lab[]={1},ps[]={1,2},lbs[]={1};sam3_tensor_view point={xy,sizeof(xy),SAM3_F32,2,ps},label={lab,sizeof(lab),SAM3_I64,1,lbs};
    OK(sam3_predictor_add_points(running,0,id,&point,&label,NULL,1,1,0,0,&result));output(result,"point");
    OK(sam3_predictor_add_points(running,0,id,&empty,&labels,NULL,1,1,0,0,&result));output(result,"cleared");
    OK(sam3_predictor_add_points(running,0,id,&empty,&labels,NULL,1,1,0,0,&result));output(result,"cleared_twice");
    uint8_t* mask=(uint8_t*)calloc((size_t)(height*width),1);CHECK(mask);for(int64_t y=height/4;y<height/2;++y)for(int64_t x=width/4;x<width/2;++x)mask[y*width+x]=1;mask[0]=1;int64_t shape[]={height,width};sam3_tensor_view mv={mask,(uint64_t)(height*width),SAM3_BOOL,2,shape};
    OK(sam3_predictor_add_mask(running,0,9000,&mv,&result));output(result,"mask_new");
    OK(sam3_predictor_add_points(running,0,9000,&point,&label,NULL,1,1,0,0,&result));output(result,"mask_point");
    OK(sam3_predictor_add_points(running,0,9000,&empty,&labels,NULL,1,1,0,0,&result));output(result,"mask_cleared");
    for(int64_t i=0;i<height*width;++i)mask[i]=!mask[i];OK(sam3_predictor_add_mask(running,0,9000,&mv,&result));output(result,"mask_replaced");free(mask);OK(sam3_predictor_remove_object(running,9000));
    OK(sam3_predictor_reset(running));OK(sam3_predictor_add_prompt(running,0,&prompt,&result));output(result,"reset");OK(sam3_predictor_info(running,&result));CHECK(scalar(result,"visual_encodes")==2);sam3_result_release(result);
  }else{
    /* Full propagation buffers outputs. All cancellation forms stop at one. */
    for(callback_mode=1;callback_mode<=3;++callback_mode){OK(sam3_predictor_add_prompt(running,0,&prompt,&result));sam3_result_release(result);callbacks=0;const sam3_status expected=callback_mode==3?SAM3_CALLBACK_ERROR:SAM3_OK;CHECK(sam3_predictor_propagate(running,0,3,0,0,callback,NULL)==expected);CHECK(callbacks==1);sam3_result_release(retained);retained=NULL;}
    callback_mode=0;OK(sam3_predictor_add_prompt(running,0,&prompt,&result));output(result,"0.person");propagate(0,3,".person_track");CHECK(callbacks==4);
    float boxes[]={.30f,.15f,.35f,.70f};int64_t labels[]={1},bs[]={1,4},ls[]={1};sam3_tensor_view bv={boxes,sizeof(boxes),SAM3_F32,2,bs},lv={labels,sizeof(labels),SAM3_I64,1,ls};sam3_semantic_prompt box;OK(sam3_semantic_prompt_init(&box));box.boxes_xywh=&bv;box.box_labels=&lv;
    OK(sam3_predictor_add_prompt(running,1,&box,&result));output(result,"1.box");propagate(1,1,".box_track");
    box.text=&utf8;labels[0]=0;OK(sam3_predictor_add_prompt(running,2,&box,&result));output(result,"2.text_box");OK(sam3_predictor_fetch(running,2,&result));output(result,"2.fetch");
    OK(sam3_predictor_reset(running));OK(sam3_predictor_add_prompt(running,0,&prompt,&result));output(result,"0.reset_person");
    /* Fresh instance-only session, followed by mask-only and reverse tracking. */
    OK(sam3_predictor_reset(running));float xy[]={.45f,.55f};int64_t lab[]={1},ps[]={1,2},lbs[]={1};sam3_tensor_view point={xy,sizeof(xy),SAM3_F32,2,ps},label={lab,sizeof(lab),SAM3_I64,1,lbs};OK(sam3_predictor_add_points(running,0,9000,&point,&label,NULL,1,1,0,0,&result));output(result,"0.point_only");callback_mode=2;propagate(0,2,NULL);CHECK(callbacks==1);callback_mode=0;OK(sam3_predictor_remove_object(running,9000));
    uint8_t* mask=(uint8_t*)calloc((size_t)(height*width),1);CHECK(mask);for(int64_t y=height/4;y<height/2;++y)for(int64_t x=width/4;x<width/2;++x)mask[y*width+x]=1;int64_t shape[]={height,width};sam3_tensor_view mv={mask,(uint64_t)(height*width),SAM3_BOOL,2,shape};OK(sam3_predictor_add_mask(running,0,9001,&mv,&result));output(result,"0.mask_only");free(mask);propagate(0,2,NULL);CHECK(callbacks==3);
    callbacks=0;OK(sam3_predictor_propagate(running,2,2,1,0,callback,NULL));CHECK(callbacks==2);OK(sam3_predictor_remove_object(running,9001));
  }
  /* Other owner's state was never changed, despite sharing all heavy cores. */
  OK(sam3_predictor_info(other,&result));CHECK(scalar(result,"visual_encodes")==0);sam3_tensor_view ids;OK(sam3_result_get(result,"ids",&ids));CHECK(ids.bytes==0);sam3_result_release(result);
  OK(sam3_predictor_output_cache_stats(running,&result));CHECK(scalar(result,"storage")==cache_storage && scalar(result,"inspection_pinned")==0);if(cache_storage)CHECK(scalar(result,"resident_bytes")==0);printf("cache frames=%" PRId64 " masks=%" PRId64 " resident=%" PRId64 " packed=%" PRId64 " disk=%" PRId64 "\n",scalar(result,"frames"),scalar(result,"masks"),scalar(result,"resident_bytes"),scalar(result,"packed_bytes"),scalar(result,"disk_bytes"));sam3_result_release(result);
  sam3_predictor_release(running);running=NULL;sam3_predictor_release(other);CHECK(retained);dump(retained,"retained_after_destroy");sam3_result_release(retained);free(pixels);free(text);puts("pure C predictor lifecycle, callbacks and ownership passed");return 0;
}
