/* C11 media ownership/index/PNG regression, no Torch headers. */
#include "sam3/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,sam3_last_error());exit(1);}}while(0)
#define OK(x) CHECK((x)==SAM3_OK)
static const char* output_root;
static void dump(const sam3_result* result,const char* name){
  char path[4096];snprintf(path,sizeof(path),"%s/%s.json",output_root,name);FILE* meta=fopen(path,"w");CHECK(meta);fputs("{",meta);int64_t count;OK(sam3_result_field_count(result,&count));
  for(int64_t i=0;i<count;++i){const char* field;sam3_tensor_view v;OK(sam3_result_field_name(result,i,&field));OK(sam3_result_get(result,field,&v));char safe[256];CHECK(strlen(field)<sizeof(safe));strcpy(safe,field);for(char* p=safe;*p;++p)if(*p=='/')*p='_';
    snprintf(path,sizeof(path),"%s/%s-%s.bin",output_root,name,safe);FILE* data=fopen(path,"wb");CHECK(data);CHECK(fwrite(v.data,1,(size_t)v.bytes,data)==v.bytes);CHECK(fclose(data)==0);
    fprintf(meta,"%s\"%s\":{\"dtype\":%d,\"shape\":[",i?",":"",field,v.dtype);for(int64_t j=0;j<v.rank;++j)fprintf(meta,"%s%" PRId64,j?",":"",v.shape[j]);fprintf(meta,"],\"bytes\":%" PRIu64 "}",v.bytes);
  }
  fputs("}\n",meta);CHECK(fclose(meta)==0);printf("%s\n",name);fflush(stdout);
}
int main(int argc,char** argv){
  CHECK(argc>=4);CHECK(sam3_media_available());output_root=argv[3];sam3_media_options options;OK(sam3_media_options_init(&options));options.image_only=strcmp(argv[2],"image")==0;const char* threads=getenv("SAM3_PROBE_THREADS");if(threads)options.threads=atoi(threads);
  const char* color_text=getenv("SAM3_PROBE_COLOR_POLICY");const int color=color_text?atoi(color_text):SAM3_MEDIA_COLOR_STREAM;sam3_media* media=NULL;OK(sam3_media_open_with_color(argv[1],&options,color,&media));sam3_result* info=NULL;OK(sam3_media_info(media,&info));dump(info,"info");sam3_tensor_view count;OK(sam3_result_get(info,"frames",&count));const int64_t frames=*(const int64_t*)count.data;sam3_result_release(info);
  sam3_result* bad=(sam3_result*)1;CHECK(sam3_media_read_frame(media,-1,&bad)==SAM3_INVALID_ARGUMENT && bad==NULL);CHECK(sam3_media_read_frame(media,frames,&bad)==SAM3_INVALID_ARGUMENT && bad==NULL);
  const char* budget=getenv("SAM3_PROBE_CACHE_BYTES");if(budget)OK(sam3_media_set_cache_bytes(media,strtoll(budget,NULL,10)));
  sam3_result* retained=NULL;
  for(int64_t i=0;i<(argc>4?argc-4:frames);++i){const int64_t index=argc>4?strtoll(argv[i+4],NULL,10):i;sam3_result* frame=NULL;OK(sam3_media_read_frame(media,index,&frame));char tag[96];if(getenv("SAM3_PROBE_SEQUENCE"))snprintf(tag,sizeof(tag),"read-%" PRId64 "-frame-%" PRId64,i,index);else snprintf(tag,sizeof(tag),"frame-%" PRId64,index);dump(frame,tag);
    if(!retained){OK(sam3_result_retain(frame,&retained));sam3_tensor_view rgb;OK(sam3_result_get(frame,"rgb",&rgb));const int64_t h=rgb.shape[1],w=rgb.shape[2];uint8_t* pixels=(uint8_t*)malloc((size_t)(h*w*3));CHECK(pixels);const uint8_t* planar=(const uint8_t*)rgb.data;for(int64_t n=0;n<h*w;++n)for(int c=0;c<3;++c)pixels[n*3+c]=planar[c*h*w+n];sam3_rgb_view view={pixels,(uint64_t)(h*w*3),h,w,w*3};char path[4096];snprintf(path,sizeof(path),"%s/roundtrip.png",output_root);OK(sam3_media_write_png(path,&view));free(pixels);}
    sam3_result_release(frame);
  }
  sam3_result* stats=NULL;OK(sam3_media_stats(media,&stats));dump(stats,"stats");sam3_result_release(stats);
  sam3_media_release(media);CHECK(retained);dump(retained,"retained");sam3_result_release(retained);return 0;
}
