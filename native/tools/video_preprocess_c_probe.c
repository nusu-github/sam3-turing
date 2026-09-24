/* C11 upstream preprocessing oracle client; reads RGB bytes, writes F32 NCHW. */
#include "sam3/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,sam3_last_error());return 1;}}while(0)
#define OK(x) CHECK((x)==SAM3_OK)
int main(int argc,char** argv){
  CHECK(argc==7);const int64_t h=strtoll(argv[2],NULL,10),w=strtoll(argv[3],NULL,10);const int policy=atoi(argv[4]);CHECK(h>0 && w>0 && h<=INT64_MAX/w/3);const uint64_t bytes=(uint64_t)(h*w*3);CHECK(bytes<=SIZE_MAX);
  FILE* in=fopen(argv[1],"rb");CHECK(in);uint8_t* data=(uint8_t*)malloc((size_t)bytes);CHECK(data);CHECK(fread(data,1,(size_t)bytes,in)==bytes);CHECK(fclose(in)==0);
  OK(sam3_runtime_configure(1,0));sam3_rgb_view rgb={data,bytes,h,w,w*3};sam3_result* out=(sam3_result*)1;
  CHECK(sam3_preprocess_video_rgb(&rgb,99,"cpu",&out)==SAM3_INVALID_ARGUMENT && !out);CHECK(!sam3_video_preprocess_available(99));
  CHECK(sam3_video_preprocess_available(policy));OK(sam3_preprocess_video_rgb(&rgb,policy,argv[5],&out));free(data);sam3_tensor_view image;OK(sam3_result_get(out,"image",&image));CHECK(image.dtype==SAM3_F32 && image.rank==4 && image.shape[0]==1 && image.shape[1]==3 && image.shape[2]==1008 && image.shape[3]==1008);
  FILE* f=fopen(argv[6],"wb");CHECK(f);CHECK(fwrite(image.data,1,(size_t)image.bytes,f)==image.bytes);CHECK(fclose(f)==0);sam3_result_release(out);return 0;
}
