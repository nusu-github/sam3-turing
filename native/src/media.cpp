#include "sam3/media.h"
#include <c10/core/InferenceMode.h>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <array>
#ifdef SAM3_WITH_MEDIA
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/display.h>
#include <libavutil/imgutils.h>
#include <libavutil/sha.h>
#include <libswscale/swscale.h>
#include <jpeglib.h>
}
#include <setjmp.h>
namespace sam3 {
namespace {
void avcheck(int status,const char* what){if(status<0){char error[AV_ERROR_MAX_STRING_SIZE];av_strerror(status,error,sizeof(error));TORCH_CHECK(false,what,": ",error);}}
std::string extension(const std::filesystem::path& p){auto s=p.extension().u8string();for(auto& c:s)if(c>='A' && c<='Z')c+=32;return s;}
bool image_extension(const std::filesystem::path& p){const auto s=extension(p);return s==".jpg" || s==".jpeg" || s==".png" || s==".bmp" || s==".tiff" || s==".webp";}
// C allocations keep libjpeg's setjmp error path independent of C++ destructors.
struct JpegState {jpeg_decompress_struct jpeg;jpeg_error_mgr error;jmp_buf jump;char message[JMSG_LENGTH_MAX];unsigned char* pixels;};
void jpeg_error(j_common_ptr info){auto* s=reinterpret_cast<JpegState*>(info);s->error.format_message(info,s->message);longjmp(s->jump,1);}
at::Tensor jpeg_rgb(const std::filesystem::path& path){
  std::ifstream input(path,std::ios::binary);TORCH_CHECK(input,"cannot open JPEG");
  const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});TORCH_CHECK(!bytes.empty() && bytes.size()<=ULONG_MAX,"invalid JPEG length");
  auto* s=static_cast<JpegState*>(std::calloc(1,sizeof(JpegState)));TORCH_CHECK(s,"cannot allocate JPEG state");s->jpeg.err=jpeg_std_error(&s->error);s->error.error_exit=jpeg_error;
  if(setjmp(s->jump)){const std::string message=s->message;jpeg_destroy_decompress(&s->jpeg);std::free(s->pixels);std::free(s);TORCH_CHECK(false,"JPEG decode failed: ",message);}
  jpeg_create_decompress(&s->jpeg);jpeg_mem_src(&s->jpeg,bytes.data(),static_cast<unsigned long>(bytes.size()));jpeg_read_header(&s->jpeg,TRUE);
  const bool cmyk=s->jpeg.jpeg_color_space==JCS_CMYK || s->jpeg.jpeg_color_space==JCS_YCCK;s->jpeg.out_color_space=cmyk?JCS_CMYK:JCS_RGB;
  jpeg_start_decompress(&s->jpeg);const int64_t h=s->jpeg.output_height,w=s->jpeg.output_width,channels=s->jpeg.output_components;
  if(h<=0 || w<=0 || (channels!=3 && channels!=4) || uint64_t(h)>uint64_t(INT64_MAX)/uint64_t(w)/uint64_t(channels) || uint64_t(h)>SIZE_MAX/uint64_t(w)/uint64_t(channels)){jpeg_destroy_decompress(&s->jpeg);std::free(s);TORCH_CHECK(false,"invalid JPEG dimensions");}
  s->pixels=static_cast<unsigned char*>(std::malloc(size_t(h*w*channels)));if(!s->pixels){jpeg_destroy_decompress(&s->jpeg);std::free(s);throw std::bad_alloc();}
  while(s->jpeg.output_scanline<s->jpeg.output_height){JSAMPROW row=s->pixels+size_t(s->jpeg.output_scanline)*w*channels;jpeg_read_scanlines(&s->jpeg,&row,1);}
  jpeg_finish_decompress(&s->jpeg);jpeg_destroy_decompress(&s->jpeg);
  at::Tensor output;
  try{output=at::empty({h,w,3},at::kByte);auto* dst=output.data_ptr<uint8_t>();
    if(!cmyk)std::memcpy(dst,s->pixels,size_t(h*w*3));else for(int64_t i=0;i<h*w;++i)for(int c=0;c<3;++c)dst[i*3+c]=uint8_t((unsigned(s->pixels[i*4+c])*s->pixels[i*4+3]+127)/255);
  }catch(...){std::free(s->pixels);std::free(s);throw;}
  std::free(s->pixels);std::free(s);return output.permute({2,0,1}).contiguous();
}
struct Decoder {
  AVSHA* sha=av_sha_alloc();std::vector<uint8_t> packed;
  std::ifstream input;AVIOContext* io=nullptr;AVFormatContext* format=nullptr;AVCodecContext* codec=nullptr;AVPacket* packet=nullptr;AVFrame* frame=nullptr;SwsContext* scaler=nullptr;int stream=-1,rotation=0;bool draining=false,mirror=false;double fps=0;
  ~Decoder(){av_free(sha);sws_freeContext(scaler);av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&codec);avformat_close_input(&format);if(io){av_freep(&io->buffer);avio_context_free(&io);}}
  static int read_io(void* opaque,uint8_t* data,int size){auto& f=static_cast<Decoder*>(opaque)->input;f.read(reinterpret_cast<char*>(data),size);const auto n=f.gcount();return n?int(n):(f.eof()?AVERROR_EOF:AVERROR(EIO));}
  static int64_t seek_io(void* opaque,int64_t offset,int whence){auto& f=static_cast<Decoder*>(opaque)->input;f.clear();if(whence==AVSEEK_SIZE){const auto pos=f.tellg();f.seekg(0,std::ios::end);const auto size=f.tellg();f.seekg(pos);return size==std::streampos(-1)?AVERROR(EIO):int64_t(size);}whence&=~AVSEEK_FORCE;const auto dir=whence==SEEK_SET?std::ios::beg:whence==SEEK_CUR?std::ios::cur:std::ios::end;if(whence!=SEEK_SET && whence!=SEEK_CUR && whence!=SEEK_END)return AVERROR(EINVAL);f.seekg(offset,dir);return f?int64_t(f.tellg()):AVERROR(EIO);}
  void open(const std::filesystem::path& path,int threads,bool orient){
    input.open(path,std::ios::binary);TORCH_CHECK(input,"cannot open media file: ",path.u8string());
    auto* buffer=static_cast<unsigned char*>(av_malloc(32768));TORCH_CHECK(buffer,"cannot allocate media buffer");io=avio_alloc_context(buffer,32768,0,this,read_io,nullptr,seek_io);if(!io){av_free(buffer);throw std::bad_alloc();}
    format=avformat_alloc_context();TORCH_CHECK(format,"cannot allocate media format");format->pb=io;format->flags|=AVFMT_FLAG_CUSTOM_IO;
    avcheck(avformat_open_input(&format,nullptr,nullptr,nullptr),"open media");avcheck(avformat_find_stream_info(format,nullptr),"inspect media");
    const AVCodec* decoder=nullptr;stream=av_find_best_stream(format,AVMEDIA_TYPE_VIDEO,-1,-1,&decoder,0);avcheck(stream,"find video stream");
    codec=avcodec_alloc_context3(decoder);TORCH_CHECK(codec,"cannot allocate decoder");avcheck(avcodec_parameters_to_context(codec,format->streams[stream]->codecpar),"copy codec parameters");codec->thread_count=threads;avcheck(avcodec_open2(codec,decoder,nullptr),"open decoder");
    packet=av_packet_alloc();frame=av_frame_alloc();TORCH_CHECK(packet && frame,"cannot allocate decoder buffers");const auto rate=av_guess_frame_rate(format,format->streams[stream],nullptr);fps=rate.den?av_q2d(rate):0;
    if(orient){size_t size=0;const auto* matrix=av_stream_get_side_data(format->streams[stream],AV_PKT_DATA_DISPLAYMATRIX,&size);if(matrix && size>=9*sizeof(int32_t)){const auto* m=reinterpret_cast<const int32_t*>(matrix);mirror=double(m[0])*m[4]-double(m[1])*m[3]<0;const double angle=av_display_rotation_get(m);TORCH_CHECK(std::isfinite(angle) && std::abs(angle/90-std::round(angle/90))<1e-4,"non-right-angle display rotation is not supported");rotation=int(std::llround(angle/90));}}
  }
  bool next(){
    while(true){const int status=avcodec_receive_frame(codec,frame);if(status==0)return true;if(status==AVERROR_EOF)return false;avcheck(status==AVERROR(EAGAIN)?0:status,"decode frame");TORCH_CHECK(!draining,"decoder requested packets after drain");
      int read;do{av_packet_unref(packet);read=av_read_frame(format,packet);}while(read>=0 && packet->stream_index!=stream);
      if(read==AVERROR_EOF){avcheck(avcodec_send_packet(codec,nullptr),"drain decoder");draining=true;}else{avcheck(read,"read media packet");avcheck(avcodec_send_packet(codec,packet),"send media packet");}
    }
  }
  bool seek(int64_t timestamp){
    if(av_seek_frame(format,stream,timestamp,AVSEEK_FLAG_BACKWARD)<0)return false;
    avcodec_flush_buffers(codec);av_packet_unref(packet);av_frame_unref(frame);draining=false;return true;
  }
  std::array<uint8_t,32> fingerprint(){
    // Exclude line padding, include palette data and every property used by rgb().
    const auto fmt=AVPixelFormat(frame->format);const int size=av_image_get_buffer_size(fmt,frame->width,frame->height,1);avcheck(size,"size decoded pixels");
    packed.resize(size);avcheck(av_image_copy_to_buffer(packed.data(),size,frame->data,frame->linesize,fmt,frame->width,frame->height,1),"pack decoded pixels");
    TORCH_CHECK(sha,"cannot allocate frame hash");avcheck(av_sha_init(sha,256),"initialize frame hash");
    const int64_t properties[]={frame->width,frame->height,frame->format,frame->colorspace,frame->color_range};
    av_sha_update(sha,reinterpret_cast<const uint8_t*>(properties),sizeof(properties));av_sha_update(sha,packed.data(),packed.size());
    std::array<uint8_t,32> result;av_sha_final(sha,result.data());return result;
  }
  std::pair<int64_t,int64_t> dimensions()const {return rotation%2?std::make_pair(frame->width,frame->height):std::make_pair(frame->height,frame->width);}
  double seconds()const {return frame->best_effort_timestamp==AV_NOPTS_VALUE?std::numeric_limits<double>::quiet_NaN():frame->best_effort_timestamp*av_q2d(format->streams[stream]->time_base);}
  double duration()const {return frame->duration*av_q2d(format->streams[stream]->time_base);}
  at::Tensor rgb(MediaColorPolicy policy=MediaColorPolicy::Stream){
    const bool opencv=policy==MediaColorPolicy::OpenCV;
    TORCH_CHECK(frame->width>0 && frame->width<=INT_MAX/3 && frame->height>0,"invalid decoded dimensions");auto result=at::empty({frame->height,frame->width,3},at::kByte);
    scaler=sws_getCachedContext(scaler,frame->width,frame->height,AVPixelFormat(frame->format),frame->width,frame->height,opencv?AV_PIX_FMT_BGR24:AV_PIX_FMT_RGB24,opencv?SWS_BICUBIC:SWS_BILINEAR,nullptr,nullptr,nullptr);TORCH_CHECK(scaler,"cannot create RGB converter");
    int colorspace=SWS_CS_DEFAULT;switch(frame->colorspace){case AVCOL_SPC_BT709:colorspace=SWS_CS_ITU709;break;case AVCOL_SPC_BT2020_NCL:case AVCOL_SPC_BT2020_CL:colorspace=SWS_CS_BT2020;break;case AVCOL_SPC_FCC:colorspace=SWS_CS_FCC;break;case AVCOL_SPC_SMPTE240M:colorspace=SWS_CS_SMPTE240M;break;default:break;}
    const auto* coeff=sws_getCoefficients(colorspace);if(!opencv)avcheck(sws_setColorspaceDetails(scaler,coeff,frame->color_range==AVCOL_RANGE_JPEG,coeff,1,0,1<<16,1<<16),"set RGB colorspace");
    uint8_t* output[]={result.data_ptr<uint8_t>(),nullptr,nullptr,nullptr};int stride[]={frame->width*3,0,0,0};TORCH_CHECK(sws_scale(scaler,frame->data,frame->linesize,0,frame->height,output,stride)==frame->height,"incomplete RGB conversion");
    result=result.permute({2,0,1});if(opencv)result=result.flip({0});if(mirror)result=result.flip({1});if(rotation%4)result=at::rot90(result,rotation,{1,2});return result.contiguous();
  }
};
at::Tensor image_rgb(const std::filesystem::path& path,int threads){
  // JPEG magic detection also covers extensionless/misnamed image input.
  std::ifstream in(path,std::ios::binary);unsigned char magic[2]={};in.read(reinterpret_cast<char*>(magic),2);if(magic[0]==0xff && magic[1]==0xd8)return jpeg_rgb(path);
  Decoder decoder;decoder.open(path,threads,false);TORCH_CHECK(decoder.next(),"image has no decoded frame");return decoder.rgb();
}
}
struct MediaSource::Impl {
  std::filesystem::path path;MediaOptions options;MediaColorPolicy color;MediaInfo metadata;std::vector<std::filesystem::path> images;std::vector<std::pair<double,double>> timestamps;std::unique_ptr<Decoder> decoder;int64_t next_index=0,cached_index=-1;MediaFrame cached;std::mutex mutex;
  struct IndexedFrame {int64_t pts;std::array<uint8_t,32> digest;};
  std::vector<IndexedFrame> index;std::vector<int64_t> keyframes;
  bool seekable=true,verify_decoder=false;
  MediaStats counters;int64_t cache_limit=64*1024*1024,block=-1;
  std::map<int64_t,at::Tensor> window;
  void reopen(){decoder=std::make_unique<Decoder>();decoder->open(path,options.threads,true);next_index=0;verify_decoder=false;}
  bool next(){const bool okay=decoder->next();if(okay)++counters.read_decoded_frames;return okay;}
  void clear_window(){window.clear();counters.cache_bytes=0;block=-1;}
  bool positioned_read(int64_t target,int64_t start,int64_t capacity){
    while(next_index<=target){
      if(!next())return false;
      if(verify_decoder && (decoder->frame->best_effort_timestamp!=index[next_index].pts || decoder->fingerprint()!=index[next_index].digest))return false;
      if(next_index>=start && capacity>0){auto rgb=decoder->rgb(color);auto it=window.find(next_index);if(it==window.end()){counters.cache_bytes+=rgb.nbytes();window.emplace(next_index,std::move(rgb));}}
      ++next_index;
    }
    return true;
  }
  bool seek_read(int64_t target,int64_t start,int64_t capacity){
    auto key=std::upper_bound(keyframes.begin(),keyframes.end(),start);if(key==keyframes.begin())return false;--key;
    ++counters.seek_attempts;
    if(!decoder->seek(index[*key].pts) || !next())return false;
    const auto pts=decoder->frame->best_effort_timestamp;
    auto found=std::lower_bound(index.begin(),index.end(),pts,[](const auto& a,int64_t b){return a.pts<b;});
    if(found==index.end() || found->pts!=pts || found-index.begin()>start)return false;
    next_index=found-index.begin();verify_decoder=true;
    if(decoder->fingerprint()!=found->digest)return false;
    if(next_index>=start && capacity>0){auto rgb=decoder->rgb(color);counters.cache_bytes+=rgb.nbytes();window.emplace(next_index,std::move(rgb));}
    ++next_index;return positioned_read(target,start,capacity);
  }
  Impl(const std::filesystem::path& p,const MediaOptions& o,MediaColorPolicy c):path(p),options(o),color(c){
    TORCH_CHECK(c==MediaColorPolicy::Stream || c==MediaColorPolicy::OpenCV,"invalid media color policy");
    TORCH_CHECK(o.threads>0,"decoder threads must be positive");TORCH_CHECK(std::filesystem::exists(p),"media path does not exist");
    if(std::filesystem::is_directory(p)){
      for(const auto& entry:std::filesystem::directory_iterator(p))if(entry.is_regular_file() && image_extension(entry.path()))images.push_back(entry.path());
      TORCH_CHECK(!images.empty(),"image folder is empty");std::sort(images.begin(),images.end(),[](const auto& a,const auto& b){return a.filename().u8string()<b.filename().u8string();});
      bool numeric=true;std::map<std::filesystem::path,int64_t> numbers;for(auto& entry:images){try{size_t used=0;const auto stem=entry.stem().u8string();auto n=std::stoll(stem,&used);if(used!=stem.size())throw std::invalid_argument("name");numbers[entry]=n;}catch(const std::exception&){numeric=false;break;}}
      if(numeric)std::stable_sort(images.begin(),images.end(),[&](auto& a,auto& b){return numbers[a]<numbers[b];});
      TORCH_CHECK(!o.image_only || images.size()==1,"image-only mode requires one image");
    }else if(o.image_only || image_extension(p))images.push_back(p);
    metadata.image_only=o.image_only;
    if(!images.empty()){cached={image_rgb(images.front(),o.threads),0,0};cached_index=0;metadata.frames=images.size();metadata.height=cached.rgb.size(1);metadata.width=cached.rgb.size(2);}
    else {auto scan=std::make_unique<Decoder>();scan->open(path,o.threads,true);metadata.fps=scan->fps;
      while(scan->next()){const auto [h,w]=scan->dimensions();if(timestamps.empty()){metadata.height=h;metadata.width=w;}TORCH_CHECK(h==metadata.height && w==metadata.width,"video changes dimensions");timestamps.emplace_back(scan->seconds(),scan->duration());
        const auto pts=scan->frame->best_effort_timestamp;if(pts==AV_NOPTS_VALUE || (!index.empty() && pts<=index.back().pts))seekable=false;
        if(scan->frame->flags&AV_FRAME_FLAG_KEY)keyframes.push_back(index.size());
        index.push_back({pts,scan->fingerprint()});++counters.index_decoded_frames;}
      TORCH_CHECK(!timestamps.empty(),"video has no decoded frames");metadata.frames=timestamps.size();
    }
  }
  MediaFrame read(int64_t requested){
    std::lock_guard<std::mutex> guard(mutex);TORCH_CHECK(requested>=0 && requested<metadata.frames,"media frame index out of range");
    if(requested==cached_index){++counters.cache_hits;return {cached.rgb.clone(),cached.seconds,cached.duration};}
    if(!images.empty())cached={image_rgb(images[requested],options.threads),0,0};
    else {
      const int64_t frame_bytes=metadata.height*metadata.width*3;
      const int64_t capacity=cache_limit/frame_bytes;
      const int64_t start=capacity?(requested/capacity)*capacity:requested;
      if(block!=start){clear_window();block=start;}
      const auto hit=window.find(requested);
      if(hit!=window.end()){++counters.cache_hits;cached={hit->second,timestamps[requested].first,timestamps[requested].second};}
      else {
        if(!decoder)reopen();
        bool okay=false;
        // Seek only when there is a keyframe beyond the sequential cursor, or
        // when moving backwards. Never identify frames using rounded seconds.
        auto key=std::upper_bound(keyframes.begin(),keyframes.end(),start);
        const bool later_key=key!=keyframes.begin() && *std::prev(key)>next_index;
        if(seekable && !keyframes.empty() && (requested<next_index || later_key)){
          try{okay=seek_read(requested,start,capacity);}catch(const c10::Error&){okay=false;}
          if(!okay){++counters.seek_fallbacks;seekable=false;clear_window();block=start;reopen();}
        }else {if(requested<next_index)reopen();}
        if(!okay){try{okay=positioned_read(requested,start,capacity);}catch(const c10::Error&){if(!verify_decoder)throw;okay=false;}
          if(!okay && verify_decoder){++counters.seek_fallbacks;seekable=false;clear_window();block=start;reopen();okay=positioned_read(requested,start,capacity);}
        }
        TORCH_CHECK(okay,"media changed since index scan");
        auto found=window.find(requested);cached={found==window.end()?decoder->rgb(color):found->second,timestamps[requested].first,timestamps[requested].second};
      }
    }
    cached_index=requested;return {cached.rgb.clone(),cached.seconds,cached.duration};
  }
};
MediaSource::MediaSource(const std::filesystem::path& p,const MediaOptions& o):MediaSource(p,o,MediaColorPolicy::Stream){}
MediaSource::MediaSource(const std::filesystem::path& p,const MediaOptions& o,MediaColorPolicy c):impl_(std::make_unique<Impl>(p,o,c)){}
MediaSource::~MediaSource()=default;
const MediaInfo& MediaSource::info()const{return impl_->metadata;}
MediaFrame MediaSource::read(int64_t index){c10::InferenceMode inference(false);return impl_->read(index);}
MediaStats MediaSource::stats()const{std::lock_guard<std::mutex> guard(impl_->mutex);auto result=impl_->counters;result.cache_limit_bytes=impl_->cache_limit;result.indexed_seek_enabled=impl_->seekable && !impl_->keyframes.empty();return result;}
void MediaSource::set_cache_bytes(int64_t bytes){TORCH_CHECK(bytes>=0,"cache bytes must be nonnegative");std::lock_guard<std::mutex> guard(impl_->mutex);impl_->clear_window();impl_->cache_limit=bytes;}
bool media_available()noexcept{return true;}
void write_rgb_png(const std::filesystem::path& path,const at::Tensor& rgb){
  TORCH_CHECK(rgb.dim()==3 && rgb.size(0)==3 && rgb.scalar_type()==at::kByte && rgb.size(1)>0 && rgb.size(2)>0 && rgb.size(1)<=INT_MAX && rgb.size(2)<=INT_MAX/3,"PNG output requires U8 RGB [3,H,W]");
  Decoder owner;const auto* encoder=avcodec_find_encoder(AV_CODEC_ID_PNG);TORCH_CHECK(encoder,"PNG encoder unavailable");owner.codec=avcodec_alloc_context3(encoder);TORCH_CHECK(owner.codec,"cannot allocate PNG encoder");owner.codec->width=rgb.size(2);owner.codec->height=rgb.size(1);owner.codec->pix_fmt=AV_PIX_FMT_RGB24;owner.codec->time_base={1,1};avcheck(avcodec_open2(owner.codec,encoder,nullptr),"open PNG encoder");
  owner.frame=av_frame_alloc();owner.packet=av_packet_alloc();TORCH_CHECK(owner.frame && owner.packet,"cannot allocate PNG buffers");const auto pixels=rgb.cpu().permute({1,2,0}).contiguous();owner.frame->format=AV_PIX_FMT_RGB24;owner.frame->width=rgb.size(2);owner.frame->height=rgb.size(1);owner.frame->data[0]=const_cast<uint8_t*>(pixels.const_data_ptr<uint8_t>());owner.frame->linesize[0]=rgb.size(2)*3;avcheck(avcodec_send_frame(owner.codec,owner.frame),"encode PNG");avcheck(avcodec_receive_packet(owner.codec,owner.packet),"receive PNG");std::ofstream out(path,std::ios::binary);TORCH_CHECK(out,"cannot open PNG output");out.write(reinterpret_cast<const char*>(owner.packet->data),owner.packet->size);out.close();TORCH_CHECK(out,"cannot write PNG output");
}
}
#else
namespace sam3 {
struct MediaSource::Impl{};
MediaSource::MediaSource(const std::filesystem::path&,const MediaOptions&){TORCH_CHECK(false,"native media support disabled; build with SAM3_WITH_MEDIA=ON");}
MediaSource::MediaSource(const std::filesystem::path&,const MediaOptions&,MediaColorPolicy){TORCH_CHECK(false,"native media support disabled; build with SAM3_WITH_MEDIA=ON");}
MediaSource::~MediaSource()=default;
const MediaInfo& MediaSource::info()const{TORCH_CHECK(false,"native media support disabled");}
MediaFrame MediaSource::read(int64_t){TORCH_CHECK(false,"native media support disabled");}
MediaStats MediaSource::stats()const{TORCH_CHECK(false,"native media support disabled");}
void MediaSource::set_cache_bytes(int64_t){TORCH_CHECK(false,"native media support disabled");}
bool media_available()noexcept{return false;}
void write_rgb_png(const std::filesystem::path&,const at::Tensor&){TORCH_CHECK(false,"native media support disabled");}
}
#endif
