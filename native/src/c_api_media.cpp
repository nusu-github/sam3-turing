#include "media_internal.h"
using namespace sam3::api;
namespace {void available(){if(!sam3::media_available())throw Failure(SAM3_UNSUPPORTED,"build with SAM3_WITH_MEDIA=ON to enable native media");}}
extern "C" {
int32_t sam3_media_available(void) noexcept{return sam3::media_available();}
sam3_status sam3_media_options_init(sam3_media_options* o) noexcept{return protect([&]{require(o,"media options are required");*o={};o->struct_size=sizeof(*o);o->threads=1;});}
sam3_status sam3_media_open(const char* path,const sam3_media_options* o,sam3_media** out) noexcept{return protect([&]{output(out);available();options(o);require(path && *path && o->threads>0,"path and positive thread count are required");auto handle=std::make_unique<sam3_media>();handle->value=std::make_shared<sam3::MediaSource>(std::filesystem::u8path(path),sam3::MediaOptions{bool(o->image_only),o->threads});*out=handle.release();});}
void sam3_media_release(sam3_media* p) noexcept{delete p;}
sam3_status sam3_media_info(sam3_media* p,sam3_result** out) noexcept{return protect([&]{output(out);require(p,"media is required");const auto& i=p->value->info();result({{"frames",at::scalar_tensor(i.frames,at::kLong)},{"height",at::scalar_tensor(i.height,at::kLong)},{"width",at::scalar_tensor(i.width,at::kLong)},{"fps",at::scalar_tensor(i.fps,at::kDouble)},{"image_only",at::scalar_tensor(i.image_only,at::kBool)}},out);});}
sam3_status sam3_media_read_frame(sam3_media* p,int64_t frame,sam3_result** out) noexcept{return protect([&]{output(out);require(p && frame>=0 && frame<p->value->info().frames,"media and in-range frame index are required");const auto f=p->value->read(frame);result({{"rgb",f.rgb},{"seconds",at::scalar_tensor(f.seconds,at::kDouble)},{"duration",at::scalar_tensor(f.duration,at::kDouble)}},out);});}
sam3_status sam3_media_write_png(const char* path,const sam3_rgb_view* view) noexcept{return protect([&]{available();require(path && *path && view,"output path and RGB view are required");sam3::write_rgb_png(std::filesystem::u8path(path),rgb(*view));});}
}
