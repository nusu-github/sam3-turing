#include "sam3/media.h"
#include "sam3/c_api.h"
#include <ATen/Parallel.h>
#include <chrono>
#include <fstream>
#include <thread>
#include <atomic>
#include <iostream>
int main(){try{
  at::set_num_threads(1);sam3_media_options options;TORCH_CHECK(sam3_media_options_init(&options)==SAM3_OK,"media defaults");sam3_media* handle=reinterpret_cast<sam3_media*>(1);
  if(!sam3::media_available()){TORCH_CHECK(sam3_media_open("unused",&options,&handle)==SAM3_UNSUPPORTED && !handle,"disabled capability contract");return 0;}
  const auto root=std::filesystem::temp_directory_path()/std::filesystem::u8path("sam3-media-画像-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directories(root);
  struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code ec;std::filesystem::remove_all(path,ec);}} cleanup{root};
  const auto pixels=at::arange(3*5*7,at::kInt).to(at::kByte).reshape({3,5,7});sam3::write_rgb_png(root/"sample.png",pixels);sam3::MediaSource source(root/"sample.png",{true,1});TORCH_CHECK(source.info().frames==1 && source.info().height==5 && source.info().width==7,"PNG metadata");auto first=source.read(0);TORCH_CHECK(at::equal(first.rgb,pixels),"PNG roundtrip");first.rgb.zero_();TORCH_CHECK(at::equal(source.read(0).rgb,pixels),"source cache aliases returned pixels");
  std::atomic<bool> okay{true};std::vector<std::thread> threads;for(int i=0;i<4;++i)threads.emplace_back([&]{try{for(int j=0;j<4;++j)if(!at::equal(source.read(0).rgb,pixels))okay=false;}catch(...){okay=false;}});for(auto& t:threads)t.join();TORCH_CHECK(okay,"concurrent source reads");
  options.image_only=1;TORCH_CHECK(sam3_media_open((root/"sample.png").u8string().c_str(),&options,&handle)==SAM3_OK,"C source open");sam3_result* result=nullptr;TORCH_CHECK(sam3_media_read_frame(handle,0,&result)==SAM3_OK,"C frame read");sam3_media_release(handle);sam3_tensor_view view{};TORCH_CHECK(sam3_result_get(result,"rgb",&view)==SAM3_OK && view.bytes==pixels.nbytes(),"result survives source");sam3_result_release(result);
  std::ofstream corrupt(root/"corrupt.jpg",std::ios::binary);const unsigned char bytes[]={255,216,255,255};corrupt.write(reinterpret_cast<const char*>(bytes),sizeof(bytes));corrupt.close();TORCH_CHECK(sam3_media_open((root/"corrupt.jpg").u8string().c_str(),&options,&handle)==SAM3_RUNTIME_ERROR && !handle,"corrupt JPEG must return an error, not abort");
  TORCH_CHECK(sam3_media_open((root/"missing").u8string().c_str(),&options,&handle)==SAM3_RUNTIME_ERROR && !handle,"missing file");options.threads=0;TORCH_CHECK(sam3_media_open("unused",&options,&handle)==SAM3_INVALID_ARGUMENT && !handle,"invalid threads");
  std::cout<<"media PNG ownership, Unicode, concurrent reads and errors passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
