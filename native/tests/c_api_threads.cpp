#include "sam3/c_api.h"
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
int main(){
  std::mutex mutex;std::condition_variable ready;int arrived=0;std::atomic<bool> valid{true};
  const auto run=[&](bool options){
    const auto status=options?sam3_context_options_init(nullptr):sam3_video_cancel(nullptr);
    {std::unique_lock<std::mutex> guard(mutex);++arrived;ready.notify_all();ready.wait(guard,[&]{return arrived==2;});}
    if(status!=SAM3_INVALID_ARGUMENT || std::strcmp(sam3_last_error(),options?"options are required":"video is required")!=0)valid=false;
  };
  std::thread first(run,true),second(run,false);first.join();second.join();
  if(!valid){std::cerr<<"C API errors leaked between threads\n";return 1;}
  std::cout<<"C ABI thread-local errors remain independent\n";return 0;
}
