#include "c_api_internal.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <limits>
namespace sam3::api {
namespace {thread_local char message[4096]{};
at::ScalarType scalar(int32_t value){switch(value){case SAM3_U8:return at::kByte;case SAM3_BOOL:return at::kBool;case SAM3_I32:return at::kInt;case SAM3_I64:return at::kLong;case SAM3_F16:return at::kHalf;case SAM3_BF16:return at::kBFloat16;case SAM3_F32:return at::kFloat;case SAM3_F64:return at::kDouble;default:throw Failure(SAM3_INVALID_ARGUMENT,"unsupported tensor dtype");}}
}
void error(const char* value) noexcept {std::strncpy(message,value?value:"unknown error",sizeof(message)-1);message[sizeof(message)-1]=0;}
int32_t dtype(at::ScalarType value){switch(value){case at::kByte:return SAM3_U8;case at::kBool:return SAM3_BOOL;case at::kInt:return SAM3_I32;case at::kLong:return SAM3_I64;case at::kHalf:return SAM3_F16;case at::kBFloat16:return SAM3_BF16;case at::kFloat:return SAM3_F32;case at::kDouble:return SAM3_F64;default:throw Failure(SAM3_UNSUPPORTED,"result dtype has no C representation");}}
at::Tensor tensor(const sam3_tensor_view* v){
  if(!v)return {};require(v->rank>=0 && v->rank<=64 && (v->rank==0 || v->shape),"invalid tensor dimensions");
  const auto type=scalar(v->dtype);uint64_t count=1;
  for(int64_t i=0;i<v->rank;++i){require(v->shape[i]>=0,"negative tensor dimension");const uint64_t size=v->shape[i];require(size==0 || count<=uint64_t(INT64_MAX)/size,"tensor size overflow");count*=size;}
  const uint64_t element=c10::elementSize(type);require(count<=uint64_t(INT64_MAX)/element,"tensor byte size overflow");
  require(v->bytes>=count*element && (count==0 || v->data),"tensor buffer is missing or undersized");
  const at::IntArrayRef shape(v->shape,v->rank);if(!count)return at::empty(shape,at::TensorOptions().dtype(type));
  return at::from_blob(const_cast<void*>(v->data),shape,at::TensorOptions().dtype(type)).clone();
}
at::Tensor rgb(const sam3_rgb_view& v){
  require(v.height>0 && v.width>0 && v.width<=INT64_MAX/3,"invalid RGB dimensions");const auto row=v.row_bytes?v.row_bytes:v.width*3;
  require(row>=v.width*3 && row<=INT64_MAX/v.height,"invalid RGB row stride");const auto bytes=uint64_t(v.height-1)*uint64_t(row)+uint64_t(v.width)*3;
  require(v.data && v.bytes>=bytes,"RGB buffer is missing or undersized");
  return at::from_blob(const_cast<uint8_t*>(v.data),{v.height,v.width,3},{row,3,1},at::kByte).permute({2,0,1}).contiguous().clone();
}
Context::Context(const sam3_context_options& o):store(std::filesystem::u8path(o.weight_directory)),model(o.model==SAM3_MODEL_3?"sam3":"sam3.1"),
    mode(o.precision==SAM3_FP32?"fp32":o.precision==SAM3_FP16?"fp16":"bf16_reference"),device(o.device?o.device:"cpu"),vocabulary(o.vocabulary_path?std::filesystem::u8path(o.vocabulary_path):std::filesystem::path()) {
  require(device.is_cpu() || device.is_cuda(),"only CPU and CUDA execution devices are supported");device=at::empty({0},at::TensorOptions().device(device)).device();
}
std::shared_ptr<const VisionEncoder> Context::vision(){std::lock_guard<std::mutex> guard(mutex);if(!vision_cache)vision_cache=std::make_shared<VisionEncoder>(store,model,device);return vision_cache;}
std::shared_ptr<const TextEncoder> Context::text(){std::lock_guard<std::mutex> guard(mutex);if(!text_cache)text_cache=std::make_shared<TextEncoder>(store,model,device);return text_cache;}
std::shared_ptr<const GroundingDetector> Context::detector(){std::lock_guard<std::mutex> guard(mutex);if(!detector_cache)detector_cache=std::make_shared<GroundingDetector>(store,model,device);return detector_cache;}
std::shared_ptr<const InteractiveImageSession> Context::interactive(){std::lock_guard<std::mutex> guard(mutex);if(!interactive_cache)interactive_cache=std::make_shared<InteractiveImageSession>(store,model,device);return interactive_cache;}
std::shared_ptr<const Sam3TrackingFrame> Context::tracking(){std::lock_guard<std::mutex> guard(mutex);if(!tracking_cache)tracking_cache=std::make_shared<Sam3TrackingFrame>(store,device);return tracking_cache;}
std::shared_ptr<const Sam31TrackingFrame> Context::multiplex(){std::lock_guard<std::mutex> guard(mutex);if(!multiplex_cache)multiplex_cache=std::make_shared<Sam31TrackingFrame>(store,device);return multiplex_cache;}
at::Tensor Context::tokenize(const char* const* texts,int64_t count){
  require(texts && count>0 && count<=INT64_MAX/32,"nonempty text batch is required");std::vector<std::string> strings;for(int64_t i=0;i<count;++i){require(texts[i],"text pointer is null");strings.emplace_back(texts[i]);}
  return tokenize(strings);
}
at::Tensor Context::tokenize(const std::vector<std::string>& strings){
  const auto count=int64_t(strings.size());require(count>0,"nonempty text batch is required");
  std::lock_guard<std::mutex> guard(mutex);require(!vocabulary.empty(),"vocabulary_path is required for tokenization");if(!tokenizer)tokenizer=std::make_unique<Tokenizer>(vocabulary);
  const auto rows=tokenizer->tokenize(strings);auto out=at::empty({count,32},at::kLong);for(int64_t i=0;i<count;++i)std::copy(rows[i].begin(),rows[i].end(),out.data_ptr<int64_t>()+i*32);return out;
}
void Context::trim(){std::lock_guard<std::mutex> guard(mutex);const auto clear=[](auto& p){if(p && p.use_count()==1)p.reset();};clear(vision_cache);clear(text_cache);clear(detector_cache);clear(interactive_cache);clear(tracking_cache);clear(multiplex_cache);tokenizer.reset();}
void result(std::map<std::string,at::Tensor> fields,sam3_result** out){auto value=std::make_unique<sam3_result>();for(auto& [key,tensor]:fields)if(tensor.defined())value->fields.emplace(key,std::move(tensor));*out=value.release();}
void video_result(const TrackingSessionOutput& value,sam3_result** out){result({{"frame",at::scalar_tensor(value.index,at::kLong)},{"ids",at::tensor(value.object_ids,at::kLong)},{"masks",value.masks},{"low_masks",value.low_masks},{"logits",value.object_logits}},out);}
bool emit(const TrackingSessionOutput& value,sam3_output_callback callback,void* user){
  if(!callback)return true;sam3_result* raw=nullptr;video_result(value,&raw);const auto result=std::unique_ptr<sam3_result,decltype(&sam3_result_release)>(raw,sam3_result_release);
  const auto status=callback(user,raw);if(status!=0 && status!=1)throw Failure(SAM3_CALLBACK_ERROR,"output callback reported failure");return status==0;
}
}
using namespace sam3::api;
extern "C" {
uint32_t sam3_abi_version(void) noexcept{return SAM3_ABI_VERSION;}
const char* sam3_last_error(void) noexcept{return sam3::api::message;}
sam3_status sam3_runtime_configure(int32_t threads,int32_t tf32) noexcept{return protect([&]{require(threads>0,"CPU thread count must be positive");at::set_num_threads(threads);at::globalContext().setAllowTF32CuBLAS(tf32!=0);at::globalContext().setAllowTF32CuDNN(tf32!=0);});}
sam3_status sam3_context_options_init(sam3_context_options* o) noexcept{return protect([&]{require(o,"options are required");*o={};o->struct_size=sizeof(*o);o->abi_version=SAM3_ABI_VERSION;o->device="cpu";o->model=SAM3_MODEL_3;o->precision=SAM3_FP32;});}
sam3_status sam3_context_create(const sam3_context_options* o,sam3_context** out) noexcept{return protect([&]{output(out);options(o);require(o->abi_version==SAM3_ABI_VERSION,"unsupported ABI version");require(o->weight_directory && *o->weight_directory,"weight directory is required");require(o->model==SAM3_MODEL_3 || o->model==SAM3_MODEL_31,"unknown model");require(o->precision>=SAM3_FP32 && o->precision<=SAM3_BF16_REFERENCE,"unknown precision");auto handle=std::make_unique<sam3_context>();handle->value=std::make_shared<Context>(*o);*out=handle.release();});}
void sam3_context_release(sam3_context* value) noexcept{delete value;}
sam3_status sam3_context_trim(sam3_context* value) noexcept{return protect([&]{require(value,"context is required");value->value->trim();});}
sam3_status sam3_result_retain(const sam3_result* value,sam3_result** out) noexcept{return protect([&]{output(out);require(value,"result is required");value->refs.fetch_add(1,std::memory_order_relaxed);*out=const_cast<sam3_result*>(value);});}
void sam3_result_release(sam3_result* value) noexcept{if(value && value->refs.fetch_sub(1,std::memory_order_acq_rel)==1)delete value;}
sam3_status sam3_result_field_count(const sam3_result* value,int64_t* out) noexcept{return protect([&]{require(value && out,"result and count output are required");*out=value->fields.size();});}
sam3_status sam3_result_field_name(const sam3_result* value,int64_t index,const char** out) noexcept{return protect([&]{require(out,"field name output is required");*out=nullptr;require(value && index>=0 && uint64_t(index)<value->fields.size(),"result field index out of range");auto it=value->fields.begin();std::advance(it,index);*out=it->first.c_str();});}
sam3_status sam3_result_get(const sam3_result* value,const char* name,sam3_tensor_view* out) noexcept{return protect([&]{require(out,"tensor output is required");*out={};require(value && name,"result and field name are required");std::lock_guard<std::mutex> guard(value->mutex);const auto it=value->fields.find(name);require(it!=value->fields.end(),"result field does not exist");auto host=value->host.find(name);if(host==value->host.end())host=value->host.emplace(name,it->second.cpu().contiguous()).first;const auto& tensor=host->second;*out={tensor.const_data_ptr(),uint64_t(tensor.nbytes()),sam3::api::dtype(tensor.scalar_type()),tensor.dim(),tensor.sizes().data()};});}
sam3_status sam3_tokenize(sam3_context* context,const char* const* texts,int64_t count,sam3_result** out) noexcept{return protect([&]{output(out);require(context,"context is required");result({{"tokens",context->value->tokenize(texts,count)}},out);});}
sam3_status sam3_encode_text(sam3_context* context,const char* const* texts,int64_t count,sam3_result** out) noexcept{return protect([&]{output(out);require(context,"context is required");auto& c=*context->value;auto tokens=c.tokenize(texts,count);const auto encoded=c.text()->forward(tokens.to(c.device),c.mode);result({{"tokens",tokens},{"padding",std::get<0>(encoded)},{"features",std::get<1>(encoded)},{"embeddings",std::get<2>(encoded)}},out);});}
sam3_status sam3_tokenize_utf8(sam3_context* context,const sam3_utf8_view* texts,int64_t count,sam3_result** out) noexcept{return protect([&]{output(out);require(context && texts && count>0 && count<=INT64_MAX/32,"context and nonempty text batch are required");std::vector<std::string> strings;for(int64_t i=0;i<count;++i){require((texts[i].data || texts[i].bytes==0) && texts[i].bytes<=SIZE_MAX,"invalid UTF-8 view");strings.emplace_back(texts[i].data?texts[i].data:"",size_t(texts[i].bytes));}result({{"tokens",context->value->tokenize(strings)}},out);});}
sam3_status sam3_encode_text_utf8(sam3_context* context,const sam3_utf8_view* texts,int64_t count,sam3_result** out) noexcept{return protect([&]{output(out);require(context && texts && count>0 && count<=INT64_MAX/32,"context and nonempty text batch are required");std::vector<std::string> strings;for(int64_t i=0;i<count;++i){require((texts[i].data || texts[i].bytes==0) && texts[i].bytes<=SIZE_MAX,"invalid UTF-8 view");strings.emplace_back(texts[i].data?texts[i].data:"",size_t(texts[i].bytes));}auto& c=*context->value;auto tokens=c.tokenize(strings);const auto encoded=c.text()->forward(tokens.to(c.device),c.mode);result({{"tokens",tokens},{"padding",std::get<0>(encoded)},{"features",std::get<1>(encoded)},{"embeddings",std::get<2>(encoded)}},out);});}
sam3_status sam3_encode_tokens(sam3_context* context,const sam3_tensor_view* input,sam3_result** out) noexcept{return protect([&]{output(out);require(context && input,"context and tokens are required");auto& c=*context->value;auto tokens=tensor(input);const auto encoded=c.text()->forward(tokens.to(c.device),c.mode);result({{"tokens",tokens},{"padding",std::get<0>(encoded)},{"features",std::get<1>(encoded)},{"embeddings",std::get<2>(encoded)}},out);});}
}
