#include "sam3/tokenizer.h"
#include <iostream>
#include <stdexcept>
namespace {
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected vocabulary path");
    sam3::Tokenizer tokenizer(std::filesystem::u8path(argv[1]));
    require(tokenizer.encode("truck")==std::vector<int64_t>{4629},"unframed BPE output");
    require(tokenizer.tokenize({}).empty(),"empty prompt batch");
    require(tokenizer.tokenize({"truck"},1)[0]==std::vector<int64_t>{49407},"one-token end marker");
    require(tokenizer.tokenize({"truck"},2)[0]==std::vector<int64_t>({49406,49407}),"truncated end marker");
    const auto batch=tokenizer.tokenize({"truck",""},7);
    require(batch[0]==std::vector<int64_t>({49406,4629,49407,0,0,0,0}),"batched token padding");
    require(batch[1]==std::vector<int64_t>({49406,49407,0,0,0,0,0}),"empty text framing");
    require(tokenizer.tokenize({"truck"},77)[0].size()==77,"variable context");
    bool rejected=false;try { tokenizer.tokenize({"truck"},0); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid context accepted");
    rejected=false;try { tokenizer.encode(std::string("\xff")); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid UTF-8 accepted");
    sam3::Tokenizer moved(std::move(tokenizer));
    require(moved.clean("&AMP; CAF\xc3\x89")=="& caf\xc3\xa9","reusable moved tokenizer");
    std::cout<<"native tokenizer API checks passed\n";return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
