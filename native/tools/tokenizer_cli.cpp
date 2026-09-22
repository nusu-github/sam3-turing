#include "sam3/tokenizer.h"
#include <iostream>
#include <stdexcept>

namespace {
int unhex(char c) {
  if (c>='0' && c<='9') return c-'0';
  if (c>='a' && c<='f') return c-'a'+10;
  if (c>='A' && c<='F') return c-'A'+10;
  throw std::invalid_argument("invalid hexadecimal text");
}
std::string decode_hex(const std::string& s) {
  if (s.size()%2) throw std::invalid_argument("odd hexadecimal length");
  std::string out;for (size_t i=0;i<s.size();i+=2) out.push_back(char(16*unhex(s[i])+unhex(s[i+1])));return out;
}
std::string hex(const std::string& s) {
  const char* digits="0123456789abcdef";std::string out;
  for (uint8_t c:s) { out+=digits[c>>4];out+=digits[c&15]; }return out;
}
}
int main(int argc,char** argv) {
  try {
    if (argc!=2 && argc!=3) throw std::invalid_argument("usage: sam3_tokenize BPE.gz [--hex-stream]; UTF-8 text lines on stdin");
    const bool binary=argc==3 && std::string(argv[2])=="--hex-stream";
    if (argc==3 && !binary) throw std::invalid_argument("unknown tokenizer option");
    sam3::Tokenizer tokenizer(std::filesystem::u8path(argv[1]));std::string line;
    while (std::getline(std::cin,line)) {
      if (!line.empty() && line.back()=='\r') line.pop_back();
      const auto text=binary?decode_hex(line):line;
      if (binary) std::cout<<hex(tokenizer.clean(text))<<'\t';
      const auto ids=tokenizer.tokenize({text})[0];
      for (size_t i=0;i<ids.size();++i) { if (i) std::cout<<',';std::cout<<ids[i]; }
      std::cout<<'\n';
    }
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
