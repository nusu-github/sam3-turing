// ftfy algorithm adapted under MIT; tables also contain PSF/Unicode data.
// See native/third_party/ftfy and native/third_party/python.
#include "sam3/tokenizer.h"
#include "tokenizer_tables.h"
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uniset.h>
#include <unicode/locid.h>
#include <zlib.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace sam3 {
namespace {
using U = icu::UnicodeString;
using Pattern = std::unique_ptr<icu::RegexPattern>;
void check(UErrorCode status) {
  if (U_FAILURE(status)) throw std::runtime_error(std::string("tokenizer ICU: ")+u_errorName(status));
}
U unicode(const std::string& s) { return U::fromUTF8(s); }
std::string utf8(const U& s) { std::string out; s.toUTF8String(out); return out; }
Pattern pattern(const U& s,uint32_t flags=0) {
  UErrorCode status=U_ZERO_ERROR;
  Pattern out(icu::RegexPattern::compile(s,flags,status));check(status);return out;
}
bool search(const Pattern& p,const U& text) {
  UErrorCode status=U_ZERO_ERROR;
  std::unique_ptr<icu::RegexMatcher> m(p->matcher(text,status));check(status);
  bool found=m->find(status);check(status);return found;
}
template<class Fn> U replace(const Pattern& p,const U& text,Fn fn) {
  UErrorCode status=U_ZERO_ERROR;
  std::unique_ptr<icu::RegexMatcher> m(p->matcher(text,status));check(status);
  U out;int32_t last=0;
  while (m->find(status)) {
    const auto start=m->start(status),end=m->end(status);check(status);
    out.append(text,last,start-last);out.append(fn(text.tempSubStringBetween(start,end)));
    last=end;
  }
  check(status);out.append(text,last,text.length()-last);return out;
}
// mode 0: strict UTF-8; 1: ftfy's CESU-8/Java-null variants; 2: accept
// surrogate code points at the public boundary for source fix_surrogates.
bool decode(const std::string& bytes,U& out,int mode=0) {
  out.remove();std::vector<uint32_t> chars;
  for (size_t i=0;i<bytes.size();) {
    const uint8_t b=bytes[i++];uint32_t cp=b;int n=0;
    if (b<0x80) {}
    else if (mode==1 && b==0xc0 && i<bytes.size() && uint8_t(bytes[i])==0x80) { ++i;cp=0; }
    else {
      if (b>=0xc2 && b<=0xdf) { n=1;cp=b&31; }
      else if (b>=0xe0 && b<=0xef) { n=2;cp=b&15; }
      else if (b>=0xf0 && b<=0xf4) { n=3;cp=b&7; }
      else return false;
      if (i+n>bytes.size()) return false;
      for (int j=0;j<n;++j) { uint8_t c=bytes[i++];if ((c&0xc0)!=0x80) return false;cp=(cp<<6)|(c&63); }
      if (cp>(uint32_t)0x10ffff || (n==1 && cp<0x80) || (n==2 && cp<0x800) || (n==3 && cp<0x10000)) return false;
    }
    if (mode==0 && cp>=0xd800 && cp<=0xdfff) return false;
    chars.push_back(cp);
  }
  for (size_t i=0;i<chars.size();++i) {
    auto cp=chars[i];
    if (mode==1 && cp>=0xd800 && cp<=0xdfff) {
      if (cp>0xdbff || i+1==chars.size() || chars[i+1]<0xdc00 || chars[i+1]>0xdfff) return false;
      cp=0x10000+(cp-0xd800)*1024+(chars[++i]-0xdc00);
    }
    out.append(UChar32(cp));
  }
  return true;
}
U byte_string(const std::string& bytes) { U s;for (uint8_t b:bytes) s.append(UChar32(b));return s; }
std::string raw_bytes(const U& s) { std::string b;for (int32_t i=0;i<s.length();++i) b.push_back(char(s[i]));return b; }
template<size_t N> std::unordered_map<uint32_t,U> mapping(const tokenizer_data::Mapping (&rows)[N]) {
  std::unordered_map<uint32_t,U> m;for (const auto& r:rows) m.emplace(r.code,unicode(r.value));return m;
}
template<size_t N> std::unordered_map<std::string,U> entities(const tokenizer_data::Entity (&rows)[N]) {
  std::unordered_map<std::string,U> m;for (const auto& r:rows) m.emplace(r.name,unicode(r.value));return m;
}
struct Cleaner {
  Pattern bad=pattern(unicode(tokenizer_data::badness));
  Pattern detector=pattern(unicode(tokenizer_data::utf8_detector));
  Pattern altered=pattern(unicode(tokenizer_data::altered));
  Pattern lossy=pattern(unicode(tokenizer_data::lossy));
  Pattern grave=pattern(unicode(tokenizer_data::a_grave));
  Pattern ansi=pattern(unicode(tokenizer_data::ansi));
  Pattern strict=pattern(unicode(tokenizer_data::html_strict));
  Pattern loose=pattern(unicode(tokenizer_data::html_loose));
  Pattern whitespace=pattern(unicode(std::string("[")+tokenizer_data::spaces+"]+"));
  std::unordered_map<uint32_t,U> simple=mapping(tokenizer_data::simple_fixes);
  std::unordered_map<uint32_t,U> invalid=mapping(tokenizer_data::invalid_charrefs);
  std::unordered_map<std::string,U> html=entities(tokenizer_data::html5);
  std::unordered_map<std::string,U> ftfy_html=entities(tokenizer_data::ftfy_entities);
  std::unordered_set<uint32_t> invalid_cp{std::begin(tokenizer_data::invalid_codepoints),std::end(tokenizer_data::invalid_codepoints)};
  std::unordered_set<uint32_t> controls{std::begin(tokenizer_data::controls),std::end(tokenizer_data::controls)};
  std::array<std::unordered_map<uint32_t,uint8_t>,9> encoders;
  icu::UnicodeSet strip_set;
  const icu::Normalizer2* nfc=nullptr;
  Cleaner() {
    UErrorCode status=U_ZERO_ERROR;
    strip_set.applyPattern(unicode(std::string("[")+tokenizer_data::py_spaces+"]"),status);check(status);strip_set.freeze();
    nfc=icu::Normalizer2::getNFCInstance(status);check(status);
    for (size_t c=0;c<encoders.size();++c)
      for (int b=0;b<256;++b) encoders[c][tokenizer_data::codecs[c][b]]=b;
  }
  U numeric(const std::string& entity) const {
    size_t i=1;int base=10;
    if (i<entity.size() && (entity[i]=='x' || entity[i]=='X')) { ++i;base=16; }
    uint32_t cp=0;
    for (;i<entity.size() && entity[i]!=';';++i) {
      const auto c=entity[i];const int d=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c-'A'+10;
      cp=std::min<uint32_t>(0x110000,cp*base+d);
    }
    auto found=invalid.find(cp);if (found!=invalid.end()) return found->second;
    if (cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff)) return U(UChar32(0xfffd));
    if (invalid_cp.count(cp)) return {};
    return U(UChar32(cp));
  }
  U unescape(const U& text,bool conservative) const {
    return replace(conservative?strict:loose,text,[&](const U& match) {
      const auto full=utf8(match);
      if (conservative) {
        auto found=ftfy_html.find(full);if (found!=ftfy_html.end()) return found->second;
        if (full.size()>2 && full[1]=='#') {
          auto result=unescape(match,false);
          if (result.indexOf(UChar(';'))<0) return result;
        }
        return U(match);
      }
      const auto name=full.substr(1);
      if (name[0]=='#') return numeric(name);
      for (size_t len=name.size();len>1;--len) {
        auto found=html.find(name.substr(0,len));
        if (found!=html.end()) return U(found->second).append(unicode(name.substr(len)));
      }
      return U(match);
    });
  }
  U encoding_step(const U& text) const {
    if (!search(bad,text)) return text;
    for (size_t codec=0;codec<encoders.size();++codec) {
      std::string bytes;bool possible=true;
      for (int32_t i=0;i<text.length();) {
        auto cp=text.char32At(i);i+=U16_LENGTH(cp);auto found=encoders[codec].find(cp);
        if (found==encoders[codec].end()) { possible=false;break; }
        bytes.push_back(char(found->second));
      }
      if (!possible) continue;
      U b=byte_string(bytes);
      if (search(altered,b)) {
        b=replace(grave,b,[](const U&) { return U(UChar32(0xc3)).append(UChar32(0xa0)).append(UChar32(32)); });
        b=replace(altered,b,[](U s) { for (int32_t i=0;i<s.length();++i) if (s[i]==32) s.setCharAt(i,0xa0);return s; });
      }
      if (codec>=1 && codec<=5) b=replace(lossy,b,[](const U&) { return byte_string("\xef\xbf\xbd"); });
      bytes=raw_bytes(b);U decoded;
      if (decode(bytes,decoded,1)) return decoded;
    }
    auto partial=replace(detector,text,[&](const U& s) {
      return s.length()<text.length() && search(bad,s) ? fix_encoding(s) : U(s);
    });
    if (partial!=text) return partial;
    U fixed;
    for (int32_t i=0;i<text.length();) {
      auto cp=text.char32At(i);i+=U16_LENGTH(cp);
      fixed.append(UChar32(cp>=0x80 && cp<=0x9f?tokenizer_data::codecs[1][cp]:cp));
    }
    return fixed;
  }
  U fix_encoding(U text) const {
    for (;;) { auto fixed=encoding_step(text);if (fixed==text) return text;text=std::move(fixed); }
  }
  U segment(U text,bool html_enabled) const {
    for (;;) {
      const auto original=text;
      if (html_enabled) text=unescape(text,true);
      text=fix_encoding(text);
      U fixed;
      for (int32_t i=0;i<text.length();) {
        const auto cp=text.char32At(i);i+=U16_LENGTH(cp);
        if (cp=='\r' && i<text.length() && text[i]=='\n') ++i;
        if (cp>=0xd800 && cp<=0xdfff) { fixed.append(UChar32(0xfffd));continue; }
        auto found=simple.find(cp);
        if (found==simple.end()) fixed.append(cp);else fixed.append(found->second);
      }
      fixed=replace(ansi,fixed,[](const U&) { return U(); });
      U filtered;
      for (int32_t i=0;i<fixed.length();) {
        auto cp=fixed.char32At(i);i+=U16_LENGTH(cp);if (!controls.count(cp)) filtered.append(cp);
      }
      UErrorCode status=U_ZERO_ERROR;nfc->normalize(filtered,text,status);check(status);
      if (text==original) return text;
    }
  }
  U clean(const std::string& bytes) const {
    U input;if (!decode(bytes,input,2)) throw std::invalid_argument("prompt must be UTF-8");
    U text;bool html_enabled=true;
    for (int32_t start=0;start<input.length();) {
      auto end=input.indexOf(UChar('\n'),start);end=end<0?input.length():end+1;
      end=std::min(end,input.moveIndex32(start,1000000));
      auto part=input.tempSubStringBetween(start,end);
      if (part.indexOf(UChar('<'))>=0) html_enabled=false;
      text.append(segment(part,html_enabled));start=end;
    }
    text=unescape(unescape(text,false),false);
    text=replace(whitespace,text,[](const U&) { return U(" "); });
    int32_t start=0,end=text.length();
    while (start<end && strip_set.contains(text.char32At(start))) start=text.moveIndex32(start,1);
    while (end>start && strip_set.contains(text.char32At(text.moveIndex32(end,-1)))) end=text.moveIndex32(end,-1);
    text=U(text,start,end-start);text.toLower(icu::Locale::getRoot());return text;
  }
};
std::string read_vocabulary(const std::filesystem::path& path) {
  // ifstream accepts native Unicode paths on Windows; gzopen(char*) does not.
  std::ifstream in(path,std::ios::binary);if (!in) throw std::runtime_error("cannot open BPE vocabulary");
  std::string compressed((std::istreambuf_iterator<char>(in)),{}),out;
  if (compressed.size()>std::numeric_limits<uInt>::max()) throw std::runtime_error("BPE gzip too large");
  z_stream stream{};stream.next_in=reinterpret_cast<Bytef*>(compressed.data());stream.avail_in=compressed.size();
  if (inflateInit2(&stream,16+MAX_WBITS)!=Z_OK) throw std::runtime_error("cannot initialize gzip decoder");
  int result=Z_OK;std::array<char,65536> block{};
  while (result==Z_OK) {
    stream.next_out=reinterpret_cast<Bytef*>(block.data());stream.avail_out=block.size();
    result=inflate(&stream,Z_NO_FLUSH);out.append(block.data(),block.size()-stream.avail_out);
  }
  inflateEnd(&stream);
  if (result!=Z_STREAM_END) throw std::runtime_error("invalid BPE gzip");
  return out;
}
}
struct Tokenizer::Impl {
  Cleaner cleaner;
  Pattern tokens;
  std::array<std::string,256> byte_encoder;
  std::unordered_map<std::string,int64_t> vocab,ranks;
  explicit Impl(const std::filesystem::path& path) {
    const std::string letters=tokenizer_data::letters,numbers=tokenizer_data::numbers,other=tokenizer_data::other;
    // Source regex IGNORECASE has asymmetric property/negated-property
    // behavior (notably U+0345 matches neither). Freeze each branch separately;
    // applying ICU case folding to the union changes token boundaries.
    tokens=pattern(unicode("(?i:<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d)|["+letters+"]+|["+numbers+"]|["+other+"]+"));
    std::vector<int> order;for (int i=33;i<=126;++i) order.push_back(i);
    for (int i=161;i<=172;++i) order.push_back(i);for (int i=174;i<=255;++i) order.push_back(i);
    for (auto b:order) byte_encoder[b]=utf8(U(UChar32(b)));
    int next=256;for (int b=0;b<256;++b) if (byte_encoder[b].empty()) { order.push_back(b);byte_encoder[b]=utf8(U(UChar32(next++))); }
    int64_t id=0;for (auto b:order) vocab[byte_encoder[b]]=id++;
    for (auto b:order) vocab[byte_encoder[b]+"</w>"]=id++;
    std::istringstream input(read_vocabulary(path));std::string line;std::getline(input,line);
    for (int rank=0;rank<48894;++rank) {
      if (!std::getline(input,line)) throw std::runtime_error("BPE vocabulary is missing merges");
      std::istringstream row(line);std::string a,b;row>>a>>b;
      if (a.empty() || b.empty()) throw std::runtime_error("invalid BPE merge");
      ranks[a+'\n'+b]=rank;vocab[a+b]=id++;
    }
    vocab["<start_of_text>"]=id++;vocab["<end_of_text>"]=id++;
    if (id!=49408) throw std::runtime_error("unexpected BPE vocabulary size");
  }
  std::vector<int64_t> encode(const std::string& text) const {
    auto cleaned=cleaner.clean(text);std::vector<int64_t> result;
    replace(tokens,cleaned,[&](const U& token) {
      const auto bytes=utf8(token);
      if (bytes=="<start_of_text>" || bytes=="<end_of_text>") { result.push_back(vocab.at(bytes));return U(); }
      std::vector<std::string> word;for (uint8_t b:bytes) word.push_back(byte_encoder[b]);
      word.back()+="</w>";
      while (word.size()>1) {
        int64_t best=std::numeric_limits<int64_t>::max();size_t pos=0;
        for (size_t i=0;i+1<word.size();++i) {
          auto found=ranks.find(word[i]+'\n'+word[i+1]);
          if (found!=ranks.end() && found->second<best) { best=found->second;pos=i; }
        }
        if (best==std::numeric_limits<int64_t>::max()) break;
        const auto a=word[pos],b=word[pos+1];std::vector<std::string> merged;
        for (size_t i=0;i<word.size();++i) {
          if (i+1<word.size() && word[i]==a && word[i+1]==b) { merged.push_back(a+b);++i; }
          else merged.push_back(std::move(word[i]));
        }
        word=std::move(merged);
      }
      for (const auto& piece:word) result.push_back(vocab.at(piece));return U();
    });
    return result;
  }
};
Tokenizer::Tokenizer(const std::filesystem::path& path):impl_(std::make_unique<Impl>(path)) {}
Tokenizer::~Tokenizer()=default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept=default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept=default;
std::string Tokenizer::clean(const std::string& text) const { return utf8(impl_->cleaner.clean(text)); }
std::vector<int64_t> Tokenizer::encode(const std::string& text) const { return impl_->encode(text); }
std::vector<std::vector<int64_t>> Tokenizer::tokenize(const std::vector<std::string>& texts,int64_t context) const {
  if (context<=0) throw std::invalid_argument("token context must be positive");
  std::vector<std::vector<int64_t>> result;
  for (const auto& text:texts) {
    auto ids=encode(text);ids.insert(ids.begin(),49406);ids.push_back(49407);
    if (ids.size()>static_cast<size_t>(context)) { ids.resize(context);ids.back()=49407; }
    else ids.resize(context,0);
    result.push_back(std::move(ids));
  }
  return result;
}
}
