#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace sam3::detail {
// Compatibility with CPython 3.12's nonnegative 64-bit integer set traversal.
// The original predictor iterates temporary sets into ordered memory history.
// No Python API is used. Probe/growth rules are described by CPython's
// Objects/setobject.c (PSF license: native/third_party/python/LICENSE.txt).
class FrameSet {
  std::vector<int64_t> table_=std::vector<int64_t>(8,-1);
  size_t used_=0;
  bool place(int64_t key) {
    const auto mask=table_.size()-1;uint64_t perturb=uint64_t(key)%((uint64_t(1)<<61)-1);
    size_t index=perturb&mask;
    for(;;) {
      const auto end=index+(index+9<=mask?9:0);
      for(auto slot=index;slot<=end;++slot) {
        if(table_[slot]==key)return false;
        if(table_[slot]<0){table_[slot]=key;return true;}
      }
      perturb>>=5;index=(index*5+1+perturb)&mask;
    }
  }
  void resize(size_t minimum) {
    size_t size=8;while(size<=minimum)size*=2;
    if(size==table_.size())return;
    auto previous=std::move(table_);table_.assign(size,-1);
    for(const auto key:previous)if(key>=0)place(key);
  }
 public:
  void update(const std::vector<int64_t>& keys,bool dictionary=false) {
    // set.update(dict) preallocates; set.update(dict.keys()) does not.
    if(dictionary && (used_+keys.size())*5>=(table_.size()-1)*3)resize((used_+keys.size())*2);
    for(const auto key:keys) {
      if(key<0)throw std::invalid_argument("negative frame index");
      if(place(key) && ++used_*5>=(table_.size()-1)*3)resize(used_*(used_>50000?2:4));
    }
  }
  std::vector<int64_t> order()const {std::vector<int64_t> out;for(const auto key:table_)if(key>=0)out.push_back(key);return out;}
};
inline std::vector<int64_t> frame_set_order(const std::vector<std::vector<int64_t>>& groups,bool dictionary) {
  FrameSet frames;for(const auto& group:groups)frames.update(group,dictionary);return frames.order();
}
}
