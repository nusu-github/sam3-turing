#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
namespace sam3 {
// Small frame-keyed dictionaries whose iteration must retain annotation order.
// Updating a value keeps its place; erase followed by insertion appends it.
template<class T> class OrderedFrames {
  std::vector<std::pair<int64_t,T>> entries_;
 public:
  auto begin(){return entries_.begin();} auto end(){return entries_.end();}
  auto begin()const{return entries_.begin();} auto end()const{return entries_.end();}
  auto find(int64_t key){return std::find_if(begin(),end(),[&](const auto& e){return e.first==key;});}
  auto find(int64_t key)const{return std::find_if(begin(),end(),[&](const auto& e){return e.first==key;});}
  size_t count(int64_t key)const{return find(key)!=end();}
  size_t size()const{return entries_.size();}
  T& operator[](int64_t key){auto it=find(key);if(it!=end())return it->second;entries_.emplace_back(key,T{});return entries_.back().second;}
  T& at(int64_t key){auto it=find(key);if(it==end())throw std::out_of_range("unknown frame");return it->second;}
  const T& at(int64_t key)const{auto it=find(key);if(it==end())throw std::out_of_range("unknown frame");return it->second;}
  void erase(int64_t key){auto it=find(key);if(it!=end())entries_.erase(it);}
  void clear(){entries_.clear();}
};
}
