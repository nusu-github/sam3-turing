#pragma once
#include <ATen/ATen.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cctype>
namespace sam3::cli {
inline std::string ppm_token(std::istream& in) {
  std::string token;
  while (in) {
    in>>std::ws;
    if (in.peek()!='#') { in>>token;return token; }
    in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
  }
  TORCH_CHECK(false,"truncated PPM header");
}
inline at::Tensor read_ppm(const std::filesystem::path& path) {
  std::ifstream in(path,std::ios::binary);
  TORCH_CHECK(in,"cannot open input image");
  TORCH_CHECK(ppm_token(in)=="P6","this development probe accepts binary P6 RGB PPM");
  const auto w=std::stoll(ppm_token(in)),h=std::stoll(ppm_token(in));
  TORCH_CHECK(ppm_token(in)=="255","PPM must have 8-bit RGB samples");
  const int delimiter=in.get();
  TORCH_CHECK(delimiter!=EOF && std::isspace(static_cast<unsigned char>(delimiter)),"invalid PPM separator");
  if (delimiter=='\r' && in.peek()=='\n') in.get();
  TORCH_CHECK(w>0 && h>0 && w<=std::numeric_limits<int64_t>::max()/3/h,"invalid PPM dimensions");
  auto pixels=at::empty({h,w,3},at::TensorOptions().dtype(at::kByte));
  in.read(reinterpret_cast<char*>(pixels.mutable_data_ptr<uint8_t>()),pixels.numel());
  TORCH_CHECK(in.gcount()==pixels.numel(),"truncated PPM pixels");
  return pixels.permute({2,0,1});
}
}
