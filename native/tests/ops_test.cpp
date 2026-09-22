#include "sam3/ops.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 2, "expected cpu|cuda");
    const auto device = at::Device(argv[1]);
    c10::InferenceMode guard;
    const auto masks = at::tensor({1,0,1,0,0,0,0,1,1, 0,1,0,0,0,0,0,0,0}).reshape({2,3,3}).to(at::kBool).to(device);
    const auto packed = sam3::pack_masks(masks);
    TORCH_CHECK(at::equal(packed.cpu(), at::tensor({133,1,2,0}).reshape({2,2}).to(at::kByte)), "packed bytes mismatch");
    TORCH_CHECK(at::equal(sam3::unpack_masks(packed,3,3).squeeze(1), masks), "round trip mismatch");
    const auto empty = at::empty({0,3,3}, masks.options());
    TORCH_CHECK(sam3::pack_masks(empty).sizes() == at::IntArrayRef({0,2}), "empty batch mismatch");
    const auto logits = at::tensor({-1.f,1.f,1.f,-1.f}).reshape({1,2,2}).to(device);
    const auto expected = at::upsample_bilinear2d(logits.unsqueeze(1), {3,7}, false).sigmoid().gt(0.5);
    TORCH_CHECK(at::equal(sam3::unpack_masks(sam3::resize_and_pack_masks(logits,3,7),3,7),expected), "resize mismatch");
    const auto ious = at::tensor({1.f,0.8f,0.5f, 0.8f,1.f,0.9f, 0.5f,0.9f,1.f}).reshape({3,3}).to(device);
    const auto scores = at::tensor({0.9f,0.9f,0.7f}).to(device);
    TORCH_CHECK(at::equal(sam3::generic_nms(ious,scores,0.5).cpu(),at::tensor({0,2},at::kLong)), "NMS tie or threshold mismatch");
    std::cout << "PASS: native masks and stable NMS on " << argv[1] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
