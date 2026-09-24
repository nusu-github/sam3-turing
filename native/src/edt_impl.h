#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define SAM3_HD __host__ __device__
#else
#define SAM3_HD
#endif
namespace sam3 {
// Lower envelope of parabolas for one dimension of the squared distance field.
// Scratch arrays have length >= length; no length+1 sentinel is needed.
SAM3_HD inline void edt_line(const float* input, float* output, int64_t* locations,
                             double* boundaries, int64_t length, int64_t stride) {
  int64_t last = 0;
  locations[0] = 0;
  boundaries[0] = -1e30;
  for (int64_t q = 1; q < length; ++q) {
    double intersection;
    while (true) {
      const auto r = locations[last * stride];
      const double qd = static_cast<double>(q), rd = static_cast<double>(r);
      intersection = (static_cast<double>(input[q * stride]) - input[r * stride] + qd * qd - rd * rd) / (2 * (qd - rd));
      if (last == 0 || intersection > boundaries[last * stride]) break;
      --last;
    }
    ++last;
    locations[last * stride] = q;
    boundaries[last * stride] = intersection;
  }
  int64_t k = 0;
  for (int64_t q = 0; q < length; ++q) {
    while (k < last && boundaries[(k + 1) * stride] < static_cast<double>(q)) ++k;
    const auto r = locations[k * stride];
    const double delta = static_cast<double>(q) - static_cast<double>(r);
    output[q * stride] = static_cast<float>(input[r * stride] + delta * delta);
  }
}
}
#undef SAM3_HD
