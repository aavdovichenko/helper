#pragma once

#include "simd_int_avx.h"

#define PLATFORM_CPU_FEATURE_INT128x2

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<int128_t> : public BaseAvxSimdIntType<int128_t, AvxSimdIntType<int128_t>>
{
  using BaseAvxSimdIntType<int128_t, AvxSimdIntType<int128_t>>::BaseAvxSimdIntType;

  static AvxSimdIntType<int128_t> create(int128_t lo, int128_t hi);

  template<int i0, int i1> inline AvxSimdIntType<int128_t> shuffled() const;
  template<int i0, int i1> static inline AvxSimdIntType<int128_t> shuffle(__m256i a);
  template<int i0, int i1> static inline AvxSimdIntType<int128_t> shuffle(__m256i a, __m256i b);
};

template<>
struct SIMD<int128_t, 2> : public AvxIntSimd<int128_t>
{
};

// implementation

template<int i0, int i1>
inline AvxSimdIntType<int128_t> AvxSimdIntType<int128_t>::shuffled() const
{
  return shuffle<i0, i1>(*this);
}

template<int i0, int i1>
inline AvxSimdIntType<int128_t> AvxSimdIntType<int128_t>::shuffle(__m256i a)
{
  static_assert(i0 >= 0 && i0 < 2, "invalid i0 value");
  static_assert(i1 >= 0 && i1 < 2, "invalid i1 value");
  return AvxSimdIntType<int128_t>{_mm256_permute2x128_si256(a, a, (i1 << 4) | i0)};
}

template<int i0, int i1>
inline AvxSimdIntType<int128_t> AvxSimdIntType<int128_t>::shuffle(__m256i a, __m256i b)
{
  static_assert(i0 >= 0 && i0 < 4, "invalid i0 value");
  static_assert(i1 >= 0 && i1 < 4, "invalid i1 value");
  return AvxSimdIntType<int128_t>{_mm256_permute2x128_si256(a, b, (i1 << 4) | i0)};
}

inline AvxSimdIntType<int128_t> AvxSimdIntType<int128_t>::create(int128_t lo, int128_t hi)
{
  return AvxSimdIntType<int128_t>{_mm256_setr_m128i(lo, hi)};
}

}

}
