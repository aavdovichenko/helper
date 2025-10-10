#pragma once

#include "simd_int_avx.h"

#define PLATFORM_CPU_FEATURE_INT64x4

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<int64_t> : public BaseAvxSimdIntType<int64_t, AvxSimdIntType<int64_t>>
{
  using BaseAvxSimdIntType<int64_t, AvxSimdIntType<int64_t>>::BaseAvxSimdIntType;

  static AvxSimdIntType<int64_t> populate(int64_t value);

  template<int i0, int i1, int i2, int i3>
  static inline AvxSimdIntType<int64_t> shuffle(__m256i a);
  template<int i0, int i1, int i2, int i3>
  static inline AvxSimdIntType<int64_t> shuffle(__m256i a, __m256i b);

  AvxSimdIntType<int64_t> operator<<(int count) const;

  inline AvxSimdIntType<int64_t> revertedByteOrder() const;
};

template<>
struct SIMD<int64_t, 4> : public AvxIntSimd<int64_t>
{
};

// implementation

inline AvxSimdIntType<int64_t> AvxSimdIntType<int64_t>::populate(int64_t value)
{
  return AvxSimdIntType<int64_t>::fromNativeType(_mm256_set1_epi64x(value));
}

template<int i0, int i1, int i2, int i3>
inline AvxSimdIntType<int64_t> AvxSimdIntType<int64_t>::shuffle(__m256i a)
{
  static_assert(i0 >= 0 && i0 < 4, "invalid i0 value");
  static_assert(i1 >= 0 && i1 < 4, "invalid i1 value");
  static_assert(i2 >= 0 && i2 < 4, "invalid i2 value");
  static_assert(i3 >= 0 && i3 < 4, "invalid i3 value");
  return AvxSimdIntType<int64_t>{_mm256_permute4x64_epi64(a, _MM_SHUFFLE(i3, i2, i1, i0))};
}

template<int i0, int i1, int i2, int i3>
inline AvxSimdIntType<int64_t> AvxSimdIntType<int64_t>::shuffle(__m256i a, __m256i b)
{
  static_assert(i0 >= 0 && i0 < 8, "invalid i0 value");
  static_assert(i1 >= 0 && i1 < 8, "invalid i1 value");
  static_assert(i2 >= 0 && i2 < 8, "invalid i2 value");
  static_assert(i3 >= 0 && i3 < 8, "invalid i3 value");

  static_assert((i0 == 0 && i1 == 4 && i2 == 2 && i3 == 6) || (i0 == 1 && i1 == 5 && i2 == 3 && i3 == 7), "not implemented");
  if (i0 == 0 && i1 == 4 && i2 == 2 && i3 == 6)
    return AvxSimdIntType<int64_t>{_mm256_unpacklo_epi64(a, b)};
  if (i0 == 1 && i1 == 5 && i2 == 3 && i3 == 7)
    return AvxSimdIntType<int64_t>{_mm256_unpackhi_epi64(a, b)};

  return AvxSimdIntType<int64_t>();
}

inline AvxSimdIntType<int64_t> AvxSimdIntType<int64_t>::operator<<(int count) const
{
  return AvxSimdIntType<int64_t>::fromNativeType(_mm256_slli_epi64(value, count));
}

inline AvxSimdIntType<int64_t> AvxSimdIntType<int64_t>::revertedByteOrder() const
{
  __m256i indices = _mm256_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
  return _mm256_shuffle_epi8(value, indices);
}

}

}
