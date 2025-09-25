#pragma once

#include "simd_int_avx.h"

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<uint32_t> : public BaseAvxSimdIntType<uint32_t, AvxSimdIntType<uint32_t>>
{
  using BaseAvxSimdIntType<uint32_t, AvxSimdIntType<uint32_t>>::BaseAvxSimdIntType;

  AvxSimdIntType<uint32_t> operator+(const AvxSimdIntType<uint32_t>& other) const;

  AvxSimdIntType<uint32_t> operator>>(int count) const;
};

template<>
struct SIMD<uint32_t, 8> : public AvxIntSimd<uint32_t>
{
  static inline Type populate(uint32_t value);

  static inline Type mulExtended(Type a, Type b, Type& abhi);
};

// implementation

inline AvxSimdIntType<uint32_t> AvxSimdIntType<uint32_t>::operator+(const AvxSimdIntType<uint32_t>& other) const
{
  return AvxSimdIntType<uint32_t>::fromNativeType(_mm256_add_epi32(value, other.value));
}

inline AvxSimdIntType<uint32_t> AvxSimdIntType<uint32_t>::operator>>(int count) const
{
  return AvxSimdIntType<uint32_t>::fromNativeType(_mm256_srli_epi32(value, count));
}

inline SIMD<uint32_t, 8>::Type SIMD<uint32_t, 8>::populate(uint32_t value)
{
  return Type{_mm256_set1_epi32(value)};
}

inline SIMD<uint32_t, 8>::Type SIMD<uint32_t, 8>::mulExtended(Type a, Type b, Type& abhi)
{
  __m256i ab0246 = _mm256_mul_epu32(a, b);
  __m256i ab1357 = _mm256_mul_epu32(_mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1)), _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1)));
  abhi = Type{_mm256_blend_epi32(_mm256_shuffle_epi32(ab0246, _MM_SHUFFLE(2, 3, 0, 1)), ab1357, 0xaa)};
  return Type{_mm256_blend_epi32(ab0246, _mm256_shuffle_epi32(ab1357, _MM_SHUFFLE(2, 3, 0, 1)), 0xaa)};
}

}

}
