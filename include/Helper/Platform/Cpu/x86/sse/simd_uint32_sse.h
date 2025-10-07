#pragma once

#include <immintrin.h> // for _mm_blend_epi32()

#include "simd_int_sse.h"

#define PLATFORM_CPU_FEATURE_INT32x4

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<uint32_t> : public BaseSseSimdIntType<uint32_t, SseSimdIntType<uint32_t>>
{
  using BaseSseSimdIntType<uint32_t, SseSimdIntType<uint32_t>>::BaseSseSimdIntType;

  SseSimdIntType<uint32_t> operator+(const SseSimdIntType<uint32_t>& other) const;

  SseSimdIntType<uint32_t> operator>>(int count) const;
};

template<>
struct SIMD<uint32_t, 4> : public SseIntSimd<uint32_t>
{
  static inline Type populate(uint32_t value);

  static inline Type mulExtended(Type a, Type b, Type& abhi);
};

// implementation

inline SseSimdIntType<uint32_t> SseSimdIntType<uint32_t>::operator+(const SseSimdIntType<uint32_t>& other) const
{
  return SseSimdIntType<uint32_t>::fromNativeType(_mm_add_epi32(value, other.value));
}

inline SseSimdIntType<uint32_t> SseSimdIntType<uint32_t>::operator>>(int count) const
{
  return SseSimdIntType<uint32_t>::SseSimdIntType(_mm_srli_epi32(value, count));
}

inline SIMD<uint32_t, 4>::Type SIMD<uint32_t, 4>::populate(uint32_t value)
{
  return Type{_mm_set1_epi32(value)};
}

inline SIMD<uint32_t, 4>::Type SIMD<uint32_t, 4>::mulExtended(Type a, Type b, Type& abhi)
{
  __m128 ab01 = _mm_castsi128_ps(_mm_mul_epu32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 1, 0)), _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 1, 0))));
  __m128 ab23 = _mm_castsi128_ps(_mm_mul_epu32(_mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 2)), _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 2))));
  abhi = Type{_mm_castps_si128(_mm_shuffle_ps(ab01, ab23, _MM_SHUFFLE(3, 1, 3, 1)))};
  return Type{_mm_castps_si128(_mm_shuffle_ps(ab01, ab23, _MM_SHUFFLE(2, 0, 2, 0)))};
}

}

}
