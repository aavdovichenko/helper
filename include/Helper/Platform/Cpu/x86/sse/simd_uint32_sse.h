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
};

template<>
struct SIMD<uint32_t, 4> : public SseIntSimd<uint32_t>
{
  static inline Type populate(uint32_t value);

  static inline Type mulExtended(Type a, Type b, Type& abhi);
};

inline SIMD<uint32_t, 4>::Type SIMD<uint32_t, 4>::populate(uint32_t value)
{
  return Type{_mm_set1_epi32(value)};
}

inline SIMD<uint32_t, 4>::Type SIMD<uint32_t, 4>::mulExtended(Type a, Type b, Type& abhi)
{
  __m128i ab02 = _mm_mul_epu32(a, b);
  __m128i ab13 = _mm_mul_epu32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1)), _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1)));
  abhi = Type{_mm_blend_epi32(_mm_shuffle_epi32(ab02, _MM_SHUFFLE(2, 3, 0, 1)), ab13, 0xaa)};
  return Type{_mm_blend_epi32(ab02, _mm_shuffle_epi32(ab13, _MM_SHUFFLE(2, 3, 0, 1)), 0xaa)};
}

namespace uint32
{

static inline SIMD<uint32_t, 4>::Type operator+(SIMD<uint32_t, 4>::Type a, SIMD<uint32_t, 4>::Type b)
{
  return SIMD<uint32_t, 4>::Type{_mm_add_epi32(a, b)};
}

static inline SIMD<uint32_t, 4>::Type operator>>(SIMD<uint32_t, 4>::Type a, int shift)
{
  return SIMD<uint32_t, 4>::Type{_mm_srli_epi32(a, shift)};
}

#if 1
static inline SIMD<uint32_t, 4>::Type operator&(SIMD<uint32_t, 4>::Type a, SIMD<uint32_t, 4>::Type b)
{
  return SIMD<uint32_t, 4>::Type{_mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)))};
}
#endif

}

}

}
