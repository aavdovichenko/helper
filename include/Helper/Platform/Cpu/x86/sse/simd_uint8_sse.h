#pragma once

#include "simd_int_sse.h"

#define PLATFORM_CPU_FEATURE_UINT8x16

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<uint8_t> : public BaseSseSimdIntType<uint8_t, SseSimdIntType<uint8_t>>
{
  using BaseSseSimdIntType<uint8_t, SseSimdIntType<uint8_t>>::BaseSseSimdIntType;

  static SseSimdIntType<uint8_t> populate(uint8_t value);

  SseSimdIntType<uint8_t> operator>>(int count) const;
  SseSimdIntType<uint8_t> operator<<(int count) const;
};

template<>
struct SIMD<uint8_t, 16> : public SseIntSimd<uint8_t>
{
  static SseSimdIntType<uint8_t> populate(uint8_t value);
};

// SseSimdIntType<uint8_t>

inline SseSimdIntType<uint8_t> SseSimdIntType<uint8_t>::populate(uint8_t value)
{
  return _mm_set1_epi8(value);
}

inline SseSimdIntType<uint8_t> SseSimdIntType<uint8_t>::operator>>(int count) const
{
  return _mm_and_si128(_mm_srli_epi16(value, count), _mm_set1_epi8((uint8_t)0xff >> count));
}

inline SseSimdIntType<uint8_t> SseSimdIntType<uint8_t>::operator<<(int count) const
{
  return _mm_and_si128(_mm_slli_epi16(value, count), _mm_set1_epi8((uint8_t)0xff << count));
}

// SIMD<uint8_t, 16>

inline SseSimdIntType<uint8_t> SIMD<uint8_t, 16>::populate(uint8_t value)
{
  return _mm_set1_epi8(value);
}

}

}
