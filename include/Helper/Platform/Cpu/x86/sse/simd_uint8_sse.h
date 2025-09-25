#pragma once

#include "simd_int_sse.h"

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<uint8_t> : public BaseSseSimdIntType<uint8_t, SseSimdIntType<uint8_t>>
{
  using BaseSseSimdIntType<uint8_t, SseSimdIntType<uint8_t>>::BaseSseSimdIntType;

  static SseSimdIntType<uint8_t> populate(uint8_t value);
};

template<>
struct SIMD<uint8_t, 16> : public SseIntSimd<uint8_t>
{
};

inline SseSimdIntType<uint8_t> SseSimdIntType<uint8_t>::populate(uint8_t value)
{
  return _mm_set1_epi8(value);
}

}

}
