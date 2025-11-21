#pragma once

#include "simd_int_sse.h"

#define PLATFORM_CPU_FEATURE_UINT16x8

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<uint16_t> : public BaseSseSimdIntType<uint16_t, SseSimdIntType<uint16_t>>
{
  using BaseSseSimdIntType<uint16_t, SseSimdIntType<uint16_t>>::BaseSseSimdIntType;

  static SseSimdIntType<uint16_t> populate(uint16_t value);

  SseSimdIntType<uint16_t> operator+(const SseSimdIntType<uint16_t>& other) const;
  SseSimdIntType<uint16_t> operator-(const SseSimdIntType<uint16_t>& other) const;
};

template<>
struct SIMD<uint16_t, 8> : public SseIntSimd<uint16_t>
{
  static inline Type populate(uint16_t value);

  template<bool dstAligned, bool srcAligned>
  static inline void transpose(uint16_t* dst, size_t dstStride, const uint16_t* src, size_t srcStride);
};

// SseSimdIntType<uint16_t>

inline SseSimdIntType<uint16_t> SseSimdIntType<uint16_t>::populate(uint16_t value)
{
  return _mm_set1_epi16(value);
}

inline SseSimdIntType<uint16_t> SseSimdIntType<uint16_t>::operator+(const SseSimdIntType<uint16_t>& other) const
{
  return SseSimdIntType<uint16_t>{_mm_add_epi16(value, other.value)};
}

inline SseSimdIntType<uint16_t> SseSimdIntType<uint16_t>::operator-(const SseSimdIntType<uint16_t>& other) const
{
  return SseSimdIntType<uint16_t>{_mm_sub_epi16(value, other.value)};
}

// SIMD<uint16_t, 8>

inline SseSimdIntType<uint16_t> SIMD<uint16_t, 8>::populate(uint16_t value)
{
  return SseSimdIntType<uint16_t>{_mm_set1_epi16(value)};
}

template<bool dstAligned, bool srcAligned>
inline void SIMD<uint16_t, 8>::transpose(uint16_t* dst, size_t dstStride, const uint16_t* src, size_t srcStride)
{
  Type w0 = load<srcAligned>(src + 0 * srcStride), w1 = load<srcAligned>(src + 1 * srcStride);
  Type w2 = load<srcAligned>(src + 2 * srcStride), w3 = load<srcAligned>(src + 3 * srcStride);
  Type w4 = load<srcAligned>(src + 4 * srcStride), w5 = load<srcAligned>(src + 5 * srcStride);
  Type w6 = load<srcAligned>(src + 6 * srcStride), w7 = load<srcAligned>(src + 7 * srcStride);

  transposeSseInt16(w0.value, w1.value, w2.value, w3.value, w4.value, w5.value, w6.value, w7.value);

  w0.store<dstAligned>(dst + 0 * dstStride); w1.store<dstAligned>(dst + 1 * dstStride);
  w2.store<dstAligned>(dst + 2 * dstStride); w3.store<dstAligned>(dst + 3 * dstStride);
  w4.store<dstAligned>(dst + 4 * dstStride); w5.store<dstAligned>(dst + 5 * dstStride);
  w6.store<dstAligned>(dst + 6 * dstStride); w7.store<dstAligned>(dst + 7 * dstStride);
}

}

}
