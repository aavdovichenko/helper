#pragma once

#include "simd_int_sse.h"

#define PLATFORM_CPU_FEATURE_UINT64x2

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<uint64_t> : public BaseSseSimdIntType<uint64_t, SseSimdIntType<uint64_t>>
{
  using BaseSseSimdIntType<uint64_t, SseSimdIntType<uint64_t>>::BaseSseSimdIntType;

  SseSimdIntType<uint64_t> operator+(const SseSimdIntType<uint64_t>& other) const;

  SseSimdIntType<uint64_t> operator>>(int count) const;
  SseSimdIntType<uint64_t> operator<<(int count) const;
};

template<>
struct SIMD<uint64_t, 2> : public SseIntSimd<uint64_t>
{
  static inline Type populate(uint64_t value);
};

// implementation

inline SseSimdIntType<uint64_t> SseSimdIntType<uint64_t>::operator+(const SseSimdIntType<uint64_t>& other) const
{
  return SseSimdIntType<uint64_t>::fromNativeType(_mm_add_epi64(value, other.value));
}

inline SseSimdIntType<uint64_t> SseSimdIntType<uint64_t>::operator>>(int count) const
{
  return SseSimdIntType<uint64_t>::fromNativeType(_mm_srli_epi64(value, count));
}

inline SseSimdIntType<uint64_t> SseSimdIntType<uint64_t>::operator<<(int count) const
{
  return SseSimdIntType<uint64_t>::fromNativeType(_mm_slli_epi64(value, count));
}

inline SIMD<uint64_t, 2>::Type SIMD<uint64_t, 2>::populate(uint64_t value)
{
  return Type{_mm_set1_epi64x(value)};
}

}

}
