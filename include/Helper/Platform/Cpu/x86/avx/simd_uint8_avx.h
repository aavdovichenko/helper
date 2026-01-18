#pragma once

#include "simd_int_avx.h"

#define PLATFORM_CPU_FEATURE_UINT8x32

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<uint8_t> : public BaseAvxSimdIntType<uint8_t, AvxSimdIntType<uint8_t>>
{
  using BaseAvxSimdIntType<uint8_t, AvxSimdIntType<uint8_t>>::BaseAvxSimdIntType;

  AvxSimdIntType<uint8_t> operator>>(int count) const;
  AvxSimdIntType<uint8_t> operator<<(int count) const;
};

template<>
struct SIMD<uint8_t, 32> : public AvxIntSimd<uint8_t>
{
  static inline Type populate(uint8_t value);
};

// implementation

// AvxSimdIntType<uint8_t>

inline AvxSimdIntType<uint8_t> AvxSimdIntType<uint8_t>::operator>>(int count) const
{
  return AvxSimdIntType<uint8_t>::fromNativeType(_mm256_and_si256(_mm256_srli_epi16(value, count), _mm256_set1_epi8((uint8_t)0xff >> count)));
}

inline AvxSimdIntType<uint8_t> AvxSimdIntType<uint8_t>::operator<<(int count) const
{
  return AvxSimdIntType<uint8_t>::fromNativeType(_mm256_and_si256(_mm256_slli_epi16(value, count), _mm256_set1_epi8((uint8_t)0xff << count)));
}

// SIMD<uint8_t, 32>

inline AvxSimdIntType<uint8_t> SIMD<uint8_t, 32>::populate(uint8_t value)
{
  return AvxSimdIntType<uint8_t>{_mm256_set1_epi8(value)};
}

}

}
