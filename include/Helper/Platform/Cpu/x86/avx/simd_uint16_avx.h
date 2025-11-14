#pragma once

#include "simd_int_avx.h"

#define PLATFORM_CPU_FEATURE_UINT16x16

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<uint16_t> : public BaseAvxSimdIntType<uint16_t, AvxSimdIntType<uint16_t>>
{
  using BaseAvxSimdIntType<uint16_t, AvxSimdIntType<uint16_t>>::BaseAvxSimdIntType;

  AvxSimdIntType<uint16_t> operator+(const AvxSimdIntType<uint16_t>& other) const;
  AvxSimdIntType<uint16_t> operator-(const AvxSimdIntType<uint16_t>& other) const;
};

template<>
struct SIMD<uint16_t, 16> : public AvxIntSimd<uint16_t>
{
  static inline Type populate(uint16_t value);
};

// implementation

// AvxSimdIntType<uint16_t>

inline AvxSimdIntType<uint16_t> AvxSimdIntType<uint16_t>::operator+(const AvxSimdIntType<uint16_t>& other) const
{
  return AvxSimdIntType<uint16_t>{_mm256_add_epi16(value, other.value)};
}

inline AvxSimdIntType<uint16_t> AvxSimdIntType<uint16_t>::operator-(const AvxSimdIntType<uint16_t>& other) const
{
  return AvxSimdIntType<uint16_t>{_mm256_sub_epi16(value, other.value)};
}

// SIMD<uint16_t, 16>

inline AvxSimdIntType<uint16_t> SIMD<uint16_t, 16>::populate(uint16_t value)
{
  return AvxSimdIntType<uint16_t>{_mm256_set1_epi16(value)};
}

}

}
