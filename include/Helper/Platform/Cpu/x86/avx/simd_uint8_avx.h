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
};

template<>
struct SIMD<uint8_t, 32> : public AvxIntSimd<uint8_t>
{
  static inline Type populate(uint8_t value);
};

// implementation

inline AvxSimdIntType<uint8_t> SIMD<uint8_t, 32>::populate(uint8_t value)
{
  return AvxSimdIntType<uint8_t>{_mm256_set1_epi8(value)};
}

}

}
