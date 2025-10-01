#pragma once

#include "simd_int_avx.h"

#define PLATFORM_CPU_FEATURE_UINT64x4

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<uint64_t> : public BaseAvxSimdIntType<uint64_t, AvxSimdIntType<uint64_t>>
{
  using BaseAvxSimdIntType<uint64_t, AvxSimdIntType<uint64_t>>::BaseAvxSimdIntType;

  AvxSimdIntType<uint64_t> operator+(const AvxSimdIntType<uint64_t>& other) const;

  AvxSimdIntType<uint64_t> operator>>(int count) const;
};

template<>
struct SIMD<uint64_t, 4> : public AvxIntSimd<uint64_t>
{
  static inline Type populate(uint64_t value);
};

// implementation

inline AvxSimdIntType<uint64_t> AvxSimdIntType<uint64_t>::operator+(const AvxSimdIntType<uint64_t>& other) const
{
  return AvxSimdIntType<uint64_t>::fromNativeType(_mm256_add_epi64(value, other.value));
}

inline AvxSimdIntType<uint64_t> AvxSimdIntType<uint64_t>::operator>>(int count) const
{
  return AvxSimdIntType<uint64_t>::fromNativeType(_mm256_srli_epi64(value, count));
}

inline SIMD<uint64_t, 4>::Type SIMD<uint64_t, 4>::populate(uint64_t value)
{
  return Type{_mm256_set1_epi64x(value)};
}

}

}
