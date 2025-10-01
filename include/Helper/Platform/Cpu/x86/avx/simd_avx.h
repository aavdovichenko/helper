#pragma once

#include <immintrin.h>

#include "../x86.h"

// x86 AVX simd functions

#if defined(PLATFORM_COMPILER_MSVC) || defined(__AVX2__)
#define PLATFORM_CPU_FEATURE_AVX2
#endif

#if defined(PLATFORM_CPU_FEATURE_AVX2)
#include "simd_int8_avx.h"
#include "simd_int16_avx.h"
#include "simd_int32_avx.h"
#include "simd_int64_avx.h"
#include "simd_int128_avx.h"
#include "simd_uint32_avx.h"
#include "simd_uint64_avx.h"
#endif
#include "simd_float_avx.h"

namespace Platform
{

namespace Cpu
{

static inline bool isAVXEnabled();
#if defined(PLATFORM_CPU_FEATURE_AVX2)
static inline bool isAVX2Enabled();
#else
static inline constexpr bool isAVX2Enabled();
#endif

// implementation

static inline bool isAVXEnabled()
{
  return getx86CpuFeaturesWord(2) & (1 << 28);
}

#if defined(PLATFORM_CPU_FEATURE_AVX2)
static inline bool isAVX2Enabled()
{
  return getx86CpuExtendedFeaturesWord(1) & (1 << 5);
}
#else
static inline constexpr bool isAVX2Enabled()
{
  return false;
}
#endif

}

}
