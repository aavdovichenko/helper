#pragma once

#include <immintrin.h>

#include "../../cpu.h"
#include "../x86.h"

// x86 AVX simd functions

#if defined(PLATFORM_CPU_FEATURE_AVX2)
#include "simd_int8_avx.h"
#include "simd_int16_avx.h"
#include "simd_int32_avx.h"
#include "simd_int64_avx.h"
#include "simd_int128_avx.h"
#include "simd_uint32_avx.h"
#include "simd_uint64_avx.h"
#endif
#if defined(PLATFORM_CPU_FEATURE_AVX)
#include "simd_float_avx.h"
#endif

namespace Platform
{

namespace Cpu
{

static inline bool isAVXEnabled();
static inline bool isAVX2Enabled();

// implementation

static inline bool isAVXEnabled()
{
  return getx86CpuFeaturesWord(2) & (1 << 28);
}

static inline bool isAVX2Enabled()
{
  return getx86CpuExtendedFeaturesWord(1) & (1 << 5);
}

}

}
