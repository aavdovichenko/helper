#pragma once

#include <pmmintrin.h> // sse/sse2/sse3

#include "../../cpu.h"
#include "../x86.h"

#include "simd_int8_sse.h"
#include "simd_int16_sse.h"
#include "simd_int32_sse.h"
#include "simd_int64_sse.h"
#include "simd_uint8_sse.h"
#include "simd_uint16_sse.h"
#include "simd_uint32_sse.h"
#include "simd_float_sse.h"

// x86 SSE simd functions

/* TODO : implement
#if SIMD_DOUBLE_MAX_WIDTH < 2
#undef SIMD_DOUBLE_MAX_WIDTH
#define SIMD_DOUBLE_MAX_WIDTH 2
#endif
*/

namespace Platform
{

namespace Cpu
{

static inline bool isSSE3Enabled();
static inline bool isSSSE3Enabled();
static inline bool isSSE41Enabled();

template<typename FloatType> static inline constexpr int floatsPerSimdSSE();

template<typename FloatType> static inline FloatType mulZeroPaddedVectorsSSE(const FloatType* a, const FloatType* b, int size);
template<typename FloatType> static inline FloatType mulZeroPaddedVectorsSSE3(const FloatType* a, const FloatType* b, int size);

// implementation

static inline bool isSSE3Enabled()
{
  return getx86CpuFeaturesWord(2) & (1 << 0);
}

static inline bool isSSSE3Enabled()
{
  return getx86CpuFeaturesWord(2) & (1 << 9);
}

static inline bool isSSE41Enabled()
{
  return getx86CpuFeaturesWord(2) & (1 << 19);
}

template<typename FloatType> static inline constexpr int floatsPerSimdSSE()
{
  return 16 / sizeof(FloatType);
}

}

}
