#pragma once


#ifdef PLATFORM_CPU_FEATURE_NO_SSE41
#  include "../emulated/rounding_emulated.h"
#else
#  include <smmintrin.h>

namespace Platform
{

namespace Cpu
{

static inline int ifloorf(float x)
{
  __m128 xmm = _mm_set_ss(x);
  return _mm_cvtss_si32(_mm_floor_ss(xmm, xmm));
}

static inline int ifloord(double x)
{
  __m128d xmm = _mm_set_sd(x);
  return _mm_cvtsd_si32(_mm_floor_sd(xmm, xmm));
}

static inline int iceilf(float x)
{
  __m128 xmm = _mm_set_ss(x);
  return _mm_cvtss_si32(_mm_ceil_ss(xmm, xmm));
}

static inline int iceild(double x)
{
  __m128d xmm = _mm_set_sd(x);
  return _mm_cvtsd_si32(_mm_ceil_sd(xmm, xmm));
}

}

}
#endif
