#pragma once

#include <emmintrin.h>

namespace Platform
{

namespace Cpu
{

static inline float sqrtf(float x)
{
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
}

static inline double sqrtd(double x)
{
  __m128d x128 = _mm_set_sd(x);
  return _mm_cvtsd_f64(_mm_sqrt_sd(x128, x128));
}

static inline float rsqrtf(float x)
{
  return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
}

static inline double rsqrtd(double x)
{
  __m128d x128 = _mm_set_sd(x);
  return 1.0 / _mm_cvtsd_f64(_mm_sqrt_sd(x128, x128));
}

}

}
