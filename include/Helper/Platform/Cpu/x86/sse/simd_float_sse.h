#pragma once

#include <pmmintrin.h> // sse/sse2/sse3

#include "../simd_x86.h"

#define SIMD_FLOAT4_SUPPORTED // TODO: remove
#define PLATFORM_CPU_FEATURE_FLOATx4

#if SIMD_FLOAT_MAX_WIDTH < 4
#undef SIMD_FLOAT_MAX_WIDTH
#define SIMD_FLOAT_MAX_WIDTH 4
#endif

namespace Platform
{

namespace Cpu
{

static inline float reduce_mm128SSE(__m128 x128);
static inline float reduce_mm128SSE3(__m128 x128);
static inline double reduce_mm128dSSE(__m128d x128);

template<>
struct SIMD<float, 4> : public x86Simd
{
  typedef __m128 Type;
  typedef __m128i ConditionType;

  static constexpr bool isSupported()
  {
    return true;
  }

  static inline Type zero()
  {
    return _mm_setzero_ps();
  }

  static inline Type populate(float value)
  {
    return _mm_set1_ps(value);
  }

  static inline Type create(float v0, float v1, float v2, float v3)
  {
    return _mm_set_ps(v3, v2, v1, v0);
  }

  static inline Type rotate(Type value)
  {
    return _mm_shuffle_ps(value, value, _MM_SHUFFLE(0,3,2,1));
  }

  static inline float least(Type value)
  {
    return _mm_cvtss_f32(value);
  }

  static inline Type min(Type a, Type b)
  {
    return _mm_min_ps(a, b);
  }

  static inline Type max(Type a, Type b)
  {
    return _mm_max_ps(a, b);
  }

  static inline Type select(Type a, Type b, ConditionType condition)
  {
    return _mm_add_ps(_mm_and_ps(_mm_castsi128_ps(condition), a), _mm_andnot_ps(_mm_castsi128_ps(condition), b));
  }

  static inline Type sqrt(Type value)
  {
    return _mm_sqrt_ps(value);
  }

  static inline __m128i ifloor(Type value)
  {
#ifdef SIMD_SSE41_RUNTIME_SUPPORTED
    return _mm_cvtps_epi32(_mm_floor_ps(value));
#else
    __m128i ivalue = _mm_cvtps_epi32(value);

    return _mm_sub_epi32(ivalue, _mm_and_si128(_mm_set1_epi32(1), _mm_castps_si128(_mm_cmplt_ps(value, _mm_cvtepi32_ps(ivalue)))));
#endif
  }

  static inline __m128i iceil(Type value)
  {
#ifdef SIMD_SSE41_RUNTIME_SUPPORTED
    return _mm_cvtps_epi32(_mm_ceil_ps(value));
#else
    __m128i ivalue = _mm_cvtps_epi32(value);

    return _mm_add_epi32(ivalue, _mm_and_si128(_mm_set1_epi32(1), _mm_castps_si128(_mm_cmpgt_ps(value, _mm_cvtepi32_ps(ivalue)))));
#endif
  }

  static inline Type load(const float* src)
  {
    return _mm_load_ps(src);
  }

  static inline Type loadUnaligned(const float* src)
  {
    return _mm_loadu_ps(src);
  }

  static inline void store(float* dst, Type value)
  {
    _mm_store_ps(dst, value);
  }

  static inline void storeUnaligned(float* dst, Type value)
  {
    _mm_storeu_ps(dst, value);
  }

  static inline float reductionSum(Type value)
  {
    return reduce_mm128SSE(value);
  }
};

static inline __m128 mul_add(__m128 t1, __m128 m1, __m128 m2)
{
  return _mm_add_ps(t1, _mm_mul_ps(m1, m2));
}

static inline __m128 mul_sub(__m128 me, __m128 m1, __m128 m2)
{
  return _mm_sub_ps(me, _mm_mul_ps(m1, m2));
}

static inline float reduce_mm128SSE(__m128 x128)
{
#if 0
  alignas(16) float x[4];
  _mm_store_ps(x, x128);

  return (x[0] + x[1]) + (x[2] + x[3]);
#else
  x128 = _mm_add_ps(x128, _mm_shuffle_ps(x128, x128, _MM_SHUFFLE(3, 3, 1, 1))); // 0 + 1, ..., 2 + 3
  x128 = _mm_add_ss(x128, _mm_shuffle_ps(x128, x128, _MM_SHUFFLE(3, 2, 1, 2))); // (0 + 1) + (2 + 3)
  return _mm_cvtss_f32(x128);
#endif  
}

static inline float reduce_mm128SSE3(__m128 x128)
{
  x128 = _mm_hadd_ps(x128, x128);
  x128 = _mm_hadd_ps(x128, x128);

  return _mm_cvtss_f32(x128);
}

static inline double reduce_mm128dSSE(__m128d x128)
{
  alignas(16) double x[2];
  _mm_store_pd(x, x128);

  return x[0] + x[1];
}

}

}

#ifdef PLATFORM_COMPILER_MSVC
static inline __m128 operator-(__m128 a)
{
  return _mm_sub_ps(_mm_setzero_ps(), a);
}

static inline __m128 operator+(__m128 a, __m128 b)
{
  return _mm_add_ps(a, b);
}

static inline __m128 operator-(__m128 a, __m128 b)
{
  return _mm_sub_ps(a, b);
}

static inline __m128 operator*(__m128 a, __m128 b)
{
  return _mm_mul_ps(a, b);
}

static inline __m128 operator/(__m128 a, __m128 b)
{
  return _mm_div_ps(a, b);
}

static inline __m128 operator+=(__m128& a, __m128 b)
{
  return a = _mm_add_ps(a, b);
}

static inline __m128i operator<(__m128 a, __m128 b)
{
  return _mm_castps_si128(_mm_cmplt_ps(a, b));
}

static inline __m128i operator<=(__m128 a, __m128 b)
{
  return _mm_castps_si128(_mm_cmple_ps(a, b));
}

static inline __m128i operator>(__m128 a, __m128 b)
{
  return _mm_castps_si128(_mm_cmpgt_ps(a, b));
}

static inline __m128i operator>=(__m128 a, __m128 b)
{
  return _mm_castps_si128(_mm_cmpge_ps(a, b));
}
#endif
