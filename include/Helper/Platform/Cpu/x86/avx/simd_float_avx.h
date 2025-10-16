#pragma once

#include <immintrin.h>

#include "../simd_x86.h"

#define SIMD_FLOAT8_SUPPORTED // TODO : remove
#define PLATFORM_CPU_FEATURE_FLOATx8

#if SIMD_FLOAT_MAX_WIDTH < 8
#undef SIMD_FLOAT_MAX_WIDTH
#define SIMD_FLOAT_MAX_WIDTH 8
#endif
/* TODO : implement
#if SIMD_DOUBLE_MAX_WIDTH < 4
#undef SIMD_DOUBLE_MAX_WIDTH
#define SIMD_DOUBLE_MAX_WIDTH 4
#endif
*/

namespace Platform
{

namespace Cpu
{

static inline bool isAVXEnabled();

static inline float reduce_mm256(__m256 x256);
static inline double reduce_mm256d(__m256d x256);

template<>
struct SIMD<float, 8> : public x86Simd<32>
{
  typedef __m256 Type;
  typedef __m256i ConditionType;

  static bool isSupported()
  {
    static bool avxEnabled = isAVXEnabled();
    return avxEnabled;
  }

  static inline Type zero()
  {
    return _mm256_setzero_ps();
  }

  static inline Type populate(float value)
  {
    return _mm256_set1_ps(value);
  }

  static inline Type create(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
  {
    return _mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0);
  }

  static inline Type rotate(Type value)
  {
    __m256 t0 = _mm256_permute_ps(value, _MM_SHUFFLE(0, 3, 2, 1));
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x01);
    __m256 y = _mm256_blend_ps(t0, t1, 0x88);
    return y;
  }

  static inline float least(Type value)
  {
    return _mm_cvtss_f32(_mm256_castps256_ps128(value));
  }

  static inline Type min(Type a, Type b)
  {
    return _mm256_min_ps(a, b);
  }

  static inline Type max(Type a, Type b)
  {
    return _mm256_max_ps(a, b);
  }

  static inline Type select(Type a, Type b, ConditionType condition)
  {
    return _mm256_add_ps(_mm256_and_ps(_mm256_castsi256_ps(condition), a), _mm256_andnot_ps(_mm256_castsi256_ps(condition), b));
  }

  static inline Type sqrt(Type value)
  {
    return _mm256_sqrt_ps(value);
  }

  static inline __m256i ifloor(Type value)
  {
    return _mm256_cvtps_epi32(_mm256_floor_ps(value));
  }

  static inline __m256i iceil(Type value)
  {
    return _mm256_cvtps_epi32(_mm256_ceil_ps(value));
  }

  static inline Type load(const float* src)
  {
    return _mm256_load_ps(src);
  }

  static inline Type loadUnaligned(const float* src)
  {
    return _mm256_loadu_ps(src);
  }

  static inline void store(float* dst, Type value)
  {
    _mm256_store_ps(dst, value);
  }

  static inline void storeUnaligned(float* dst, Type value)
  {
    _mm256_storeu_ps(dst, value);
  }

  static inline float reductionSum(Type value)
  {
    return reduce_mm256(value);
  }
};

static inline __m256 mul_add(__m256 t1, __m256 m1, __m256 m2)
{
  return _mm256_add_ps(t1, _mm256_mul_ps(m1, m2));
}

static inline __m256 mul_sub(__m256 me, __m256 m1, __m256 m2)
{
  return _mm256_sub_ps(me, _mm256_mul_ps(m1, m2));
}

template<typename FloatType> static inline constexpr int floatsPerSimdAVX();

// implementation

static inline float reduce_mm256(__m256 x256)
{
#if 1
  alignas(32) float x[8];
  _mm256_store_ps(x, x256);

  return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
#else
  return reduce_mm128(_mm_add_ps(_mm256_extractf128_ps(x256, 0), _mm256_extractf128_ps(x256, 1)));
#endif
}

template<typename FloatType> static inline constexpr int floatsPerSimdAVX()
{
  return 32 / sizeof(FloatType);
}

static inline double reduce_mm256d(__m256d x256)
{
  alignas(32) double x[4];
  _mm256_store_pd(x, x256);

  return x[0] + x[1] + x[2] + x[3];
}

}

}

#ifdef PLATFORM_COMPILER_MSVC
// float AVX operators

static inline __m256 operator-(__m256 a)
{
  return _mm256_sub_ps(_mm256_setzero_ps(), a);
}

static inline __m256 operator+(__m256 a, __m256 b)
{
  return _mm256_add_ps(a, b);
}

static inline __m256 operator-(__m256 a, __m256 b)
{
  return _mm256_sub_ps(a, b);
}

static inline __m256 operator*(__m256 a, __m256 b)
{
  return _mm256_mul_ps(a, b);
}

static inline __m256 operator/(__m256 a, __m256 b)
{
  return _mm256_div_ps(a, b);
}

static inline __m256 operator+=(__m256& a, __m256 b)
{
  return a = _mm256_add_ps(a, b);
}

static inline __m256i operator<(__m256 a, __m256 b)
{
  return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LT_OQ));
}

static inline __m256i operator<=(__m256 a, __m256 b)
{
  return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
}

static inline __m256i operator>(__m256 a, __m256 b)
{
  return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GT_OQ));
}

static inline __m256i operator>=(__m256 a, __m256 b)
{
  return _mm256_castps_si256(_mm256_cmp_ps(a, b, _CMP_GE_OQ));
}

static inline __m256i operator~(__m256i a)
{
  return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff))));
}

static inline __m256i operator&(__m256i a, __m256i b)
{
  return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
  //  return _mm256_and_si256(a, b); // AVX2
}

static inline __m256i operator|(__m256i a, __m256i b)
{
  return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
#endif
