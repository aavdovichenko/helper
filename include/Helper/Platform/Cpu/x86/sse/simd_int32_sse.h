#pragma once

#include "simd_int_sse.h"

#include "../simd_x86.h"

#define SIMD_INT4_SUPPORTED

#if SIMD_INT_MAX_WIDTH < 4
#undef SIMD_INT_MAX_WIDTH
#define SIMD_INT_MAX_WIDTH 4
#endif

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<int32_t> : public BaseSseSimdIntType<int32_t, SseSimdIntType<int32_t>>
{
  using BaseSseSimdIntType<int32_t, SseSimdIntType<int32_t>>::BaseSseSimdIntType;

  template<int i0, int i1, int i2, int i3> 
  inline SseSimdIntType<int32_t> shuffled() const;
  template<int i0, int i1, int i2, int i3>
  static inline SseSimdIntType<int32_t> shuffle(__m128i a, __m128i b);
};

template<>
struct SIMD<int32_t, 4> : public SseIntSimd<int32_t>
{
  static inline Type populate(int value);

  static inline Type rotate(Type value);
  static inline int32_t least(Type value);

  static inline Type min(Type a, Type b);
  static inline Type max(Type a, Type b);

  static inline Type abs(Type a);
  static inline Type mulSign(Type a, Type sign);
  static inline Type mulExtended(Type a, Type b, Type& abhi);

  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose(Type* dst, const int* src);

  static void extractByteComponents(ParamType a, ParamType b, uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3);

  static inline Type interleaveLow16Bit(Type a, Type b);
};

// implementation

template<int i0, int i1, int i2, int i3>
inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::shuffled() const
{
  static_assert(i0 >= 0 && i0 < 4 && i1 >= 0 && i1 < 4 && i2 >= 0 && i2 < 4 && i3 >= 0 && i3 < 4, "invalid index");
  return SseSimdIntType<int32_t>{_mm_shuffle_epi32(value, _MM_SHUFFLE(i3, i2, i1, i0))};
}

template<int i0, int i1, int i2, int i3>
inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::shuffle(__m128i a, __m128i b)
{
  static_assert(i0 >= 0 && i0 < 8 && i1 >= 0 && i1 < 8 && i2 >= 0 && i2 < 8 && i3 >= 0 && i3 < 8, "invalid index");
  static_assert((i0 == 0 && i1 == 4 && i2 == 1 && i3 == 5) || (i0 == 2 && i1 == 6 && i2 == 3 && i3 == 7), "not implemented");
  if (i0 == 0 && i1 == 4 && i2 == 1 && i3 == 5)
    return SseSimdIntType<int32_t>{_mm_unpacklo_epi32(a, b)};
  if (i0 == 2 && i1 == 6 && i2 == 3 && i3 == 7)
    return SseSimdIntType<int32_t>{_mm_unpackhi_epi32(a, b)};
  // TODO: implement more
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::populate(int32_t value)
{
  return Type{_mm_set1_epi32(value)};
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::rotate(Type value)
{
  return Type{_mm_shuffle_epi32(value, _MM_SHUFFLE(0, 3, 2, 1))};
}

inline int32_t SIMD<int32_t, 4>::least(Type value)
{
  return _mm_cvtsi128_si32(value);
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::min(Type a, Type b)
{
  __m128i cmp = _mm_cmplt_epi32(a, b);
  return Type{_mm_add_epi32(_mm_and_si128(cmp, a), _mm_andnot_si128(cmp, b))};
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::max(Type a, Type b)
{
  __m128i cmp = _mm_cmpgt_epi32(a, b);
  return Type{_mm_add_epi32(_mm_and_si128(cmp, a), _mm_andnot_si128(cmp, b))};
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::abs(Type a)
{
  return Type{_mm_abs_epi32(a)};
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::mulSign(Type a, Type sign)
{
  return Type{_mm_sign_epi32(a, sign)};
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::mulExtended(Type a, Type b, Type& abhi)
{
  __m128i ab02 = _mm_mul_epi32(a, b);
  __m128i ab13 = _mm_mul_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1)), _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1)));
  abhi = Type{_mm_blend_epi32(_mm_shuffle_epi32(ab02, _MM_SHUFFLE(2, 3, 0, 1)), ab13, 0xaa)};
  return Type{_mm_blend_epi32(ab02, _mm_shuffle_epi32(ab13, _MM_SHUFFLE(2, 3, 0, 1)), 0xaa)};
}

template<bool aligned, int dstStride, int srcStride>
inline void SIMD<int32_t, 4>::transpose(Type* dst, const int* src)
{
  __m128 v0 = _mm_cvtepi32_ps(load<aligned>(src + 0 * 4 * srcStride)), v1 = _mm_cvtepi32_ps(load<aligned>(src + 1 * 4 * srcStride));
  __m128 v2 = _mm_cvtepi32_ps(load<aligned>(src + 2 * 4 * srcStride)), v3 = _mm_cvtepi32_ps(load<aligned>(src + 3 * 4 * srcStride));
  _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
  dst[0 * dstStride] = _mm_cvtps_epi32(v0);
  dst[1 * dstStride] = _mm_cvtps_epi32(v1);
  dst[2 * dstStride] = _mm_cvtps_epi32(v2);
  dst[3 * dstStride] = _mm_cvtps_epi32(v3);
}

inline void SIMD<int32_t, 4>::extractByteComponents(ParamType w0, ParamType w1, uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3)
{
  // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
  // 04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37

  __m128i c = _mm_unpacklo_epi8(w0.value, w1.value); // 00 04 10 14 20 24 30 34 01 05 11 15 21 25 31 35
  __m128i d = _mm_unpackhi_epi8(w0.value, w1.value); // 02 06 12 16 22 26 32 36 03 07 13 17 23 27 33 37

  __m128i a = _mm_unpacklo_epi8(c, d); // 00 02 04 06 10 12 14 16 20 22 24 26 30 32 34 36
  __m128i b = _mm_unpackhi_epi8(c, d); // 01 03 05 07 11 13 15 17 21 23 25 27 31 33 35 37

  c = _mm_unpacklo_epi8(a, b); // 00 01 02 03 04 05 06 07 10 11 12 13 14 15 16 17
  d = _mm_unpackhi_epi8(a, b); // 20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 37

  c0 = _mm_extract_epi64(c, 0);
  c1 = _mm_extract_epi64(c, 1);
  c2 = _mm_extract_epi64(d, 0);
  c3 = _mm_extract_epi64(d, 1);
}

inline SIMD<int32_t, 4>::Type SIMD<int32_t, 4>::interleaveLow16Bit(Type a, Type b)
{
  return Type{_mm_blend_epi16(a, _mm_shufflehi_epi16(_mm_shufflelo_epi16(b, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1)), 0xaa)};
}

// int32 SSE operators

namespace int32
{

static inline SIMD<int32_t, 4>::Type operator~(SIMD<int32_t, 4>::Type a)
{
  return SIMD<int32_t, 4>::Type{_mm_castps_si128(_mm_xor_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(_mm_set1_epi32(0xffffffff))))};
}

static inline SIMD<int32_t, 4>::Type operator&(SIMD<int32_t, 4>::Type a, SIMD<int32_t, 4>::Type b)
{
  return SIMD<int32_t, 4>::Type{_mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)))};
}

static inline SIMD<int32_t, 4>::Type operator|(SIMD<int32_t, 4>::Type a, SIMD<int32_t, 4>::Type b)
{
  return SIMD<int32_t, 4>::Type{_mm_castps_si128(_mm_or_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)))};
}

static inline SIMD<int32_t, 4>::Type operator+(SIMD<int32_t, 4>::Type a, SIMD<int32_t, 4>::Type b)
{
  return SIMD<int32_t, 4>::Type{_mm_add_epi32(a, b)};
}

static inline SIMD<int32_t, 4>::Type operator+=(SIMD<int32_t, 4>::Type& a, SIMD<int32_t, 4>::Type b)
{
  return a = SIMD<int32_t, 4>::Type{_mm_add_epi32(a, b)};
}

static inline SIMD<int32_t, 4>::Type operator-(SIMD<int32_t, 4>::Type a, SIMD<int32_t, 4>::Type b)
{
  return SIMD<int32_t, 4>::Type{_mm_sub_epi32(a, b)};
}

#ifndef PLATFORM_CPU_FEATURE_NO_SSE41
static inline SIMD<int32_t, 4>::Type operator*(SIMD<int32_t, 4>::Type a, SIMD<int32_t, 4>::Type b)
{
  return SIMD<int32_t, 4>::Type{_mm_mullo_epi32(a, b)};
}

static inline SIMD<int32_t, 4>::Type operator*(SIMD<int32_t, 4>::Type a, int32_t b)
{
  return SIMD<int32_t, 4>::Type{_mm_mullo_epi32(a, _mm_set1_epi32(b))};
}

static inline SIMD<int32_t, 4>::Type operator*=(SIMD<int32_t, 4>::Type& a, int32_t b)
{
  return a = SIMD<int32_t, 4>::Type{_mm_mullo_epi32(a, _mm_set1_epi32(b))};
}
#endif

static inline SIMD<int32_t, 4>::Type operator<<(SIMD<int32_t, 4>::Type a, int count)
{
  return SIMD<int32_t, 4>::Type{_mm_slli_epi32(a, count)};
}

static inline SIMD<int32_t, 4>::Type operator>>(SIMD<int32_t, 4>::Type a, int count)
{
  return SIMD<int32_t, 4>::Type{_mm_srai_epi32(a, count)};
}

static inline SIMD<int32_t, 4>::ConditionType operator<(SIMD<int32_t, 4>::Type a, SIMD<int32_t, 4>::Type b)
{
  return SIMD<int32_t, 4>::ConditionType{_mm_cmplt_epi32(a, b)};
}

}

}

}
