#pragma once

#include <immintrin.h> // for _mm_blend_epi32()

#include "simd_int_sse.h"

#include "../simd_x86.h"

#define SIMD_INT4_SUPPORTED // TODO: remove
#define PLATFORM_CPU_FEATURE_INT32x4

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

  SseSimdIntType<int32_t> operator+(const SseSimdIntType<int32_t>& other) const;
  SseSimdIntType<int32_t> operator-(const SseSimdIntType<int32_t>& other) const;
  SseSimdIntType<int32_t>& operator+=(const SseSimdIntType<int32_t>& other);

  SseSimdIntType<int32_t> operator>>(int count) const;
  SseSimdIntType<int32_t> operator<<(int count) const;

  SseIntSimd<int32_t>::ConditionType operator<(const SseSimdIntType<int32_t>& other) const;

  template<int i0, int i1, int i2, int i3> 
  inline SseSimdIntType<int32_t> shuffled() const;
  template<int i0, int i1, int i2, int i3>
  static inline SseSimdIntType<int32_t> shuffle(__m128i a, __m128i b);

#ifdef PLATFORM_CPU_FEATURE_SSE41
  SseSimdIntType<int32_t> operator*(const SseSimdIntType<int32_t>& other) const;
  SseSimdIntType<int32_t> operator*(int32_t factor) const;
  SseSimdIntType<int32_t>& operator*=(int32_t factor);

  static inline SseSimdIntType<int32_t> fromPackedUint8(uint32_t packed);
  inline void setFromPackedUint8(uint32_t packed);
#endif
};

template<>
struct SIMD<int32_t, 4> : public SseIntSimd<int32_t>
{
  static bool isSupported(SimdFeatures features = 0);

  static inline Type populate(int value);

  static inline Type rotate(Type value);
  static inline int32_t least(Type value);

  static inline Type min(Type a, Type b);
  static inline Type max(Type a, Type b);

  static inline Type abs(Type a);
  static inline Type mulSign(Type a, Type sign);
  static inline Type mulExtended(Type a, Type b, Type& abhi);

  template<int dstStride = 1>
  static inline void transpose(Type* dst, Type w0, Type w1, Type w2, Type w3);
  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose(Type* dst, const int32_t* src);

  static void extractByteComponents(ParamType a, uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3);
  static void extractByteComponents(ParamType a, ParamType b, uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3);

  static inline Type interleaveLow16Bit(Type a, Type b);
};

// implementation

// SseSimdIntType<int32_t>

inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::operator+(const SseSimdIntType<int32_t>& other) const
{
  return SseSimdIntType<int32_t>::fromNativeType(_mm_add_epi32(value, other.value));
}

inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::operator-(const SseSimdIntType<int32_t>& other) const
{
  return SseSimdIntType<int32_t>::fromNativeType(_mm_sub_epi32(value, other.value));
}

inline SseSimdIntType<int32_t>& SseSimdIntType<int32_t>::operator+=(const SseSimdIntType<int32_t>& other)
{
  value = _mm_add_epi32(value, other.value);
  return *this;
}

inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::operator>>(int count) const
{
  return SseSimdIntType<int32_t>::fromNativeType(_mm_srai_epi32(value, count));
}

inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::operator<<(int count) const
{
  return SseSimdIntType<int32_t>::fromNativeType(_mm_slli_epi32(value, count));
}

inline SseIntSimd<int32_t>::ConditionType SseSimdIntType<int32_t>::operator<(const SseSimdIntType<int32_t>& other) const
{
  return SseIntSimd<int32_t>::ConditionType{_mm_cmplt_epi32(value, other.value)};
}

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

#ifdef PLATFORM_CPU_FEATURE_SSE41
inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::operator*(const SseSimdIntType<int32_t>& other) const
{
  return SseSimdIntType<int32_t>::fromNativeType(_mm_mullo_epi32(value, other.value));
}

inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::operator*(int32_t factor) const
{
  return SseSimdIntType<int32_t>::fromNativeType(_mm_mullo_epi32(value, _mm_set1_epi32(factor)));
}

inline SseSimdIntType<int32_t>& SseSimdIntType<int32_t>::operator*=(int32_t factor)
{
  value = _mm_mullo_epi32(value, _mm_set1_epi32(factor));
  return *this;
}

inline SseSimdIntType<int32_t> SseSimdIntType<int32_t>::fromPackedUint8(uint32_t packed)
{
  return _mm_cvtepu8_epi32(_mm_set1_epi32(packed));
}

inline void SseSimdIntType<int32_t>::setFromPackedUint8(uint32_t packed)
{
  value = _mm_cvtepu8_epi32(_mm_set1_epi32(packed));
}
#endif

// SIMD<int32_t, 4>

inline bool SIMD<int32_t, 4>::isSupported(SimdFeatures features)
{
  static bool sse41Enabled = isSSE41Enabled();
  if ((features & SimdFeature::Multiplication) && !sse41Enabled)
    return false;

  return SseIntSimd<int32_t>::isSupported(features);
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
  abhi = Type{_mm_blend_epi32(_mm_shuffle_epi32(ab02, _MM_SHUFFLE(2, 3, 0, 1)), ab13, 0xa)};
  return Type{_mm_blend_epi32(ab02, _mm_shuffle_epi32(ab13, _MM_SHUFFLE(2, 3, 0, 1)), 0xa)};
}

template<int dstStride>
inline void SIMD<int32_t, 4>::transpose(Type* dst, Type w0, Type w1, Type w2, Type w3)
{
  __m128 v0 = _mm_castsi128_ps(w0), v1 = _mm_castsi128_ps(w1), v2 = _mm_castsi128_ps(w2), v3 = _mm_castsi128_ps(w3);
  _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
  dst[0 * dstStride] = _mm_castps_si128(v0);
  dst[1 * dstStride] = _mm_castps_si128(v1);
  dst[2 * dstStride] = _mm_castps_si128(v2);
  dst[3 * dstStride] = _mm_castps_si128(v3);
}

template<bool aligned, int dstStride, int srcStride>
inline void SIMD<int32_t, 4>::transpose(Type* dst, const int32_t* src)
{
  transpose<dstStride>(dst, load<aligned>(src + 0 * 4 * srcStride), load<aligned>(src + 1 * 4 * srcStride), load<aligned>(src + 2 * 4 * srcStride), load<aligned>(src + 3 * 4 * srcStride));
}

inline void SIMD<int32_t, 4>::extractByteComponents(ParamType a, uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3)
{
                                                             // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
  __m128i b = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 2, 3, 2)); // 02 12 22 32 03 13 23 33 ...

  __m128i c = _mm_unpacklo_epi8(a, b);               // 00 02 10 12 20 22 30 32 01 03 11 13 21 23 31 33
  b = _mm_shuffle_epi32(c, _MM_SHUFFLE(3, 2, 3, 2)); // 01 03 11 13 21 23 31 33 ...
  c = _mm_unpacklo_epi8(c, b);                       // 00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33

  c0 = _mm_extract_epi32(c, 0);
  c1 = _mm_extract_epi32(c, 1);
  c2 = _mm_extract_epi32(c, 2);
  c3 = _mm_extract_epi32(c, 3);
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

}

}
