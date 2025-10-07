#pragma once

#include "simd_int_sse.h"

#define PLATFORM_CPU_FEATURE_INT16x8

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<int16_t> : public BaseSseSimdIntType<int16_t, SseSimdIntType<int16_t>>
{
  using BaseSseSimdIntType<int16_t, SseSimdIntType<int16_t>>::BaseSseSimdIntType;

  static inline SseSimdIntType<int16_t> populate(int value);
  static inline SseSimdIntType<int16_t> create(int16_t v0, int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7);

  SseSimdIntType<int16_t> operator+(const SseSimdIntType<int16_t>& other) const;
  SseSimdIntType<int16_t> operator-(const SseSimdIntType<int16_t>& other) const;
  SseSimdIntType<int16_t> operator>>(int count) const;
  SseSimdIntType<int16_t> operator<<(int count) const;

  SseSimdIntType<int16_t>& operator+=(const SseSimdIntType<int16_t>& other);

  SseIntSimd<int16_t>::ConditionType operator==(const SseSimdIntType<int16_t>& other) const;
  SseIntSimd<int16_t>::ConditionType operator<(const SseSimdIntType<int16_t>& other) const;

  template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
  inline SseSimdIntType<int16_t>& shuffle();
  template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
  inline SseSimdIntType<int16_t> shuffled() const;

  template<int n> inline SseSimdIntType<int16_t>& shiftWordsUp();
  template<int n> inline SseSimdIntType<int16_t>& shiftWordsDown();

#ifdef PLATFORM_CPU_FEATURE_SSE41
  static inline SseSimdIntType<int16_t> fromPackedUint8(uint64_t packed);
  inline void setFromPackedUint8(uint64_t packed);
#endif

  inline SseSimdIntType<int16_t> onesComplement() const;
};

template<>
struct SIMD<int16_t, 8> : public SseIntSimd<int16_t>
{
  typedef SseSimdIntType<int8_t> Int8Type;

  struct ExtendedType
  {
    __m128i lo, hi;

    static ExtendedType populate(int32_t value);

    template <int fixedPointBits> Type descale() const;
    template <int fixedPointBits> Type round() const;

    ExtendedType operator+(ExtendedType other) const;
    ExtendedType operator+=(ExtendedType other);
    ExtendedType operator<<(int shift);
  };

  static inline Type populate(int value);

  template<int i>
  static inline Type insert(Type a, int16_t x);

#ifdef PLATFORM_CPU_FEATURE_SSSE3
  static inline Type abs(Type a);
  static inline Type mulSign(Type a, Type sign);
#endif
  static inline Type mulFixedPoint(Type a, Type b);

  static inline ExtendedType extend(ParamType value); // (int32)(value)
  static inline ExtendedType mulExtended(Type a, Type b); // (int32)(a*b)
  static inline ExtendedType mulExtended(Type a, int16_t factor); // (int32)(a*factor)
  static inline ExtendedType mulAdd(Type a, Type b, Type c, Type d); // (int32)(a*b + c*d)
  static inline ExtendedType mulAdd(Type a, int16_t afactor, Type b, int16_t bfactor); // (int32)(a*afactor + b*bfactor)
  template <int16_t aFactor, int16_t bFactor>
  static inline ExtendedType mulAdd(ParamType a, ParamType b); // (int32)(a*aFactor + b*bFactor)

  template<int dstStride = 1>
  static inline void transpose(Type* dst, Type w0, Type w1, Type w2, Type w3, Type w4, Type w5, Type w6, Type w7);
  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose(Type* dst, const int16_t* src);

  static int64_t conditionBitMask(ConditionParamType c0, ConditionParamType c1);
};

// implementation

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::populate(int value)
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_set1_epi16(value));
}

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::create(int16_t v0, int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7)
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7));
}

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::operator+(const SseSimdIntType<int16_t>& other) const
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_add_epi16(value, other.value));
}

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::operator-(const SseSimdIntType<int16_t>& other) const
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_sub_epi16(value, other.value));
}

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::operator>>(int count) const
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_srai_epi16(value, count));
}

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::operator<<(int count) const
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_slli_epi16(value, count));
}

inline SseSimdIntType<int16_t>& SseSimdIntType<int16_t>::operator+=(const SseSimdIntType<int16_t>& other)
{
  value = _mm_add_epi16(value, other.value);
  return *this;
}

inline SseIntSimd<int16_t>::ConditionType SseSimdIntType<int16_t>::operator==(const SseSimdIntType<int16_t>& other) const
{
  return SseIntSimd<int16_t>::ConditionType{_mm_cmpeq_epi16(value, other.value)};
}

inline SseIntSimd<int16_t>::ConditionType SseSimdIntType<int16_t>::operator<(const SseSimdIntType<int16_t>& other) const
{
  return SseIntSimd<int16_t>::ConditionType{_mm_cmplt_epi16(value, other.value)};
}

#ifdef PLATFORM_CPU_FEATURE_SSE41
inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::fromPackedUint8(uint64_t packed)
{
  return _mm_cvtepu8_epi16(_mm_set1_epi64x(packed));
}

inline void SseSimdIntType<int16_t>::setFromPackedUint8(uint64_t packed)
{
  value = _mm_cvtepu8_epi16(_mm_set1_epi64x(packed));
}
#endif

inline SseSimdIntType<int16_t> SseSimdIntType<int16_t>::onesComplement() const
{
  return SseSimdIntType<int16_t>::fromNativeType(_mm_add_epi16(value, _mm_cmplt_epi16(value, _mm_setzero_si128())));
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline SseSimdIntType<int16_t>& Platform::Cpu::SseSimdIntType<int16_t>::shuffle()
{
  value = shuffled<i0, i1, i2, i3, i4, i5, i6, i7>().value;
  return *this;
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
inline SseSimdIntType<int16_t> Platform::Cpu::SseSimdIntType<int16_t>::shuffled() const
{
  static_assert(i0 >= 0 && i1 >= 0 && i2 >= 0 && i3 >= 0 && i4 >= 0 && i5 >= 0 && i6 >= 0 && i7 >= 0, "invalid index");
  static_assert(i0 < 8 && i1 < 8 && i2 < 8 && i3 < 8 && i4 < 8 && i5 < 8 && i6 < 8 && i7 < 8, "invalid index");
  static_assert((i0 < 4 && i1 < 4 && i2 < 4 && i3 < 4) && (i4 >= 4 && i5 >= 4 && i6 >= 4 && i7 >= 4), "not implemented"); // TODO: implement

  if (i0 != 0 || i1 != 1 || i2 != 2 || i3 != 3)
    return _mm_shufflelo_epi16(value, _MM_SHUFFLE(i3, i2, i1, i0));
  if (i4 != 4 || i5 != 5 || i6 != 6 || i7 != 7)
    return _mm_shufflehi_epi16(value, _MM_SHUFFLE(i7 - 4, i6 - 4, i5 - 4, i4 - 4));

  return value;
}

template<int n>
inline SseSimdIntType<int16_t>& SseSimdIntType<int16_t>::shiftWordsUp()
{
  value = _mm_slli_si128(value, n * 2);
  return *this;
}

template<int n>
inline SseSimdIntType<int16_t>& SseSimdIntType<int16_t>::shiftWordsDown()
{
  value = _mm_srli_si128(value, n * 2);
  return *this;
}

template<> template<int i>
inline SseSimdIntType<int16_t>& SimdIntType<int16_t, __m128i, SseSimdIntType<int16_t>>::insert(int16_t x)
{
  value = _mm_insert_epi16(value, x, i);
  return *static_cast<SseSimdIntType<int16_t>*>(this);
}

inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::populate(int value)
{
  return Type{_mm_set1_epi16(value)};
}

inline int64_t SIMD<int16_t, 8>::conditionBitMask(ConditionParamType c0, ConditionParamType c1)
{
  return _mm_movemask_epi8(_mm_packs_epi16(c0.value, c1.value));
}

template<int i>
inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::insert(Type a, int16_t x)
{
  return Type{_mm_insert_epi16(a, x, i)};
}

#ifdef PLATFORM_CPU_FEATURE_SSSE3
inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::abs(Type a)
{
  return Type{_mm_abs_epi16(a)};
}

inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::mulSign(Type a, Type sign)
{
  return Type{_mm_sign_epi16(a, sign)};
}
#endif

inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::mulFixedPoint(Type a, Type b)
{
  return Type{_mm_mulhi_epu16(a, b)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::extend(ParamType value)
{
  __m128i sign = _mm_cmpgt_epi16(_mm_setzero_si128(), value.value);
  return ExtendedType{_mm_unpacklo_epi16(value.value, sign), _mm_unpackhi_epi16(value.value, sign)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::mulExtended(Type a, Type b)
{
  __m128i ablo = _mm_mullo_epi16(a, b);
  __m128i abhi = _mm_mulhi_epi16(a, b);
  return ExtendedType{_mm_unpacklo_epi16(ablo, abhi), _mm_unpackhi_epi16(ablo, abhi)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::mulExtended(Type a, int16_t factor)
{
  __m128i b = _mm_set1_epi16(factor);
  __m128i ablo = _mm_mullo_epi16(a, b);
  __m128i abhi = _mm_mulhi_epi16(a, b);
  return ExtendedType{_mm_unpacklo_epi16(ablo, abhi), _mm_unpackhi_epi16(ablo, abhi)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::mulAdd(Type a, Type b, Type c, Type d)
{
  __m128i aclo = _mm_unpacklo_epi16(a, c);
  __m128i bdlo = _mm_unpacklo_epi16(b, d);
  __m128i achi = _mm_unpackhi_epi16(a, c);
  __m128i bdhi = _mm_unpackhi_epi16(b, d);

  return ExtendedType{_mm_madd_epi16(aclo, bdlo), _mm_madd_epi16(achi, bdhi)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::mulAdd(Type a, int16_t afactor, Type b, int16_t bfactor)
{
  __m128i factor = _mm_unpacklo_epi16(_mm_set1_epi16(afactor), _mm_set1_epi16(bfactor));
  __m128i ablo = _mm_unpacklo_epi16(a, b);
  __m128i abhi = _mm_unpackhi_epi16(a, b);
  return ExtendedType{_mm_madd_epi16(ablo, factor), _mm_madd_epi16(abhi, factor)};
}

template<int16_t aFactor, int16_t bFactor>
inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::mulAdd(ParamType a, ParamType b)
{
  __m128i factor = _mm_setr_epi16(aFactor, bFactor, aFactor, bFactor, aFactor, bFactor, aFactor, bFactor);
  return ExtendedType{_mm_madd_epi16(_mm_unpacklo_epi16(a.value, b.value), factor), _mm_madd_epi16(_mm_unpackhi_epi16(a.value, b.value), factor)};
}

template<int fixedPointBits>
inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::ExtendedType::descale() const
{
  return Type{_mm_packs_epi32(_mm_srai_epi32(lo, fixedPointBits), _mm_srai_epi32(hi, fixedPointBits))};
}

template<int fixedPointBits>
inline SIMD<int16_t, 8>::Type SIMD<int16_t, 8>::ExtendedType::round() const
{
  __m128i rounding = _mm_set1_epi32(1 << (fixedPointBits - 1));
  return Type{_mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(lo, rounding), fixedPointBits), _mm_srai_epi32(_mm_add_epi32(hi, rounding), fixedPointBits))};
}

template<int dstStride>
inline void SIMD<int16_t, 8>::transpose(Type* dst, Type w0, Type w1, Type w2, Type w3, Type w4, Type w5, Type w6, Type w7)
{
  // 00 01 02 03 04 05 06 07
  // 10 11 12 13 14 15 16 17
  // 20 21 22 23 24 25 26 27
  // 30 31 32 33 34 35 36 37
  // 40 41 42 43 44 45 46 47
  // 50 51 52 53 54 55 56 57
  // 60 61 62 63 64 65 66 67
  // 70 71 72 73 74 75 76 77

  __m128i tmp0 = _mm_unpacklo_epi16(w0, w1); // 00 10 01 11 02 12 03 13
  __m128i tmp1 = _mm_unpackhi_epi16(w0, w1); // 04 14 05 15 06 16 07 17 
  __m128i tmp2 = _mm_unpacklo_epi16(w2, w3); // 20 30 21 31 22 32 23 33
  __m128i tmp3 = _mm_unpackhi_epi16(w2, w3); // 24 34 25 35 26 36 27 37
  __m128i tmp4 = _mm_unpacklo_epi16(w4, w5); // 40 50 41 51 42 52 43 53
  __m128i tmp5 = _mm_unpackhi_epi16(w4, w5); // 44 54 45 55 46 56 47 57
  __m128i tmp6 = _mm_unpacklo_epi16(w6, w7); // 60 70 61 71 62 72 63 73
  __m128i tmp7 = _mm_unpackhi_epi16(w6, w7); // 64 74 65 75 66 76 67 77

  w0 = _mm_unpacklo_epi32(tmp0, tmp2); // 00 10 20 30 01 11 21 31
  w1 = _mm_unpackhi_epi32(tmp0, tmp2); // 02 12 22 32 03 13 23 33
  w2 = _mm_unpacklo_epi32(tmp1, tmp3); // 04 14 24 34 05 15 25 35
  w3 = _mm_unpackhi_epi32(tmp1, tmp3); // 06 16 26 36 07 17 27 37
  w4 = _mm_unpacklo_epi32(tmp4, tmp6); // 40 50 60 70 41 51 61 71 
  w5 = _mm_unpackhi_epi32(tmp4, tmp6); // 42 52 62 72 43 53 63 73
  w6 = _mm_unpacklo_epi32(tmp5, tmp7); // 44 54 64 74 45 55 65 75
  w7 = _mm_unpackhi_epi32(tmp5, tmp7); // 46 56 66 76 47 57 67 77
  
  dst[0 * dstStride].value = _mm_unpacklo_epi64(w0, w4); // 00 10 20 30 40 50 60 70
  dst[1 * dstStride].value = _mm_unpackhi_epi64(w0, w4); // 01 11 21 31 41 51 61 71 
  dst[2 * dstStride].value = _mm_unpacklo_epi64(w1, w5); // 02 12 22 32 42 52 62 72
  dst[3 * dstStride].value = _mm_unpackhi_epi64(w1, w5); // 03 13 23 33 43 53 63 73
  dst[4 * dstStride].value = _mm_unpacklo_epi64(w2, w6); // 04 14 24 34 44 54 64 74
  dst[5 * dstStride].value = _mm_unpackhi_epi64(w2, w6); // 05 15 25 35 45 55 65 75
  dst[6 * dstStride].value = _mm_unpacklo_epi64(w3, w7); // 06 16 26 36 46 56 66 76
  dst[7 * dstStride].value = _mm_unpackhi_epi64(w3, w7); // 07 17 27 37 47 57 67 77
}

template<bool aligned, int dstStride, int srcStride>
inline void SIMD<int16_t, 8>::transpose(Type* dst, const int16_t* src)
{
  transpose(dst,
    load<aligned>(src + 8 * 0 * srcStride), load<aligned>(src + 8 * 1 * srcStride),
    load<aligned>(src + 8 * 2 * srcStride), load<aligned>(src + 8 * 3 * srcStride),
    load<aligned>(src + 8 * 4 * srcStride), load<aligned>(src + 8 * 5 * srcStride),
    load<aligned>(src + 8 * 6 * srcStride), load<aligned>(src + 8 * 7 * srcStride));
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::ExtendedType::populate(int32_t value)
{
  return ExtendedType{_mm_set1_epi32(value), _mm_set1_epi32(value)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::ExtendedType::operator+(ExtendedType other) const
{
  return ExtendedType{_mm_add_epi32(lo, other.lo), _mm_add_epi32(hi, other.hi)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::ExtendedType::operator+=(ExtendedType other)
{
  return *this = ExtendedType{_mm_add_epi32(lo, other.lo), _mm_add_epi32(hi, other.hi)};
}

inline SIMD<int16_t, 8>::ExtendedType SIMD<int16_t, 8>::ExtendedType::operator<<(int shift)
{
  return ExtendedType{_mm_slli_epi32(lo, shift), _mm_slli_epi32(hi, shift)};
}

}

}
