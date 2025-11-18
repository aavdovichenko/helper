#pragma once

#include "simd_int_avx.h"
#include "../sse/simd_int8_sse.h"
#include "../sse/simd_uint8_sse.h"

#define PLATFORM_CPU_FEATURE_INT16x16

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<int16_t> : public BaseAvxSimdIntType<int16_t, AvxSimdIntType<int16_t>>
{
  using BaseAvxSimdIntType<int16_t, AvxSimdIntType<int16_t>>::BaseAvxSimdIntType;

  static inline AvxSimdIntType<int16_t> populate(int16_t value);
  static inline AvxSimdIntType<int16_t> create(int16_t v0, int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7,
    int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15);

  AvxSimdIntType<int16_t> operator+(const AvxSimdIntType<int16_t>& other) const;
  AvxSimdIntType<int16_t> operator-(const AvxSimdIntType<int16_t>& other) const;
  AvxSimdIntType<int16_t> operator>>(int count) const;
  AvxSimdIntType<int16_t> operator<<(int count) const;

  AvxSimdIntType<int16_t>& operator+=(const AvxSimdIntType<int16_t>& other);

  AvxSimdIntConditionType<int16_t> operator==(const AvxSimdIntType<int16_t>& other) const;
  AvxSimdIntConditionType<int16_t> operator!=(const AvxSimdIntType<int16_t>& other) const;
  AvxSimdIntConditionType<int16_t> operator<(const AvxSimdIntType<int16_t>& other) const;

  static inline AvxSimdIntType<int16_t> fromPackedInt8(SIMD<int8_t, 16>::ParamType packed);
  static inline AvxSimdIntType<int16_t> fromPackedUint8(SIMD<uint8_t, 16>::ParamType packed);

  inline void setFromPackedUint8(SIMD<uint8_t, 16>::ParamType packed);

  template<bool aligned> static inline AvxSimdIntType<int16_t> loadAndConvert(const int8_t* p);
  template<bool aligned> static inline AvxSimdIntType<int16_t> loadAndConvert(const uint8_t* p);

  template<bool aligned> inline void convertAndStore(int8_t* p) const;
  template<bool aligned> inline void convertAndStore(uint8_t* p) const;

  inline AvxSimdIntType<int16_t> onesComplement() const;
};

template<>
struct SIMD<int16_t, 16> : public AvxIntSimd<int16_t>
{
  typedef BaseAvxSimdIntType<int32_t, AvxSimdIntType<int32_t>> Int32Type;

  typedef __m256i MulAddFactors;
  struct ExtendedType
  {
    typedef int32_t ItemType;

    __m256i lo, hi;

    static inline ExtendedType zero();
    static inline ExtendedType populate(int32_t value);

    inline void clamp(const ExtendedType& min, const ExtendedType& max);

    template <int fixedPointBits> Type descale() const;
    template <int fixedPointBits> Type round() const;

    ExtendedType operator+(const ExtendedType& other) const;
    ExtendedType operator+=(const ExtendedType& other);
    ExtendedType operator<<(int shift);
  };

  static inline Type populate(int16_t value);

  static inline Type min(ParamType a, ParamType b);
  static inline Type max(ParamType a, ParamType b);

  static inline Type abs(ParamType a);
  static inline Type mulSign(ParamType a, ParamType sign);
  static inline Type mulFixedPoint(ParamType a, ParamType b);

  static inline ExtendedType extend(ParamType value); // (int32)(value)
  static inline ExtendedType mulExtended(ParamType a, ParamType b); // (int32)(a*b)
  static inline ExtendedType mulExtended(ParamType a, int16_t factor); // (int32)(a*factor)
  static inline MulAddFactors makeMulAddFactors(int16_t afactor, int16_t bfactor);
  static inline ExtendedType mulAdd(ParamType a, ParamType b, ParamType c, ParamType d); // (int32)(a*b + c*d)
  static inline ExtendedType mulAdd(ParamType a, int16_t afactor, ParamType b, int16_t bfactor); // (int32)(a*afactor + b*bfactor)
  static inline ExtendedType mulAdd(ParamType a, ParamType b, MulAddFactors factors); // (int32)(a*factors.afactor + b*factors.bfactor)
  template <int16_t aFactorLo, int16_t aFactorHi, int16_t bFactorLo, int16_t bFactorHi>
  static inline ExtendedType mulAdd(ParamType a, ParamType b); // (int32)(a*{aFactorHi, aFactorLo} + b*{bFactorHi, bFactorLo})
  template <int16_t aFactor, int16_t bFactor>
  static inline ExtendedType mulAdd(ParamType a, ParamType b); // (int32)(a*aFactor + b*bFactor)

  static inline Int32Type horizontalMulAdd(Type a, Type b);

  static inline Type interleaveEach4Low(Type a, Type b);
  static inline Type interleaveEach4High(Type a, Type b);

  template<int dstStride = 1>
  static inline void transpose2x8x8(Type* dst, ParamType w0, ParamType w1, ParamType w2, ParamType w3, ParamType w4, ParamType w5, ParamType w6, ParamType w7);
  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose2x8x8(Type* dst, const int16_t* src);
};

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::populate(int16_t value)
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_set1_epi16(value));
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::create(int16_t v0, int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7, int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15)
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15));
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::operator+(const AvxSimdIntType<int16_t>& other) const
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_add_epi16(value, other.value));
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::operator-(const AvxSimdIntType<int16_t>& other) const
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_sub_epi16(value, other.value));
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::operator>>(int count) const
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_srai_epi16(value, count));
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::operator<<(int count) const
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_slli_epi16(value, count));
}

inline AvxSimdIntType<int16_t>& AvxSimdIntType<int16_t>::operator+=(const AvxSimdIntType<int16_t>& other)
{
  value = _mm256_add_epi16(value, other.value);
  return *this;
}

inline AvxSimdIntConditionType<int16_t> AvxSimdIntType<int16_t>::operator==(const AvxSimdIntType<int16_t>& other) const
{
  return AvxSimdIntConditionType<int16_t>::fromNativeType(_mm256_cmpeq_epi16(value, other.value));
}

inline AvxSimdIntConditionType<int16_t> AvxSimdIntType<int16_t>::operator!=(const AvxSimdIntType<int16_t>& other) const
{
  return AvxSimdIntConditionType<int16_t>::fromNativeType(_mm256_xor_si256(_mm256_cmpeq_epi16(value, other.value), _mm256_set1_epi32(0xffffffff)));
}

inline AvxSimdIntConditionType<int16_t> AvxSimdIntType<int16_t>::operator<(const AvxSimdIntType<int16_t>& other) const
{
  return AvxSimdIntConditionType<int16_t>::fromNativeType(_mm256_cmpgt_epi16(other.value, value));
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::fromPackedInt8(SIMD<int8_t, 16>::ParamType packed)
{
  return _mm256_cvtepi8_epi16(packed.value);
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::fromPackedUint8(SIMD<uint8_t, 16>::ParamType packed)
{
  return _mm256_cvtepu8_epi16(packed.value);
}

inline void AvxSimdIntType<int16_t>::setFromPackedUint8(SIMD<uint8_t, 16>::ParamType packed)
{
  value = _mm256_cvtepu8_epi16(packed.value);
}

template<bool aligned>
inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::loadAndConvert(const int8_t* p)
{
  return _mm256_cvtepi8_epi16(aligned ? _mm_load_si128((const __m128i*)p) : _mm_loadu_si128((const __m128i*)p));
}

template<bool aligned>
inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::loadAndConvert(const uint8_t* p)
{
  return _mm256_cvtepu8_epi16(aligned ? _mm_load_si128((const __m128i*)p) : _mm_loadu_si128((const __m128i*)p));
}

template<bool aligned>
inline void AvxSimdIntType<int16_t>::convertAndStore(int8_t* p) const
{
  SseSimdIntType<int8_t>::fromNativeType(_mm256_castsi256_si128(_mm256_packs_epi16(value, _mm256_permute2x128_si256(value, value, 1)))).store<aligned>(p);
}

template<bool aligned>
inline void AvxSimdIntType<int16_t>::convertAndStore(uint8_t* p) const
{
  SseSimdIntType<uint8_t>::fromNativeType(_mm256_castsi256_si128(_mm256_packus_epi16(value, _mm256_permute2x128_si256(value, value, 1)))).store<aligned>(p);
}

inline AvxSimdIntType<int16_t> AvxSimdIntType<int16_t>::onesComplement() const
{
  return AvxSimdIntType<int16_t>::fromNativeType(_mm256_add_epi16(value, _mm256_cmpgt_epi16(_mm256_setzero_si256(), value)));
}

inline typename SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::populate(int16_t value)
{
  return Type{_mm256_set1_epi16(value)};
}

inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::min(ParamType a, ParamType b)
{
  return Type{_mm256_min_epi16(a, b)};
}

inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::max(ParamType a, ParamType b)
{
  return Type{_mm256_max_epi16(a, b)};
}

inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::abs(ParamType a)
{
  return Type{_mm256_abs_epi16(a.value)};
}

inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::mulSign(ParamType a, ParamType sign)
{
  return Type{_mm256_sign_epi16(a.value, sign.value)};
}

inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::mulFixedPoint(ParamType a, ParamType b)
{
  return Type{_mm256_mulhi_epu16(a.value, b.value)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::extend(ParamType value)
{
  __m256i sign = _mm256_cmpgt_epi16(_mm256_setzero_si256(), value.value);
  return ExtendedType{_mm256_unpacklo_epi16(value.value, sign), _mm256_unpackhi_epi16(value.value, sign)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulExtended(ParamType a, ParamType b)
{
  __m256i ablo = _mm256_mullo_epi16(a.value, b.value);
  __m256i abhi = _mm256_mulhi_epi16(a.value, b.value);
  return ExtendedType{_mm256_unpacklo_epi16(ablo, abhi), _mm256_unpackhi_epi16(ablo, abhi)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulExtended(ParamType a, int16_t factor)
{
  __m256i b = _mm256_set1_epi16(factor);
  __m256i ablo = _mm256_mullo_epi16(a.value, b);
  __m256i abhi = _mm256_mulhi_epi16(a.value, b);
  return ExtendedType{_mm256_unpacklo_epi16(ablo, abhi), _mm256_unpackhi_epi16(ablo, abhi)};
}

inline SIMD<int16_t, 16>::MulAddFactors SIMD<int16_t, 16>::makeMulAddFactors(int16_t afactor, int16_t bfactor)
{
  return _mm256_unpacklo_epi16(_mm256_set1_epi16(afactor), _mm256_set1_epi16(bfactor));
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulAdd(ParamType a, ParamType b, ParamType c, ParamType d)
{
  __m256i aclo = _mm256_unpacklo_epi16(a.value, c.value);
  __m256i bdlo = _mm256_unpacklo_epi16(b.value, d.value);
  __m256i achi = _mm256_unpackhi_epi16(a.value, c.value);
  __m256i bdhi = _mm256_unpackhi_epi16(b.value, d.value);

  return ExtendedType{_mm256_madd_epi16(aclo, bdlo), _mm256_madd_epi16(achi, bdhi)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulAdd(ParamType a, int16_t afactor, ParamType b, int16_t bfactor)
{
  __m256i factor = _mm256_unpacklo_epi16(_mm256_set1_epi16(afactor), _mm256_set1_epi16(bfactor));
  __m256i ablo = _mm256_unpacklo_epi16(a.value, b.value);
  __m256i abhi = _mm256_unpackhi_epi16(a.value, b.value);
  return ExtendedType{_mm256_madd_epi16(ablo, factor), _mm256_madd_epi16(abhi, factor)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulAdd(ParamType a, ParamType b, MulAddFactors factor)
{
  __m256i ablo = _mm256_unpacklo_epi16(a.value, b.value);
  __m256i abhi = _mm256_unpackhi_epi16(a.value, b.value);
  return ExtendedType{_mm256_madd_epi16(ablo, factor), _mm256_madd_epi16(abhi, factor)};
}

template<int16_t aFactorLo, int16_t aFactorHi, int16_t bFactorLo, int16_t bFactorHi>
inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulAdd(ParamType a, ParamType b)
{
  __m256i factor = _mm256_setr_epi16(aFactorLo, bFactorLo, aFactorLo, bFactorLo, aFactorLo, bFactorLo, aFactorLo, bFactorLo,
                                     aFactorHi, bFactorHi, aFactorHi, bFactorHi, aFactorHi, bFactorHi, aFactorHi, bFactorHi);
  return ExtendedType{_mm256_madd_epi16(_mm256_unpacklo_epi16(a.value, b.value), factor), _mm256_madd_epi16(_mm256_unpackhi_epi16(a.value, b.value), factor)};
}

template<int16_t aFactor, int16_t bFactor>
inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::mulAdd(ParamType a, ParamType b)
{
  __m256i factor = _mm256_setr_epi16(aFactor, bFactor, aFactor, bFactor, aFactor, bFactor, aFactor, bFactor,
    aFactor, bFactor, aFactor, bFactor, aFactor, bFactor, aFactor, bFactor);
  return ExtendedType{_mm256_madd_epi16(_mm256_unpacklo_epi16(a.value, b.value), factor), _mm256_madd_epi16(_mm256_unpackhi_epi16(a.value, b.value), factor)};
}

template<int dstStride>
inline void SIMD<int16_t, 16>::transpose2x8x8(Type* dst, ParamType m0, ParamType m1, ParamType m2, ParamType m3, ParamType m4, ParamType m5, ParamType m6, ParamType m7)
{
  __m256i tmp0 = _mm256_unpacklo_epi16(m0.value, m1.value); // 00 10 01 11 02 12 03 13
  __m256i tmp1 = _mm256_unpackhi_epi16(m0.value, m1.value); // 04 14 05 15 06 16 07 17 
  __m256i tmp2 = _mm256_unpacklo_epi16(m2.value, m3.value); // 20 30 21 31 22 32 23 33
  __m256i tmp3 = _mm256_unpackhi_epi16(m2.value, m3.value); // 24 34 25 35 26 36 27 37
  __m256i tmp4 = _mm256_unpacklo_epi16(m4.value, m5.value); // 40 50 41 51 42 52 43 53
  __m256i tmp5 = _mm256_unpackhi_epi16(m4.value, m5.value); // 44 54 45 55 46 56 47 57
  __m256i tmp6 = _mm256_unpacklo_epi16(m6.value, m7.value); // 60 70 61 71 62 72 63 73
  __m256i tmp7 = _mm256_unpackhi_epi16(m6.value, m7.value); // 64 74 65 75 66 76 67 77

  __m256i w0 = _mm256_unpacklo_epi32(tmp0, tmp2); // 00 10 20 30 01 11 21 31
  __m256i w1 = _mm256_unpackhi_epi32(tmp0, tmp2); // 02 12 22 32 03 13 23 33
  __m256i w2 = _mm256_unpacklo_epi32(tmp1, tmp3); // 04 14 24 34 05 15 25 35
  __m256i w3 = _mm256_unpackhi_epi32(tmp1, tmp3); // 06 16 26 36 07 17 27 37
  __m256i w4 = _mm256_unpacklo_epi32(tmp4, tmp6); // 40 50 60 70 41 51 61 71 
  __m256i w5 = _mm256_unpackhi_epi32(tmp4, tmp6); // 42 52 62 72 43 53 63 73
  __m256i w6 = _mm256_unpacklo_epi32(tmp5, tmp7); // 44 54 64 74 45 55 65 75
  __m256i w7 = _mm256_unpackhi_epi32(tmp5, tmp7); // 46 56 66 76 47 57 67 77

  dst[0 * dstStride].value = _mm256_unpacklo_epi64(w0, w4); // 00 10 20 30 40 50 60 70
  dst[1 * dstStride].value = _mm256_unpackhi_epi64(w0, w4); // 01 11 21 31 41 51 61 71 
  dst[2 * dstStride].value = _mm256_unpacklo_epi64(w1, w5); // 02 12 22 32 42 52 62 72
  dst[3 * dstStride].value = _mm256_unpackhi_epi64(w1, w5); // 03 13 23 33 43 53 63 73
  dst[4 * dstStride].value = _mm256_unpacklo_epi64(w2, w6); // 04 14 24 34 44 54 64 74
  dst[5 * dstStride].value = _mm256_unpackhi_epi64(w2, w6); // 05 15 25 35 45 55 65 75
  dst[6 * dstStride].value = _mm256_unpacklo_epi64(w3, w7); // 06 16 26 36 46 56 66 76
  dst[7 * dstStride].value = _mm256_unpackhi_epi64(w3, w7); // 07 17 27 37 47 57 67 77
}

template<bool aligned, int dstStride, int srcStride>
inline void SIMD<int16_t, 16>::transpose2x8x8(Type* dst, const int16_t* src)
{
  transpose2x8x8(dst,
    load<aligned>(src + 16 * 0 * srcStride), load<aligned>(src + 16 * 1 * srcStride),
    load<aligned>(src + 16 * 2 * srcStride), load<aligned>(src + 16 * 3 * srcStride),
    load<aligned>(src + 16 * 4 * srcStride), load<aligned>(src + 16 * 5 * srcStride),
    load<aligned>(src + 16 * 6 * srcStride), load<aligned>(src + 16 * 7 * srcStride));
}

template<int fixedPointBits>
inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::ExtendedType::descale() const
{
  return Type{_mm256_packs_epi32(_mm256_srai_epi32(lo, fixedPointBits), _mm256_srai_epi32(hi, fixedPointBits))};
}

template<int fixedPointBits>
inline SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::ExtendedType::round() const
{
  __m256i rounding = _mm256_set1_epi32(1 << (fixedPointBits - 1));
  return Type{_mm256_packs_epi32(_mm256_srai_epi32(_mm256_add_epi32(lo, rounding), fixedPointBits), _mm256_srai_epi32(_mm256_add_epi32(hi, rounding), fixedPointBits))};
}

inline SIMD<int16_t, 16>::Int32Type SIMD<int16_t, 16>::horizontalMulAdd(Type a, Type b)
{
  return Int32Type{_mm256_madd_epi16(a, b)};
}

inline typename SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::interleaveEach4Low(Type a, Type b)
{
  return Type{_mm256_unpacklo_epi16(a, b)};
}

inline typename SIMD<int16_t, 16>::Type SIMD<int16_t, 16>::interleaveEach4High(Type a, Type b)
{
  return Type{_mm256_unpackhi_epi16(a, b)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::ExtendedType::zero()
{
  return ExtendedType{_mm256_setzero_si256(), _mm256_setzero_si256()};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::ExtendedType::populate(int32_t value)
{
  return ExtendedType{_mm256_set1_epi32(value), _mm256_set1_epi32(value)};
}

inline void SIMD<int16_t, 16>::ExtendedType::clamp(const ExtendedType& min, const ExtendedType& max)
{
  lo = _mm256_max_epi32(_mm256_min_epi32(lo, max.lo), min.lo);
  hi = _mm256_max_epi32(_mm256_min_epi32(hi, max.hi), min.hi);
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::ExtendedType::operator+(const ExtendedType& other) const
{
  return ExtendedType{_mm256_add_epi32(lo, other.lo), _mm256_add_epi32(hi, other.hi)};
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::ExtendedType::operator+=(const ExtendedType& other)
{
  lo = _mm256_add_epi32(lo, other.lo);
  hi = _mm256_add_epi32(hi, other.hi);
  return *this;
}

inline SIMD<int16_t, 16>::ExtendedType SIMD<int16_t, 16>::ExtendedType::operator<<(int shift)
{
  return ExtendedType{_mm256_slli_epi32(lo, shift), _mm256_slli_epi32(hi, shift)};
}

}

}
