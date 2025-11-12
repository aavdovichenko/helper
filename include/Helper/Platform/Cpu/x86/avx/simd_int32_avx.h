#pragma once

#include "simd_int_avx.h"
#include "../sse/simd_sse.h"

#define SIMD_INT8_SUPPORTED // TODO : remove
#define PLATFORM_CPU_FEATURE_INT32x8

#if SIMD_INT_MAX_WIDTH < 8
#undef SIMD_INT_MAX_WIDTH
#define SIMD_INT_MAX_WIDTH 8
#endif

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<int32_t> : public BaseAvxSimdIntType<int32_t, AvxSimdIntType<int32_t>>
{
  using BaseAvxSimdIntType<int32_t, AvxSimdIntType<int32_t>>::BaseAvxSimdIntType;

  AvxSimdIntType<int32_t> operator+(const AvxSimdIntType<int32_t>& other) const;
  AvxSimdIntType<int32_t> operator-(const AvxSimdIntType<int32_t>& other) const;
  AvxSimdIntType<int32_t>& operator+=(const AvxSimdIntType<int32_t>& other);

  AvxSimdIntType<int32_t> operator*(const AvxSimdIntType<int32_t>& other) const;
  AvxSimdIntType<int32_t> operator*(int32_t factor) const;
  friend AvxSimdIntType<int32_t> operator*(int32_t factor, const AvxSimdIntType<int32_t>& value);
  AvxSimdIntType<int32_t>& operator*=(int32_t factor);

  AvxSimdIntType<int32_t> operator>>(int count) const;
  AvxSimdIntType<int32_t> operator<<(int count) const;

  AvxIntSimd<int32_t>::ConditionType operator<(const AvxSimdIntType<int32_t>& other) const;

  static inline AvxSimdIntType<int32_t> fromPackedInt8(uint64_t packed);
  static inline AvxSimdIntType<int32_t> fromPackedUint8(uint64_t packed);
  static inline AvxSimdIntType<int32_t> fromPackedInt16(const SseSimdIntType<int16_t>& packed);
  static inline AvxSimdIntType<int32_t> fromPackedUint16(const SseSimdIntType<uint16_t>& packed);
  static inline AvxSimdIntType<int32_t> fromPackedUint24(const AvxSimdIntType<int32_t>& packed);
  inline void setFromPackedUint8(uint64_t packed);

  inline AvxSimdIntType<int32_t> toPackedUint24() const;

  template<bool aligned> static inline AvxSimdIntType<int32_t> loadAndConvert(const int8_t* p);
  template<bool aligned> static inline AvxSimdIntType<int32_t> loadAndConvert(const uint8_t* p);
  template<bool aligned> static inline AvxSimdIntType<int32_t> loadAndConvert(const int16_t* p);
  template<bool aligned> static inline AvxSimdIntType<int32_t> loadAndConvert(const uint16_t* p);
  template<bool aligned> static inline AvxSimdIntType<int32_t> loadAndConvert(const uint32_t* p);

  template<bool aligned> inline void convertAndStore(int8_t* p) const;
  template<bool aligned> inline void convertAndStore(uint8_t* p) const;
  template<bool aligned> inline void convertAndStore(int16_t* p) const;
  template<bool aligned> inline void convertAndStore(uint16_t* p) const;
  template<bool aligned> inline void convertAndStore(uint32_t* p) const;
};

template<>
struct SIMD<int32_t, 8> : public AvxIntSimd<int32_t>
{
  static bool isSupported(SimdFeatures features = 0);

  static inline Type populate(int32_t value)
  {
    return Type{_mm256_set1_epi32(value)};
  }

  static inline Type rotate(Type value)
  {
    __m256 t0 = _mm256_permute_ps(_mm256_castsi256_ps(value), _MM_SHUFFLE(0, 3, 2, 1));
    __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0x01);
    __m256 y = _mm256_blend_ps(t0, t1, 0x88);
    return Type{_mm256_castps_si256(y)};
  }

  static inline int least(Type value)
  {
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(value));
  }

  static inline Type min(Type a, Type b)
  {
    return Type{_mm256_min_epi32(a, b)};
  }

  static inline Type max(Type a, Type b)
  {
    return Type{_mm256_max_epi32(a, b)};
  }

  static inline Type abs(Type a);
  static inline Type mulSign(Type a, Type sign);

  static inline Type mulExtended(Type a, Type b, Type& abhi)
  {
    __m256i ab0246 = _mm256_mul_epi32(a, b);
    __m256i ab1357 = _mm256_mul_epi32(_mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1)), _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1)));
    abhi = Type{_mm256_blend_epi32(_mm256_shuffle_epi32(ab0246, _MM_SHUFFLE(2, 3, 0, 1)), ab1357, 0xaa)};
    return Type{_mm256_blend_epi32(ab0246, _mm256_shuffle_epi32(ab1357, _MM_SHUFFLE(2, 3, 0, 1)), 0xaa)};
  }

  static inline Type interleaveEach2Low(__m256i a, __m256i b);
  static inline Type interleaveEach2High(__m256i a, __m256i b);

  static inline SIMD<int16_t, 16>::Type interleaveLowHigh16BitSaturated(Type lo, Type hi);

  static inline void transpose(Type& w0, Type& w1, Type& w2, Type& w3, Type& w4, Type& w5, Type& w6, Type& w7);
  template<int dstStride = 1>
  static inline void transpose(Type* dst, ParamType w0, ParamType w1, ParamType w2, ParamType w3, ParamType w4, ParamType w5, ParamType w6, ParamType w7);
  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose(Type* dst, const int32_t* src);
  template<bool dstAligned, bool srcAligned>
  static inline void transpose(int32_t* dst, size_t dstStride, const int32_t* src, size_t srcStride);

  static inline Type interleaveLow16Bit(__m256i a, __m256i b)
  {
    return Type{_mm256_blend_epi16(a, _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(b, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1)), 0xaa)};
  }

  static void extractByteComponents(ParamType a, uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3);
  static void extractByteComponents(ParamType a, ParamType b, SIMD<uint8_t, 16>::Type& c0, SIMD<uint8_t, 16>::Type& c1, SIMD<uint8_t, 16>::Type& c2, SIMD<uint8_t, 16>::Type& c3);
};

// implementation

// AvxSimdIntType<int32_t>

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::operator+(const AvxSimdIntType<int32_t>& other) const
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_add_epi32(value, other.value));
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::operator-(const AvxSimdIntType<int32_t>& other) const
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_sub_epi32(value, other.value));
}

inline AvxSimdIntType<int32_t>& AvxSimdIntType<int32_t>::operator+=(const AvxSimdIntType<int32_t>& other)
{
  value = _mm256_add_epi32(value, other.value);
  return *this;
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::operator*(const AvxSimdIntType<int32_t>& other) const
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_mullo_epi32(value, other.value));
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::operator*(int32_t factor) const
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_mullo_epi32(value, _mm256_set1_epi32(factor)));
}

inline AvxSimdIntType<int32_t> operator*(int32_t factor, const AvxSimdIntType<int32_t>& value)
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_mullo_epi32(value.value, _mm256_set1_epi32(factor)));
}

inline AvxSimdIntType<int32_t>& AvxSimdIntType<int32_t>::operator*=(int32_t factor)
{
  value = _mm256_mullo_epi32(value, _mm256_set1_epi32(factor));
  return *this;
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::operator>>(int count) const
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_srai_epi32(value, count));
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::operator<<(int count) const
{
  return AvxSimdIntType<int32_t>::fromNativeType(_mm256_slli_epi32(value, count));
}

inline AvxIntSimd<int32_t>::ConditionType AvxSimdIntType<int32_t>::operator<(const AvxSimdIntType<int32_t>& other) const
{
  return AvxIntSimd<int32_t>::ConditionType{_mm256_cmpgt_epi32(other.value, value)};
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::fromPackedInt8(uint64_t packed)
{
  return _mm256_cvtepi8_epi32(_mm_set1_epi64x(packed));
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::fromPackedUint8(uint64_t packed)
{
  return _mm256_cvtepu8_epi32(_mm_set1_epi64x(packed));
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::fromPackedInt16(const SseSimdIntType<int16_t>& packed)
{
  return _mm256_cvtepi16_epi32(packed);
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::fromPackedUint16(const SseSimdIntType<uint16_t>& packed)
{
  return _mm256_cvtepu16_epi32(packed);
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::fromPackedUint24(const AvxSimdIntType<int32_t>& packed)
{
  __m256i x = _mm256_permutevar8x32_epi32(packed.value, _mm256_setr_epi32(0, 1, 2, 3, 3, 4, 5, 7));
  return _mm256_shuffle_epi8(x, _mm256_setr_epi8(0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128, 0, 1, 2, -128, 3, 4, 5, -128, 6, 7, 8, -128, 9, 10, 11, -128));
}

inline void AvxSimdIntType<int32_t>::setFromPackedUint8(uint64_t packed)
{
  value = _mm256_cvtepu8_epi32(_mm_set1_epi64x(packed));
}

inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::toPackedUint24() const
{
  __m256i x = _mm256_shuffle_epi8(value, _mm256_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128, -128, -128, -128, 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -128, -128, -128, -128));
  return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7));
}

template<bool aligned>
inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::loadAndConvert(const int8_t* p)
{
  return fromPackedInt8(*(uint64_t*)p);
}

template<bool aligned>
inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::loadAndConvert(const uint8_t* p)
{
  return fromPackedUint8(*(uint64_t*)p);
}

template<bool aligned>
inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::loadAndConvert(const int16_t* p)
{
  return fromPackedInt16(SseSimdIntType<int16_t>::load<aligned>(p));
}

template<bool aligned>
inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::loadAndConvert(const uint16_t* p)
{
  return fromPackedUint16(SseSimdIntType<uint16_t>::load<aligned>(p));
}

template<bool aligned>
inline AvxSimdIntType<int32_t> AvxSimdIntType<int32_t>::loadAndConvert(const uint32_t* p)
{
  return load<aligned>((int32_t*)p);
}

template<bool aligned>
inline void AvxSimdIntType<int32_t>::convertAndStore(int8_t* p) const
{
  __m256i packed = _mm256_shuffle_epi8(value, _mm256_setr_epi8(0, 4, 8, 12, 16, 20, 24, 28, 8, 9, 10, 11, 12, 13, 14, 15, 0, 4, 8, 12, 16, 20, 24, 28, 8, 9, 10, 11, 12, 13, 14, 15));
  *(int64_t*)p = _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 2, 3, 4, 5, 6, 7))));
}

template<bool aligned>
inline void AvxSimdIntType<int32_t>::convertAndStore(uint8_t* p) const
{
  __m256i packed = _mm256_shuffle_epi8(value, _mm256_setr_epi8(0, 4, 8, 12, 16, 20, 24, 28, 8, 9, 10, 11, 12, 13, 14, 15, 0, 4, 8, 12, 16, 20, 24, 28, 8, 9, 10, 11, 12, 13, 14, 15));
  *(uint64_t*)p = _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 2, 3, 4, 5, 6, 7))));
}

template<bool aligned>
inline void AvxSimdIntType<int32_t>::convertAndStore(int16_t* p) const
{
  SseSimdIntType<int16_t>::fromNativeType(_mm256_castsi256_si128(_mm256_packs_epi32(value, _mm256_permute2x128_si256(value, value, 1)))).store<aligned>(p);
}

template<bool aligned>
inline void AvxSimdIntType<int32_t>::convertAndStore(uint16_t* p) const
{
  SseSimdIntType<uint16_t>::fromNativeType(_mm256_castsi256_si128(_mm256_packus_epi32(value, _mm256_permute2x128_si256(value, value, 1)))).store<aligned>(p);
}

template<bool aligned>
inline void AvxSimdIntType<int32_t>::convertAndStore(uint32_t* p) const
{
  store<aligned>((int32_t*)p);
}

// SIMD<int32_t, 8>

inline bool SIMD<int32_t, 8>::isSupported(SimdFeatures features)
{
  static bool avx2Enabled = isAVX2Enabled();
  if ((features & SimdFeature::Multiplication) && !avx2Enabled)
    return false;

  return AvxIntSimd<int32_t>::isSupported(features);
}

inline SIMD<int32_t, 8>::Type SIMD<int32_t, 8>::abs(Type a)
{
  return Type{_mm256_abs_epi32(a)};
}

inline SIMD<int32_t, 8>::Type SIMD<int32_t, 8>::mulSign(Type a, Type sign)
{
  return Type{_mm256_sign_epi32(a, sign)};
}

inline SIMD<int32_t, 8>::Type SIMD<int32_t, 8>::interleaveEach2Low(__m256i a, __m256i b)
{
  return Type{_mm256_unpacklo_epi32(a, b)};
}

inline SIMD<int32_t, 8>::Type SIMD<int32_t, 8>::interleaveEach2High(__m256i a, __m256i b)
{
  return Type{_mm256_unpackhi_epi32(a, b)};
}

inline SIMD<int16_t, 16>::Type SIMD<int32_t, 8>::interleaveLowHigh16BitSaturated(Type lo, Type hi)
{
  return SIMD<int16_t, 16>::Type{_mm256_packs_epi32(lo, hi)};
}

inline void SIMD<int32_t, 8>::transpose(Type& w0, Type& w1, Type& w2, Type& w3, Type& w4, Type& w5, Type& w6, Type& w7)
{
  transposeAvxInt32(w0.value, w1.value, w2.value, w3.value, w4.value, w5.value, w6.value, w7.value);
}

template<int dstStride>
inline void SIMD<int32_t, 8>::transpose(Type* dst, ParamType m0, ParamType m1, ParamType m2, ParamType m3, ParamType m4, ParamType m5, ParamType m6, ParamType m7)
{
  transposeAvxInt32(m0.value, m1.value, m2.value, m3.value, m4.value, m5.value, m6.value, m7.value);

  dst[0 * dstStride].value = m0.value;
  dst[1 * dstStride].value = m1.value;
  dst[2 * dstStride].value = m2.value;
  dst[3 * dstStride].value = m3.value;
  dst[4 * dstStride].value = m4.value;
  dst[5 * dstStride].value = m5.value;
  dst[6 * dstStride].value = m6.value;
  dst[7 * dstStride].value = m7.value;
}

template<bool aligned, int dstStride, int srcStride>
inline void SIMD<int32_t, 8>::transpose(Type* dst, const int32_t* src)
{
#if 1
  transpose<dstStride>(dst,
    load<aligned>(src + 8 * 0 * srcStride), load<aligned>(src + 8 * 1 * srcStride),
    load<aligned>(src + 8 * 2 * srcStride), load<aligned>(src + 8 * 3 * srcStride),
    load<aligned>(src + 8 * 4 * srcStride), load<aligned>(src + 8 * 5 * srcStride),
    load<aligned>(src + 8 * 6 * srcStride), load<aligned>(src + 8 * 7 * srcStride));
#else
  typedef typename SIMD<int, 4>::Type Type4;
  SIMD<int, 4>::transpose<aligned, dstStride * 2, srcStride * 2>((Type4*)dst, src);
  SIMD<int, 4>::transpose<aligned, dstStride * 2, srcStride * 2>((Type4*)dst + 1, src + 4 * 8 * srcStride);
  SIMD<int, 4>::transpose<aligned, dstStride * 2, srcStride * 2>((Type4*)dst + 4 * dstStride * 2, src + 4);
  SIMD<int, 4>::transpose<aligned, dstStride * 2, srcStride * 2>((Type4*)dst + 4 * dstStride * 2 + 1, src + 4 * 8 * srcStride + 4);
#endif
}

template<bool dstAligned, bool srcAligned>
inline void SIMD<int32_t, 8>::transpose(int32_t* dst, size_t dstStride, const int32_t* src, size_t srcStride)
{
  Type w0 = load<srcAligned>(src + 0 * srcStride), w1 = load<srcAligned>(src + 1 * srcStride);
  Type w2 = load<srcAligned>(src + 2 * srcStride), w3 = load<srcAligned>(src + 3 * srcStride);
  Type w4 = load<srcAligned>(src + 4 * srcStride), w5 = load<srcAligned>(src + 5 * srcStride);
  Type w6 = load<srcAligned>(src + 6 * srcStride), w7 = load<srcAligned>(src + 7 * srcStride);

  transposeAvxInt32(w0.value, w1.value, w2.value, w3.value, w4.value, w5.value, w6.value, w7.value);

  w0.store<dstAligned>(dst + 0 * dstStride); w1.store<dstAligned>(dst + 1 * dstStride);
  w2.store<dstAligned>(dst + 2 * dstStride); w3.store<dstAligned>(dst + 3 * dstStride);
  w4.store<dstAligned>(dst + 4 * dstStride); w5.store<dstAligned>(dst + 5 * dstStride);
  w6.store<dstAligned>(dst + 6 * dstStride); w7.store<dstAligned>(dst + 7 * dstStride);
}

inline void SIMD<int32_t, 8>::extractByteComponents(ParamType a, uint64_t& c0, uint64_t& c1, uint64_t& c2, uint64_t& c3)
{
  return SIMD<int32_t, 4>::extractByteComponents(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1), c0, c1, c2, c3);
}

inline void SIMD<int32_t, 8>::extractByteComponents(ParamType w0, ParamType w1, SIMD<uint8_t, 16>::Type& c0, SIMD<uint8_t, 16>::Type& c1, SIMD<uint8_t, 16>::Type& c2, SIMD<uint8_t, 16>::Type& c3)
{
  // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33 | 04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37
  // 08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B | 0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F

  __m256i c = _mm256_unpacklo_epi8(w0.value, w1.value); // 00 08 10 18 20 28 30 38 01 09 11 19 21 29 31 39 | 04 0C 14 1C 24 2C 34 3C 05 0D 15 1D 25 2D 35 3D
  __m256i d = _mm256_unpackhi_epi8(w0.value, w1.value); // 02 0A 12 1A 22 2A 32 3A 03 0B 13 1B 23 2B 33 3B | 06 0E 16 1E 26 2E 36 3E 07 0F 17 1F 27 2F 37 3F

  __m256i a = _mm256_unpacklo_epi8(c, d); // 00 02 08 0A 10 12 18 1A 20 22 28 2A 30 32 38 3A | 04 06 0C 0E 14 16 1C 1E 24 26 2C 2E 34 36 3C 3E
  __m256i b = _mm256_unpackhi_epi8(c, d); // 01 03 09 0B 11 13 19 1B 21 23 29 2B 31 33 39 3B | 05 07 0D 0F 15 17 1D 1F 25 27 2D 2F 35 37 3D 3F

  c = _mm256_unpacklo_epi8(a, b); // 00 01 02 03 08 09 0A 0B 10 11 12 13 18 19 1A 1B | 04 05 06 07 0C 0D 0E 0F 14 15 16 17 1C 1D 1E 1F
  d = _mm256_unpackhi_epi8(a, b); // 20 21 22 23 28 29 2A 2B 30 31 32 33 38 39 3A 3B | 24 25 26 27 2C 2D 2E 2F 34 35 36 37 3C 3D 3E 3F

  a = _mm256_permute2x128_si256(c, d, 0x20); // 00 01 02 03 08 09 0A 0B 10 11 12 13 18 19 1A 1B | 20 21 22 23 28 29 2A 2B 30 31 32 33 38 39 3A 3B
  b = _mm256_permute2x128_si256(c, d, 0x31); // 04 05 06 07 0C 0D 0E 0F 14 15 16 17 1C 1D 1E 1F | 24 25 26 27 2C 2D 2E 2F 34 35 36 37 3C 3D 3E 3F

  c = _mm256_unpacklo_epi32(a, b); // 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F | 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F
  d = _mm256_unpackhi_epi32(a, b); // 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F | 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F

  c0.value = _mm256_castsi256_si128(c);
  c1.value = _mm256_castsi256_si128(d);
  c2.value = _mm256_extractf128_si256(c, 1);
  c3.value = _mm256_extractf128_si256(d, 1);
}

} // namespace Cpu

} // namespace Platform
