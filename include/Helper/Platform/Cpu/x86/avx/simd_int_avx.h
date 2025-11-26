#pragma once

#include <immintrin.h>

#include "../../simd_condition.h"
#include "../../simd_int.h"
#include "../sse/simd_int_sse.h"
#include "../simd_x86.h"

#if defined(__GNUC__) && __GNUC__ < 8
#define _mm256_setr_m128i(lo, hi) (_mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1))
#endif

namespace Platform
{

namespace Cpu
{

static inline bool isAVXEnabled();
static inline bool isAVX2Enabled();

template <typename T> struct AvxSimdIntType;

template <typename T>
struct AvxSimdIntConditionType : public SimdConditionType<T, __m256i, AvxSimdIntConditionType<T>>
{
  inline AvxSimdIntConditionType() {}
  inline AvxSimdIntConditionType(__m256i value) : SimdConditionType<T, __m256i, AvxSimdIntConditionType<T>>{value} {}

  inline AvxSimdIntType<T> mask() const;
  inline int64_t bitMask() const;
};

template <typename T, typename Implementation>
struct BaseAvxSimdIntType : public SimdIntType<T, __m256i, Implementation>
{
  inline BaseAvxSimdIntType() {}
  inline BaseAvxSimdIntType(__m256i&& value) : SimdIntType<T, __m256i, Implementation>{value} {}
  inline BaseAvxSimdIntType(const __m256i& value) : SimdIntType<T, __m256i, Implementation>{value} {}

  static inline Implementation zero();

  template<bool aligned> static inline Implementation load(const T* src);
  static inline Implementation load(const T* src);
  template<bool aligned> inline void store(T* dst) const;
  static inline void store(T* dst, const Implementation& value);
  inline void store(T* dst) const;

  inline SseSimdIntType<T> lowPart() const;
  inline SseSimdIntType<T> highPart() const;

  inline Implementation operator&(const Implementation& other) const;
  inline Implementation operator|(const Implementation& other) const;
  inline Implementation& operator|=(const Implementation& other);
  inline Implementation operator^(const Implementation& other) const;
  inline Implementation andNot(const Implementation& other) const;

  static inline Implementation select(const AvxSimdIntConditionType<T>& condition, const Implementation& a, const Implementation& b);
};

template <typename T>
struct AvxIntSimd : public x86Simd<32>, public IntSimd<T, __m256i, __m256i>
{
  typedef AvxSimdIntType<T> Type;
  typedef AvxSimdIntConditionType<T> ConditionType;
#if defined(DEBUG) || defined(_DEBUG)
  typedef const Type& ParamType;
  typedef const ConditionType& ConditionParamType;
#else
  typedef Type ParamType;
  typedef ConditionType ConditionParamType;
#endif

  static bool isSupported(SimdFeatures features = 0);

  static AvxSimdIntType<T> zero();

  template<bool aligned> static inline Type load(const T* src);
  static inline Type load(const T* src);
  template<bool aligned> static inline void store(T* dst, ParamType value);
  static inline void store(T* dst, ParamType value);
  static inline void storeLow(T* dst, ParamType value);
  static inline void storeHigh(T* dst, ParamType value);

  static inline Type select(ConditionType condition, Type a, Type b);

  constexpr static int PreferedAlignment = 32;
};

// implementation

// AvxSimdIntType<T>

template<typename T>
inline AvxSimdIntType<T> AvxSimdIntConditionType<T>::mask() const
{
  return AvxSimdIntType<T>{this->value};
}

template<typename T>
inline int64_t AvxSimdIntConditionType<T>::bitMask() const
{
  return _mm256_movemask_epi8(this->value);
}

template<typename T, typename Implementation> template<bool aligned>
inline Implementation BaseAvxSimdIntType<T, Implementation>::load(const T* src)
{
  return Implementation::fromNativeType(aligned ? _mm256_load_si256((const __m256i*)src) : _mm256_loadu_si256((const __m256i*)src));
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::zero()
{
  return Implementation::fromNativeType(_mm256_setzero_si256());
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::load(const T* src)
{
  return Implementation::fromNativeType(_mm256_load_si256((const __m256i*)src));
}

template<typename T, typename Implementation> template<bool aligned>
inline void BaseAvxSimdIntType<T, Implementation>::store(T* dst) const
{
  aligned ? _mm256_store_si256((__m256i*)dst, this->value) : _mm256_storeu_si256((__m256i*)dst, this->value);
}

template<typename T, typename Implementation>
inline void BaseAvxSimdIntType<T, Implementation>::store(T* dst, const Implementation& value)
{
  _mm256_store_si256((__m256i*)dst, value.value);
}

template<typename T, typename Implementation>
inline void BaseAvxSimdIntType<T, Implementation>::store(T* dst) const
{
  _mm256_store_si256((__m256i*)dst, this->value);
}

template<typename T, typename Implementation>
inline SseSimdIntType<T> BaseAvxSimdIntType<T, Implementation>::lowPart() const
{
  return SseSimdIntType<T>{_mm256_castsi256_si128(*this)};
}

template<typename T, typename Implementation>
inline SseSimdIntType<T> BaseAvxSimdIntType<T, Implementation>::highPart() const
{
  return SseSimdIntType<T>{_mm256_extractf128_si256(*this, 1)};
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::operator&(const Implementation& other) const
{
//  return Implementation::fromNativeType(_mm256_and_si256(this->value, other.value));
  return Implementation::fromNativeType(_mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(this->value), _mm256_castsi256_ps(other.value))));
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::operator|(const Implementation& other) const
{
//  return Implementation::fromNativeType(_mm256_or_si256(this->value, other.value));
  return Implementation::fromNativeType(_mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(this->value), _mm256_castsi256_ps(other.value))));
}

template<typename T, typename Implementation>
inline Implementation& BaseAvxSimdIntType<T, Implementation>::operator|=(const Implementation& other)
{
  this->value = _mm256_or_si256(this->value, other.value);
  return *(Implementation*)this;
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::operator^(const Implementation& other) const
{
  return Implementation::fromNativeType(_mm256_xor_si256(this->value, other.value));
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::andNot(const Implementation& other) const
{
  return Implementation::fromNativeType(_mm256_andnot_si256(other.value, this->value));
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::select(const AvxSimdIntConditionType<T>& condition, const Implementation& a, const Implementation& b)
{
  return Implementation::fromNativeType(_mm256_or_si256(_mm256_and_si256(condition, a), _mm256_andnot_si256(condition, b)));
}

// AvxIntSimd<T>

template<typename T>
inline bool AvxIntSimd<T>::isSupported(SimdFeatures)
{
  static bool avxEnabled = isAVXEnabled();
  static bool avx2Enabled = isAVX2Enabled();

  return avxEnabled && avx2Enabled;
}

template<typename T>
inline AvxSimdIntType<T> AvxIntSimd<T>::zero()
{
  return AvxSimdIntType<T>::fromNativeType(_mm256_setzero_si256());
}

template<typename T> template<bool aligned>
inline typename AvxIntSimd<T>::Type AvxIntSimd<T>::load(const T* src)
{
  return Type{aligned ? _mm256_load_si256((const __m256i*)src) : _mm256_loadu_si256((const __m256i*)src)};
}

template<typename T>
inline typename AvxIntSimd<T>::Type AvxIntSimd<T>::load(const T* src)
{
  return Type{_mm256_load_si256((const __m256i*)src)};
}

template<typename T> template<bool aligned>
inline void AvxIntSimd<T>::store(T* dst, ParamType value)
{
  aligned ? _mm256_store_si256((__m256i*)dst, value.value) : _mm256_storeu_si256((__m256i*)dst, value.value);
}

template<typename T>
inline void AvxIntSimd<T>::store(T* dst, ParamType value)
{
  _mm256_store_si256((__m256i*)dst, value.value);
}

template<typename T>
inline void AvxIntSimd<T>::storeLow(T* dst, ParamType value)
{
  _mm_store_si128((__m128i*)dst, _mm256_castsi256_si128(value.value));
}

template<typename T>
inline void AvxIntSimd<T>::storeHigh(T* dst, ParamType value)
{
  _mm_store_si128((__m128i*)dst, _mm256_extractf128_si256(value.value, 1));
}

template<typename T>
inline typename AvxIntSimd<T>::Type AvxIntSimd<T>::select(ConditionType condition, Type a, Type b)
{
  return _mm256_or_si256(_mm256_and_si256(condition, a), _mm256_andnot_si256(condition, b));
}

static inline void transposeAvxInt128(__m256i& w0, __m256i& w1)
{
  __m256i tmp = _mm256_permute2x128_si256(w0, w1, 0x20);
  w1 = _mm256_permute2x128_si256(w0, w1, 0x31);
  w0 = tmp;
}

static inline void transposeAvx2x4x4Int32(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3)
{
  // 00 01 02 03
  // 10 11 12 13
  // 20 21 22 23
  // 30 31 32 33

  __m256i x0 = _mm256_unpacklo_epi32(w0, w1); // 00 10 01 11 
  __m256i x1 = _mm256_unpackhi_epi32(w0, w1); // 02 12 03 13 
  __m256i x2 = _mm256_unpacklo_epi32(w2, w3); // 20 30 21 31 
  __m256i x3 = _mm256_unpackhi_epi32(w2, w3); // 22 32 23 33 

  w0 = _mm256_unpacklo_epi64(x0, x2); // 00 10 20 30
  w1 = _mm256_unpackhi_epi64(x0, x2); // 01 11 21 31
  w2 = _mm256_unpacklo_epi64(x1, x3); // 02 12 22 32
  w3 = _mm256_unpackhi_epi64(x1, x3); // 03 13 23 33
}

static inline void transposeAvxInt32(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7)
{
  transposeAvxInt128(w0, w4);
  transposeAvxInt128(w1, w5);
  transposeAvxInt128(w2, w6);
  transposeAvxInt128(w3, w7);

  transposeAvx2x4x4Int32(w0, w1, w2, w3);
  transposeAvx2x4x4Int32(w4, w5, w6, w7);
}

static inline void transposeAvx2x8x8Int16(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7)
{
  __m256i x0 = _mm256_unpacklo_epi16(w0, w1); // 00 10 01 11 02 12 03 13
  __m256i x1 = _mm256_unpackhi_epi16(w0, w1); // 04 14 05 15 06 16 07 17 
  __m256i x2 = _mm256_unpacklo_epi16(w2, w3); // 20 30 21 31 22 32 23 33
  __m256i x3 = _mm256_unpackhi_epi16(w2, w3); // 24 34 25 35 26 36 27 37
  __m256i x4 = _mm256_unpacklo_epi16(w4, w5); // 40 50 41 51 42 52 43 53
  __m256i x5 = _mm256_unpackhi_epi16(w4, w5); // 44 54 45 55 46 56 47 57
  __m256i x6 = _mm256_unpacklo_epi16(w6, w7); // 60 70 61 71 62 72 63 73
  __m256i x7 = _mm256_unpackhi_epi16(w6, w7); // 64 74 65 75 66 76 67 77

  __m256i y0 = _mm256_unpacklo_epi32(x0, x2); // 00 10 20 30 01 11 21 31
  __m256i y1 = _mm256_unpackhi_epi32(x0, x2); // 02 12 22 32 03 13 23 33
  __m256i y2 = _mm256_unpacklo_epi32(x1, x3); // 04 14 24 34 05 15 25 35
  __m256i y3 = _mm256_unpackhi_epi32(x1, x3); // 06 16 26 36 07 17 27 37
  __m256i y4 = _mm256_unpacklo_epi32(x4, x6); // 40 50 60 70 41 51 61 71 
  __m256i y5 = _mm256_unpackhi_epi32(x4, x6); // 42 52 62 72 43 53 63 73
  __m256i y6 = _mm256_unpacklo_epi32(x5, x7); // 44 54 64 74 45 55 65 75
  __m256i y7 = _mm256_unpackhi_epi32(x5, x7); // 46 56 66 76 47 57 67 77

  w0 = _mm256_unpacklo_epi64(y0, y4); // 00 10 20 30 40 50 60 70
  w1 = _mm256_unpackhi_epi64(y0, y4); // 01 11 21 31 41 51 61 71 
  w2 = _mm256_unpacklo_epi64(y1, y5); // 02 12 22 32 42 52 62 72
  w3 = _mm256_unpackhi_epi64(y1, y5); // 03 13 23 33 43 53 63 73
  w4 = _mm256_unpacklo_epi64(y2, y6); // 04 14 24 34 44 54 64 74
  w5 = _mm256_unpackhi_epi64(y2, y6); // 05 15 25 35 45 55 65 75
  w6 = _mm256_unpacklo_epi64(y3, y7); // 06 16 26 36 46 56 66 76
  w7 = _mm256_unpackhi_epi64(y3, y7); // 07 17 27 37 47 57 67 77
}

static inline void transposeAvxInt16(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7,
  __m256i& w8, __m256i& w9, __m256i& wA, __m256i& wB, __m256i& wC, __m256i& wD, __m256i& wE, __m256i& wF)
{
  transposeAvxInt128(w0, w8);
  transposeAvxInt128(w1, w9);
  transposeAvxInt128(w2, wA);
  transposeAvxInt128(w3, wB);
  transposeAvxInt128(w4, wC);
  transposeAvxInt128(w5, wD);
  transposeAvxInt128(w6, wE);
  transposeAvxInt128(w7, wF);

  transposeAvx2x8x8Int16(w0, w1, w2, w3, w4, w5, w6, w7);
  transposeAvx2x8x8Int16(w8, w9, wA, wB, wC, wD, wE, wF);
}

static inline void transposeAvx2x16x16Int8(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7,
  __m256i& w8, __m256i& w9, __m256i& wA, __m256i& wB, __m256i& wC, __m256i& wD, __m256i& wE, __m256i& wF)
{
  __m256i x0 = _mm256_unpacklo_epi8(w0, w1); // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
  __m256i x1 = _mm256_unpackhi_epi8(w0, w1); // 08 18 09 19 0A 1A 0B 1B 0C 1C 0D 1D 0E 1E 0F 1F
  __m256i x2 = _mm256_unpacklo_epi8(w2, w3); // 20 30 21 31 22 32 23 33 24 34 25 35 26 36 27 37
  __m256i x3 = _mm256_unpackhi_epi8(w2, w3); // 28 38 29 39 2A 3A 2B 3B 2C 3C 2D 3D 2E 3E 2F 3F
  __m256i x4 = _mm256_unpacklo_epi8(w4, w5); // 40 50 41 51 42 52 43 53 44 54 45 55 46 56 47 57
  __m256i x5 = _mm256_unpackhi_epi8(w4, w5); // 48 58 49 59 4A 5A 4B 5B 4C 5C 4D 5D 4E 5E 4F 5F
  __m256i x6 = _mm256_unpacklo_epi8(w6, w7); // 60 70 61 71 62 72 63 73 64 74 65 75 66 76 67 77
  __m256i x7 = _mm256_unpackhi_epi8(w6, w7); // 68 78 69 79 6A 7A 6B 7B 6C 7C 6D 7D 6E 7E 6F 7F
  __m256i x8 = _mm256_unpacklo_epi8(w8, w9); // 80 90 81 91 82 92 83 93 84 94 85 95 86 96 87 97
  __m256i x9 = _mm256_unpackhi_epi8(w8, w9); // 88 98 89 99 8A 9A 8B 9B 8C 9C 8D 9D 8E 9E 8F 9F
  __m256i xA = _mm256_unpacklo_epi8(wA, wB); // A0 B0 A1 B1 A2 B2 A3 B3 A4 B4 A5 B5 A6 B6 A7 B7
  __m256i xB = _mm256_unpackhi_epi8(wA, wB); // A8 B8 A9 B9 AA BA AB BB AC BC AD BD AE BE AF BF
  __m256i xC = _mm256_unpacklo_epi8(wC, wD); // C0 D0 C1 D1 C2 D2 C3 D3 C4 D4 C5 D5 C6 D6 C7 D7
  __m256i xD = _mm256_unpackhi_epi8(wC, wD); // C8 D8 C9 D9 CA DA CB DB CC DC CD DD CE DE CF DF
  __m256i xE = _mm256_unpacklo_epi8(wE, wF); // E0 F0 E1 F1 E2 F2 E3 F3 E4 F4 E5 F5 E6 F6 E7 F7
  __m256i xF = _mm256_unpackhi_epi8(wE, wF); // E8 F8 E9 F9 EA FA EB FB EC FC ED FD EE FE EF FF

  __m256i y0 = _mm256_unpacklo_epi16(x0, x2); // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
  __m256i y1 = _mm256_unpackhi_epi16(x0, x2); // 04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37
  __m256i y2 = _mm256_unpacklo_epi16(x1, x3); // 08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B
  __m256i y3 = _mm256_unpackhi_epi16(x1, x3); // 0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F
  __m256i y4 = _mm256_unpacklo_epi16(x4, x6); // 40 50 60 70 41 51 61 71 42 52 62 72 43 53 63 73
  __m256i y5 = _mm256_unpackhi_epi16(x4, x6); // 44 54 64 74 45 55 65 75 46 56 66 76 47 57 67 77
  __m256i y6 = _mm256_unpacklo_epi16(x5, x7); // 48 58 68 78 49 59 69 79 4A 5A 6A 7A 4B 5B 6B 7B
  __m256i y7 = _mm256_unpackhi_epi16(x5, x7); // 4C 5C 6C 7C 4D 5D 6D 7D 4E 5E 6E 7E 4F 5F 6F 7F
  __m256i y8 = _mm256_unpacklo_epi16(x8, xA); // 80 90 A0 B0 81 91 A1 B1 82 92 A2 B2 83 93 A3 B3
  __m256i y9 = _mm256_unpackhi_epi16(x8, xA); // 84 94 A4 B4 85 95 A5 B5 86 96 A6 B6 87 97 A7 B7
  __m256i yA = _mm256_unpacklo_epi16(x9, xB); // 88 98 A8 B8 89 99 A9 B9 8A 9A AA BA 8B 9B AB BB
  __m256i yB = _mm256_unpackhi_epi16(x9, xB); // 8C 9C AC BC 8D 9D AD BD 8E 9E AE BE 8F 9F AF BF
  __m256i yC = _mm256_unpacklo_epi16(xC, xE); // C0 D0 E0 F0 C1 D1 E1 F1 C2 D2 E2 F2 C3 D3 E3 F3
  __m256i yD = _mm256_unpackhi_epi16(xC, xE); // C4 D4 E4 F4 C5 D5 E5 F5 C6 D6 E6 F6 C7 D7 E7 F7
  __m256i yE = _mm256_unpacklo_epi16(xD, xF); // C8 D8 E8 F8 C9 D9 E9 F9 CA DA EA FA CB DB EB FB
  __m256i yF = _mm256_unpackhi_epi16(xD, xF); // CC DC EC FC CD DD ED FD CE DE EE FE CF DF EF FF

  x0 = _mm256_unpacklo_epi32(y0, y4); // 00 10 20 30 40 50 60 70 01 11 21 31 41 51 61 71
  x1 = _mm256_unpackhi_epi32(y0, y4); // 02 12 22 32 42 52 62 72 03 13 23 33 43 53 63 73
  x2 = _mm256_unpacklo_epi32(y1, y5); // 04 14 24 34 44 54 64 74 05 15 25 35 45 55 65 75
  x3 = _mm256_unpackhi_epi32(y1, y5); // 06 16 26 36 46 56 66 76 07 17 27 37 47 57 67 77
  x4 = _mm256_unpacklo_epi32(y2, y6); // 08 18 28 38 48 58 68 78 09 19 29 39 49 59 69 79
  x5 = _mm256_unpackhi_epi32(y2, y6); // 0A 1A 2A 3A 4A 5A 6A 7A 0B 1B 2B 3B 4B 5B 6B 7B
  x6 = _mm256_unpacklo_epi32(y3, y7); // 0C 1C 2C 3C 4C 5C 6C 7C 0D 1D 2D 3D 4D 5D 6D 7D
  x7 = _mm256_unpackhi_epi32(y3, y7); // 0E 1E 2E 3E 4E 5E 6E 7E 0F 1F 2F 3F 4F 5F 6F 7F
  x8 = _mm256_unpacklo_epi32(y8, yC); // 80 90 A0 B0 C0 D0 E0 F0 81 91 A1 B1 C1 D1 E1 F1
  x9 = _mm256_unpackhi_epi32(y8, yC); // 82 92 A2 B2 C2 D2 E2 F2 83 93 A3 B3 C3 D3 E3 F3
  xA = _mm256_unpacklo_epi32(y9, yD); // 84 94 A4 B4 C4 D4 E4 F4 85 95 A5 B5 C5 D5 E5 F5
  xB = _mm256_unpackhi_epi32(y9, yD); // 86 96 A6 B6 C6 D6 E6 F6 87 97 A7 B7 C7 D7 E7 F7
  xC = _mm256_unpacklo_epi32(yA, yE); // 88 98 A8 B8 C8 D8 E8 F8 89 99 A9 B9 C9 D9 E9 F9
  xD = _mm256_unpackhi_epi32(yA, yE); // 8A 9A AA BA CA DA EA FA 8B 9B AB BB CB DB EB FB
  xE = _mm256_unpacklo_epi32(yB, yF); // 8C 9C AC BC CC DC EC FC 8D 9D AD BD CD DD ED FD
  xF = _mm256_unpackhi_epi32(yB, yF); // 8E 9E AE BE CE DE EE FE 8F 9F AF BF CF DF EF FF

  w0 = _mm256_unpacklo_epi64(x0, x8); // 00 10 20 30 40 50 60 70 80 90 A0 B0 C0 D0 E0 F0
  w1 = _mm256_unpackhi_epi64(x0, x8); // 01 11 21 31 41 51 61 71 81 91 A1 B1 C1 D1 E1 F1
  w2 = _mm256_unpacklo_epi64(x1, x9); // 02 12 22 32 42 52 62 72 82 92 A2 B2 C2 D2 E2 F2
  w3 = _mm256_unpackhi_epi64(x1, x9); // 03 13 23 33 43 53 63 73 83 93 A3 B3 C3 D3 E3 F3
  w4 = _mm256_unpacklo_epi64(x2, xA); // ...
  w5 = _mm256_unpackhi_epi64(x2, xA);
  w6 = _mm256_unpacklo_epi64(x3, xB);
  w7 = _mm256_unpackhi_epi64(x3, xB);
  w8 = _mm256_unpacklo_epi64(x4, xC);
  w9 = _mm256_unpackhi_epi64(x4, xC);
  wA = _mm256_unpacklo_epi64(x5, xD);
  wB = _mm256_unpackhi_epi64(x5, xD);
  wC = _mm256_unpacklo_epi64(x6, xE);
  wD = _mm256_unpackhi_epi64(x6, xE);
  wE = _mm256_unpacklo_epi64(x7, xF);
  wF = _mm256_unpackhi_epi64(x7, xF);
}

static inline void transposeAvxInt8(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7,
  __m256i& w8, __m256i& w9, __m256i& wA, __m256i& wB, __m256i& wC, __m256i& wD, __m256i& wE, __m256i& wF,
  __m256i& wG, __m256i& wH, __m256i& wI, __m256i& wJ, __m256i& wK, __m256i& wL, __m256i& wM, __m256i& wN,
  __m256i& wO, __m256i& wP, __m256i& wQ, __m256i& wR, __m256i& wS, __m256i& wT, __m256i& wU, __m256i& wV)
{
  transposeAvxInt128(w0, wG);
  transposeAvxInt128(w1, wH);
  transposeAvxInt128(w2, wI);
  transposeAvxInt128(w3, wJ);
  transposeAvxInt128(w4, wK);
  transposeAvxInt128(w5, wL);
  transposeAvxInt128(w6, wM);
  transposeAvxInt128(w7, wN);
  transposeAvxInt128(w8, wO);
  transposeAvxInt128(w9, wP);
  transposeAvxInt128(wA, wQ);
  transposeAvxInt128(wB, wR);
  transposeAvxInt128(wC, wS);
  transposeAvxInt128(wD, wT);
  transposeAvxInt128(wE, wU);
  transposeAvxInt128(wF, wV);

  transposeAvx2x16x16Int8(w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, wA, wB, wC, wD, wE, wF);
  transposeAvx2x16x16Int8(wG, wH, wI, wJ, wK, wL, wM, wN, wO, wP, wQ, wR, wS, wT, wU, wV);
}

}

}
