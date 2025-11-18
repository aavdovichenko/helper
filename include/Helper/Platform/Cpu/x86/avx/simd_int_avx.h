#pragma once

#include <immintrin.h>

#include "../../simd_condition.h"
#include "../../simd_int.h"
#include "../sse/simd_int_sse.h"
#include "../simd_x86.h"

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
  inline BaseAvxSimdIntType(__m256i value) : SimdIntType<T, __m256i, Implementation>{value} {}

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
  typedef T ItemType;
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
  return Type::fromNativeType(aligned ? _mm256_load_si256((const __m256i*)src) : _mm256_loadu_si256((const __m256i*)src));
}

template<typename T>
inline typename AvxIntSimd<T>::Type AvxIntSimd<T>::load(const T* src)
{
  return Type::fromNativeType(_mm256_load_si256((const __m256i*)src));
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

static inline void transposeAvxInt32(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7)
{
  // 00 01 02 03 04 05 06 07
  // 10 11 12 13 14 15 16 17
  // 20 21 22 23 24 25 26 27
  // 30 31 32 33 34 35 36 37
  // 40 41 42 43 44 45 46 47
  // 50 51 52 53 54 55 56 57
  // 60 61 62 63 64 65 66 67
  // 70 71 72 73 74 75 76 77

  __m256i a0 = _mm256_permute2x128_si256(w0, w4, 0x20); // 00 01 02 03 40 41 42 43
  __m256i a1 = _mm256_permute2x128_si256(w1, w5, 0x20); // 10 11 12 13 50 51 52 53
  __m256i a2 = _mm256_permute2x128_si256(w2, w6, 0x20); // 20 21 22 23 60 61 62 63
  __m256i a3 = _mm256_permute2x128_si256(w3, w7, 0x20); // 30 31 32 33 70 71 72 73
  __m256i a4 = _mm256_permute2x128_si256(w0, w4, 0x31); // 04 05 06 07 44 45 46 47
  __m256i a5 = _mm256_permute2x128_si256(w1, w5, 0x31); // 14 15 16 17 54 55 56 57
  __m256i a6 = _mm256_permute2x128_si256(w2, w6, 0x31); // 24 25 26 27 64 65 66 67
  __m256i a7 = _mm256_permute2x128_si256(w3, w7, 0x31); // 34 35 36 37 74 75 76 77

  __m256i b0 = _mm256_unpacklo_epi32(a0, a1); // 00 10 01 11 
  __m256i b1 = _mm256_unpackhi_epi32(a0, a1); // 02 12 03 13 
  __m256i b2 = _mm256_unpacklo_epi32(a2, a3); // 20 30 21 31 
  __m256i b3 = _mm256_unpackhi_epi32(a2, a3); // 22 32 23 33 
  __m256i b4 = _mm256_unpacklo_epi32(a4, a5); // 
  __m256i b5 = _mm256_unpackhi_epi32(a4, a5); // 
  __m256i b6 = _mm256_unpacklo_epi32(a6, a7); // 
  __m256i b7 = _mm256_unpackhi_epi32(a6, a7); // 

  w0 = _mm256_unpacklo_epi64(b0, b2); // 00 10 20 30
  w1 = _mm256_unpackhi_epi64(b0, b2); // 01 11 21 31
  w2 = _mm256_unpacklo_epi64(b1, b3); // 02 12 22 32
  w3 = _mm256_unpackhi_epi64(b1, b3); // 03 13 23 33
  w4 = _mm256_unpacklo_epi64(b4, b6); //
  w5 = _mm256_unpackhi_epi64(b4, b6); //
  w6 = _mm256_unpacklo_epi64(b5, b7); //
  w7 = _mm256_unpackhi_epi64(b5, b7); //
}

static inline void transposeAvxInt16(__m256i& w0, __m256i& w1, __m256i& w2, __m256i& w3, __m256i& w4, __m256i& w5, __m256i& w6, __m256i& w7,
  __m256i& w8, __m256i& w9, __m256i& wA, __m256i& wB, __m256i& wC, __m256i& wD, __m256i& wE, __m256i& wF)
{
  // 00 01 02 03 04 05 06 07 | 08 09 0A 0B 0C 0D 0E 0F
  // 10 11 12 13 14 15 16 17 | 18 19 1A 1B 1C 1D 1E 1F
  // 20 21 22 23 24 25 26 27 | 28 29 2A 2B 2C 2D 2E 2F
  // 30 31 32 33 34 35 36 37 | 38 39 3A 3B 3C 3D 3E 3F
  // 40 41 42 43 44 45 46 47 | 48 49 4A 4B 4C 4D 4E 4F
  // 50 51 52 53 54 55 56 57 | 58 59 5A 5B 5C 5D 5E 5F
  // 60 61 62 63 64 65 66 67 | 68 69 6A 6B 6C 6D 6E 6F
  // 70 71 72 73 74 75 76 77 | 78 79 7A 7B 7C 7D 7E 7F
  // 80 81 82 83 84 85 86 87 | 88 89 8A 8B 8C 8D 8E 8F
  // 90 91 92 93 94 95 96 97 | 98 99 9A 9B 9C 9D 9E 9F
  // A0 A1 A2 A3 A4 A5 A6 A7 | A8 A9 AA AB AC AD AE AF
  // B0 B1 B2 B3 B4 B5 B6 B7 | B8 B9 BA BB BC BD BE BF
  // C0 C1 C2 C3 C4 C5 C6 C7 | C8 C9 CA CB CC CD CE CF
  // D0 D1 D2 D3 D4 D5 D6 D7 | D8 D9 DA DB DC DD DE DF
  // E0 E1 E2 E3 E4 E5 E6 E7 | E8 E9 EA EB EC ED EE EF
  // F0 F1 F2 F3 F4 F5 F6 F7 | F8 F9 FA FB FC FD FE FF

  __m256i x0 = _mm256_permute2x128_si256(w0, w8, 0x20); // 00 01 02 03 04 05 06 07 | 80 81 82 83 84 85 86 87
  __m256i x1 = _mm256_permute2x128_si256(w1, w9, 0x20); // 10 11 12 13 14 15 16 17 | 90 91 92 93 94 95 96 97
  __m256i x2 = _mm256_permute2x128_si256(w2, wA, 0x20); // 20 21 22 23 24 25 26 27 | A0 A1 A2 A3 A4 A5 A6 A7
  __m256i x3 = _mm256_permute2x128_si256(w3, wB, 0x20); // 30 31 32 33 34 35 36 37 | B0 B1 B2 B3 B4 B5 B6 B7
  __m256i x4 = _mm256_permute2x128_si256(w4, wC, 0x20); // 40 41 42 43 44 45 46 47 | C0 C1 C2 C3 C4 C5 C6 C7
  __m256i x5 = _mm256_permute2x128_si256(w5, wD, 0x20); // 50 51 52 53 54 55 56 57 | D0 D1 D2 D3 D4 D5 D6 D7
  __m256i x6 = _mm256_permute2x128_si256(w6, wE, 0x20); // 60 61 62 63 64 65 66 67 | E0 E1 E2 E3 E4 E5 E6 E7
  __m256i x7 = _mm256_permute2x128_si256(w7, wF, 0x20); // 70 71 72 73 74 75 76 77 | F0 F1 F2 F3 F4 F5 F6 F7
  __m256i x8 = _mm256_permute2x128_si256(w0, w8, 0x31); // 08 09 0A 0B 0C 0D 0E 0F | 88 89 8A 8B 8C 8D 8E 8F
  __m256i x9 = _mm256_permute2x128_si256(w1, w9, 0x31); // 18 19 1A 1B 1C 1D 1E 1F | 98 99 9A 9B 9C 9D 9E 9F
  __m256i xA = _mm256_permute2x128_si256(w2, wA, 0x31); // 28 29 2A 2B 2C 2D 2E 2F | A8 A9 AA AB AC AD AE AF
  __m256i xB = _mm256_permute2x128_si256(w3, wB, 0x31); // 38 39 3A 3B 3C 3D 3E 3F | B8 B9 BA BB BC BD BE BF
  __m256i xC = _mm256_permute2x128_si256(w4, wC, 0x31); // 48 49 4A 4B 4C 4D 4E 4F | C8 C9 CA CB CC CD CE CF
  __m256i xD = _mm256_permute2x128_si256(w5, wD, 0x31); // 58 59 5A 5B 5C 5D 5E 5F | D8 D9 DA DB DC DD DE DF
  __m256i xE = _mm256_permute2x128_si256(w6, wE, 0x31); // 68 69 6A 6B 6C 6D 6E 6F | E8 E9 EA EB EC ED EE EF
  __m256i xF = _mm256_permute2x128_si256(w7, wF, 0x31); // 78 79 7A 7B 7C 7D 7E 7F | F8 F9 FA FB FC FD FE FF

  __m256i y0 = _mm256_unpacklo_epi16(x0, x1); // 00 10 01 11 02 12 03 13 | ...
  __m256i y1 = _mm256_unpackhi_epi16(x0, x1); // 04 14 05 15 06 16 07 17 |
  __m256i y2 = _mm256_unpacklo_epi16(x2, x3); // 20 30 21 31 22 32 23 33 |
  __m256i y3 = _mm256_unpackhi_epi16(x2, x3); // 24 34 25 35 26 36 27 37 |
  __m256i y4 = _mm256_unpacklo_epi16(x4, x5); // 40 50 41 51 42 52 43 53 |
  __m256i y5 = _mm256_unpackhi_epi16(x4, x5); // 44 54 45 55 46 56 47 57 |
  __m256i y6 = _mm256_unpacklo_epi16(x6, x7); // 60 70 61 71 62 72 63 73 |
  __m256i y7 = _mm256_unpackhi_epi16(x6, x7); // 64 74 65 75 66 76 67 77 |
  __m256i y8 = _mm256_unpacklo_epi16(x8, x9); // ...
  __m256i y9 = _mm256_unpackhi_epi16(x8, x9);
  __m256i yA = _mm256_unpacklo_epi16(xA, xB);
  __m256i yB = _mm256_unpackhi_epi16(xA, xB);
  __m256i yC = _mm256_unpacklo_epi16(xC, xD);
  __m256i yD = _mm256_unpackhi_epi16(xC, xD);
  __m256i yE = _mm256_unpacklo_epi16(xE, xF);
  __m256i yF = _mm256_unpackhi_epi16(xE, xF);

  x0 = _mm256_unpacklo_epi32(y0, y2); // 00 10 20 30 01 11 21 31 |
  x1 = _mm256_unpackhi_epi32(y0, y2); // 02 12 22 32 03 13 23 33 |
  x2 = _mm256_unpacklo_epi32(y1, y3); // 04 14 24 34 05 15 25 35 |
  x3 = _mm256_unpackhi_epi32(y1, y3); // 06 16 26 36 07 17 27 37 |
  x4 = _mm256_unpacklo_epi32(y4, y6); // 40 50 60 70 41 51 61 71 |
  x5 = _mm256_unpackhi_epi32(y4, y6); // 42 52 62 72 43 53 63 73 |
  x6 = _mm256_unpacklo_epi32(y5, y7); // 44 54 64 74 45 55 65 75 |
  x7 = _mm256_unpackhi_epi32(y5, y7); // 46 56 66 76 47 57 67 77 |
  x8 = _mm256_unpacklo_epi32(y8, yA); // ...
  x9 = _mm256_unpackhi_epi32(y8, yA); 
  xA = _mm256_unpacklo_epi32(y9, yB); 
  xB = _mm256_unpackhi_epi32(y9, yB); 
  xC = _mm256_unpacklo_epi32(yC, yE); 
  xD = _mm256_unpackhi_epi32(yC, yE); 
  xE = _mm256_unpacklo_epi32(yD, yF); 
  xF = _mm256_unpackhi_epi32(yD, yF); 

  w0 = _mm256_unpacklo_epi64(x0, x4); // 00 10 20 30 40 50 60 70 |
  w1 = _mm256_unpackhi_epi64(x0, x4); // 01 11 21 31 41 51 61 71 |
  w2 = _mm256_unpacklo_epi64(x1, x5); // 02 12 22 32 42 52 62 72 |
  w3 = _mm256_unpackhi_epi64(x1, x5); // 03 13 23 33 43 53 63 73 |
  w4 = _mm256_unpacklo_epi64(x2, x6); // 04 14 24 34 44 54 64 74 |
  w5 = _mm256_unpackhi_epi64(x2, x6); // 05 15 25 35 45 55 65 75 |
  w6 = _mm256_unpacklo_epi64(x3, x7); // 06 16 26 36 46 56 66 76 |
  w7 = _mm256_unpackhi_epi64(x3, x7); // 07 17 27 37 47 57 67 77 |
  w8 = _mm256_unpacklo_epi64(x8, xC);
  w9 = _mm256_unpackhi_epi64(x8, xC);
  wA = _mm256_unpacklo_epi64(x9, xD);
  wB = _mm256_unpackhi_epi64(x9, xD);
  wC = _mm256_unpacklo_epi64(xA, xE);
  wD = _mm256_unpackhi_epi64(xA, xE);
  wE = _mm256_unpacklo_epi64(xB, xF);
  wF = _mm256_unpackhi_epi64(xB, xF);
}

}

}
