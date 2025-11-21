#pragma once

#include <cstdint>
#include <pmmintrin.h> // sse/sse2/sse3

#include "../../simd_condition.h"
#include "../../simd_int.h"
#include "../simd_x86.h"

namespace Platform
{

namespace Cpu
{

static inline bool isSSSE3Enabled();
static inline bool isSSE41Enabled();

template <typename T> struct SseSimdIntType;

template <typename T>
struct SseSimdIntConditionType : public SimdConditionType<T, __m128i, SseSimdIntConditionType<T>>
{
  inline SseSimdIntConditionType() {}
  inline SseSimdIntConditionType(__m128i value) : SimdConditionType<T, __m128i, SseSimdIntConditionType<T>>{value} {}

  inline SseSimdIntType<T> mask() const;
  inline int64_t bitMask() const;
};

template <typename T, typename Implementation>
struct BaseSseSimdIntType : public SimdIntType<T, __m128i, Implementation>
{
  BaseSseSimdIntType() {}
  BaseSseSimdIntType(__m128i value) : SimdIntType<T, __m128i, Implementation>{value} {};

  static inline Implementation zero();

  template<bool aligned> static inline Implementation load(const T* src);
  static inline Implementation load(const T* src);
  template<bool aligned> inline void store(T* dst) const;
  static inline void store(T* dst, const Implementation& value);
  inline void store(T* dst) const;

  inline Implementation operator|(const Implementation& other) const;
  inline Implementation& operator|=(const Implementation& other);
  inline Implementation andNot(const Implementation& other) const;

  static inline Implementation select(const SseSimdIntConditionType<T>& condition, const Implementation& a, const Implementation& b);
};

template <typename T>
struct SseIntSimd : public x86Simd<16>, public IntSimd<T, __m128i, __m128i>
{
  typedef SseSimdIntType<T> Type;
  typedef SseSimdIntConditionType<T> ConditionType;
#if defined(DEBUG) || defined(_DEBUG)
  typedef const Type& ParamType;
  typedef const ConditionType& ConditionParamType;
#else
  typedef Type ParamType;
  typedef ConditionType ConditionParamType;
#endif

  static bool isSupported(SimdFeatures features = 0);

  static SseSimdIntType<T> zero();

  template<bool aligned> static inline Type load(const T* src);
  static inline Type load(const T* src);
  template<bool aligned> static inline void store(T* dst, ParamType value);
  static inline void store(T* dst, ParamType value);

  static inline Type select(ConditionType condition, Type a, Type b);
};

static inline void transposeSseInt16(__m128i& w0, __m128i& w1, __m128i& w2, __m128i& w3, __m128i& w4, __m128i& w5, __m128i& w6, __m128i& w7);

// implementation

// SseSimdIntType<T>

template<typename T>
inline SseSimdIntType<T> SseSimdIntConditionType<T>::mask() const
{
  return SseSimdIntType<T>{this->value};
}

template<typename T>
inline int64_t SseSimdIntConditionType<T>::bitMask() const
{
  return _mm_movemask_epi8(this->value);
}

template<typename T, typename Implementation> template<bool aligned>
inline Implementation BaseSseSimdIntType<T, Implementation>::load(const T* src)
{
  return Implementation::fromNativeType(aligned ? _mm_load_si128((const __m128i*)src) : _mm_loadu_si128((const __m128i*)src));
}

template<typename T, typename Implementation>
inline Implementation BaseSseSimdIntType<T, Implementation>::zero()
{
  return Implementation::fromNativeType(_mm_setzero_si128());
}

template<typename T, typename Implementation>
inline Implementation BaseSseSimdIntType<T, Implementation>::load(const T* src)
{
  return Implementation::fromNativeType(_mm_load_si128((const __m128i*)src));
}

template<typename T, typename Implementation> template<bool aligned>
inline void BaseSseSimdIntType<T, Implementation>::store(T* dst) const
{
  aligned ? _mm_store_si128((__m128i*)dst, this->value) : _mm_storeu_si128((__m128i*)dst, this->value);
}

template<typename T, typename Implementation>
inline void BaseSseSimdIntType<T, Implementation>::store(T* dst, const Implementation& value)
{
  _mm_store_si128((__m128i*)dst, value.value);
}

template<typename T, typename Implementation>
inline void BaseSseSimdIntType<T, Implementation>::store(T* dst) const
{
  _mm_store_si128((__m128i*)dst, this->value);
}

template<typename T, typename Implementation>
inline Implementation BaseSseSimdIntType<T, Implementation>::operator|(const Implementation& other) const
{
  return Implementation::fromNativeType(_mm_or_si128(this->value, other.value));
}

template<typename T, typename Implementation>
inline Implementation& BaseSseSimdIntType<T, Implementation>::operator|=(const Implementation& other)
{
  this->value = _mm_or_si128(this->value, other.value);
  return *(Implementation*)this;
}

template<typename T, typename Implementation>
inline Implementation BaseSseSimdIntType<T, Implementation>::andNot(const Implementation& other) const
{
  return Implementation::fromNativeType(_mm_andnot_si128(other.value, this->value));
}

template<typename T, typename Implementation>
inline Implementation BaseSseSimdIntType<T, Implementation>::select(const SseSimdIntConditionType<T>& condition, const Implementation& a, const Implementation& b)
{
  return Implementation::fromNativeType(_mm_or_si128(_mm_and_si128(condition, a), _mm_andnot_si128(condition, b)));
}

//  SseIntSimd<T>

template<typename T>
bool SseIntSimd<T>::isSupported(SimdFeatures features)
{
  static bool ssse3Enabled = isSSSE3Enabled();
  static bool sse41Enabled = isSSE41Enabled();

  if ((features & SimdFeature::InitFromUint8) && !sse41Enabled)
    return false;
  if ((features & (SimdFeature::Abs | SimdFeature::MulSign | SimdFeature::RevertByteOrder)) && !ssse3Enabled)
    return false;

  return true;
}

template<typename T>
inline SseSimdIntType<T> SseIntSimd<T>::zero()
{
  return Type{_mm_setzero_si128()};
}

template<typename T> template<bool aligned>
inline typename SseIntSimd<T>::Type SseIntSimd<T>::load(const T* src)
{
  return Type{aligned ? _mm_load_si128((const __m128i*)src) : _mm_loadu_si128((const __m128i*)src)};
}

template<typename T>
inline typename SseIntSimd<T>::Type SseIntSimd<T>::load(const T* src)
{
  return Type{_mm_load_si128((const __m128i*)src)};
}

template<typename T> template<bool aligned>
inline void SseIntSimd<T>::store(T* dst, ParamType value)
{
  aligned ? _mm_store_si128((__m128i*)dst, value.value) : _mm_storeu_si128((__m128i*)dst, value.value);
}

template<typename T>
inline void SseIntSimd<T>::store(T* dst, ParamType value)
{
  _mm_store_si128((__m128i*)dst, value.value);
}

template<typename T>
inline typename SseIntSimd<T>::Type SseIntSimd<T>::select(ConditionType condition, Type a, Type b)
{
  return _mm_or_si128(_mm_and_si128(condition, a), _mm_andnot_si128(condition, b));
}

static inline void transposeSseInt32(__m128i& w0, __m128i& w1, __m128i& w2, __m128i& w3)
{
#if 1
  // 00 01 02 03
  // 10 11 12 13
  // 20 21 22 23
  // 30 31 32 33

  __m128i a0 = _mm_unpacklo_epi32(w0, w1); // 00 10 01 11
  __m128i a1 = _mm_unpackhi_epi32(w0, w1); // 02 12 03 13
  __m128i a2 = _mm_unpacklo_epi32(w2, w3); // 20 30 21 31
  __m128i a3 = _mm_unpackhi_epi32(w2, w3); // 22 32 23 33

  w0 = _mm_unpacklo_epi64(a0, a2); // 00 10 20 30
  w1 = _mm_unpackhi_epi64(a0, a2); // 01 11 21 31
  w2 = _mm_unpacklo_epi64(a1, a3); // 02 12 22 32  
  w3 = _mm_unpackhi_epi64(a1, a3); // 03 13 23 33
#else
  __m128 v0 = _mm_castsi128_ps(w0), v1 = _mm_castsi128_ps(w1), v2 = _mm_castsi128_ps(w2), v3 = _mm_castsi128_ps(w3);
  _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
  w0 = _mm_castps_si128(v0);
  w1 = _mm_castps_si128(v1);
  w2 = _mm_castps_si128(v2);
  w3 = _mm_castps_si128(v3);
#endif
}

static inline void transposeSseInt16(__m128i& w0, __m128i& w1, __m128i& w2, __m128i& w3, __m128i& w4, __m128i& w5, __m128i& w6, __m128i& w7)
{
  // 00 01 02 03 04 05 06 07
  // 10 11 12 13 14 15 16 17
  // 20 21 22 23 24 25 26 27
  // 30 31 32 33 34 35 36 37
  // 40 41 42 43 44 45 46 47
  // 50 51 52 53 54 55 56 57
  // 60 61 62 63 64 65 66 67
  // 70 71 72 73 74 75 76 77

  __m128i a0 = _mm_unpacklo_epi16(w0, w1); // 00 10 01 11 02 12 03 13
  __m128i a1 = _mm_unpackhi_epi16(w0, w1); // 04 14 05 15 06 16 07 17 
  __m128i a2 = _mm_unpacklo_epi16(w2, w3); // 20 30 21 31 22 32 23 33
  __m128i a3 = _mm_unpackhi_epi16(w2, w3); // 24 34 25 35 26 36 27 37
  __m128i a4 = _mm_unpacklo_epi16(w4, w5); // 40 50 41 51 42 52 43 53
  __m128i a5 = _mm_unpackhi_epi16(w4, w5); // 44 54 45 55 46 56 47 57
  __m128i a6 = _mm_unpacklo_epi16(w6, w7); // 60 70 61 71 62 72 63 73
  __m128i a7 = _mm_unpackhi_epi16(w6, w7); // 64 74 65 75 66 76 67 77

  __m128i b0 = _mm_unpacklo_epi32(a0, a2); // 00 10 20 30 01 11 21 31
  __m128i b1 = _mm_unpackhi_epi32(a0, a2); // 02 12 22 32 03 13 23 33
  __m128i b2 = _mm_unpacklo_epi32(a1, a3); // 04 14 24 34 05 15 25 35
  __m128i b3 = _mm_unpackhi_epi32(a1, a3); // 06 16 26 36 07 17 27 37
  __m128i b4 = _mm_unpacklo_epi32(a4, a6); // 40 50 60 70 41 51 61 71 
  __m128i b5 = _mm_unpackhi_epi32(a4, a6); // 42 52 62 72 43 53 63 73
  __m128i b6 = _mm_unpacklo_epi32(a5, a7); // 44 54 64 74 45 55 65 75
  __m128i b7 = _mm_unpackhi_epi32(a5, a7); // 46 56 66 76 47 57 67 77

  w0 = _mm_unpacklo_epi64(b0, b4); // 00 10 20 30 40 50 60 70
  w1 = _mm_unpackhi_epi64(b0, b4); // 01 11 21 31 41 51 61 71 
  w2 = _mm_unpacklo_epi64(b1, b5); // 02 12 22 32 42 52 62 72
  w3 = _mm_unpackhi_epi64(b1, b5); // 03 13 23 33 43 53 63 73
  w4 = _mm_unpacklo_epi64(b2, b6); // 04 14 24 34 44 54 64 74
  w5 = _mm_unpackhi_epi64(b2, b6); // 05 15 25 35 45 55 65 75
  w6 = _mm_unpacklo_epi64(b3, b7); // 06 16 26 36 46 56 66 76
  w7 = _mm_unpackhi_epi64(b3, b7); // 07 17 27 37 47 57 67 77
}

static inline void transposeSseInt8(__m128i& w0, __m128i& w1, __m128i& w2, __m128i& w3, __m128i& w4, __m128i& w5, __m128i& w6, __m128i& w7,
  __m128i& w8, __m128i& w9, __m128i& wA, __m128i& wB, __m128i& wC, __m128i& wD, __m128i& wE, __m128i& wF)
{
  // 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
  // 10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F
  // 20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F
  // 30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F
  // 40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F
  // 50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F
  // 60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F
  // 70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F
  // 80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F
  // 90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F
  // A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF
  // B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF
  // C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 CA CB CC CD CE CF
  // D0 D1 D2 D3 D4 D5 D6 D7 D8 D9 DA DB DC DD DE DF
  // E0 E1 E2 E3 E4 E5 E6 E7 E8 E9 EA EB EC ED EE EF
  // F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 FA FB FC FD FE FF

  __m128i x0 = _mm_unpacklo_epi8(w0, w1); // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
  __m128i x1 = _mm_unpackhi_epi8(w0, w1); // 08 18 09 19 0A 1A 0B 1B 0C 1C 0D 1D 0E 1E 0F 1F
  __m128i x2 = _mm_unpacklo_epi8(w2, w3); // 20 30 21 31 22 32 23 33 24 34 25 35 26 36 27 37
  __m128i x3 = _mm_unpackhi_epi8(w2, w3); // 28 38 29 39 2A 3A 2B 3B 2C 3C 2D 3D 2E 3E 2F 3F
  __m128i x4 = _mm_unpacklo_epi8(w4, w5); // 40 50 41 51 42 52 43 53 44 54 45 55 46 56 47 57
  __m128i x5 = _mm_unpackhi_epi8(w4, w5); // 48 58 49 59 4A 5A 4B 5B 4C 5C 4D 5D 4E 5E 4F 5F
  __m128i x6 = _mm_unpacklo_epi8(w6, w7); // 60 70 61 71 62 72 63 73 64 74 65 75 66 76 67 77
  __m128i x7 = _mm_unpackhi_epi8(w6, w7); // 68 78 69 79 6A 7A 6B 7B 6C 7C 6D 7D 6E 7E 6F 7F
  __m128i x8 = _mm_unpacklo_epi8(w8, w9); // 80 90 81 91 82 92 83 93 84 94 85 95 86 96 87 97
  __m128i x9 = _mm_unpackhi_epi8(w8, w9); // 88 98 89 99 8A 9A 8B 9B 8C 9C 8D 9D 8E 9E 8F 9F
  __m128i xA = _mm_unpacklo_epi8(wA, wB); // A0 B0 A1 B1 A2 B2 A3 B3 A4 B4 A5 B5 A6 B6 A7 B7
  __m128i xB = _mm_unpackhi_epi8(wA, wB); // A8 B8 A9 B9 AA BA AB BB AC BC AD BD AE BE AF BF
  __m128i xC = _mm_unpacklo_epi8(wC, wD); // C0 D0 C1 D1 C2 D2 C3 D3 C4 D4 C5 D5 C6 D6 C7 D7
  __m128i xD = _mm_unpackhi_epi8(wC, wD); // C8 D8 C9 D9 CA DA CB DB CC DC CD DD CE DE CF DF
  __m128i xE = _mm_unpacklo_epi8(wE, wF); // E0 F0 E1 F1 E2 F2 E3 F3 E4 F4 E5 F5 E6 F6 E7 F7
  __m128i xF = _mm_unpackhi_epi8(wE, wF); // E8 F8 E9 F9 EA FA EB FB EC FC ED FD EE FE EF FF

  __m128i y0 = _mm_unpacklo_epi16(x0, x2); // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
  __m128i y1 = _mm_unpackhi_epi16(x0, x2); // 04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37
  __m128i y2 = _mm_unpacklo_epi16(x1, x3); // 08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B
  __m128i y3 = _mm_unpackhi_epi16(x1, x3); // 0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F
  __m128i y4 = _mm_unpacklo_epi16(x4, x6); // 40 50 60 70 41 51 61 71 42 52 62 72 43 53 63 73
  __m128i y5 = _mm_unpackhi_epi16(x4, x6); // 44 54 64 74 45 55 65 75 46 56 66 76 47 57 67 77
  __m128i y6 = _mm_unpacklo_epi16(x5, x7); // 48 58 68 78 49 59 69 79 4A 5A 6A 7A 4B 5B 6B 7B
  __m128i y7 = _mm_unpackhi_epi16(x5, x7); // 4C 5C 6C 7C 4D 5D 6D 7D 4E 5E 6E 7E 4F 5F 6F 7F
  __m128i y8 = _mm_unpacklo_epi16(x8, xA); // 80 90 A0 B0 81 91 A1 B1 82 92 A2 B2 83 93 A3 B3
  __m128i y9 = _mm_unpackhi_epi16(x8, xA); // 84 94 A4 B4 85 95 A5 B5 86 96 A6 B6 87 97 A7 B7
  __m128i yA = _mm_unpacklo_epi16(x9, xB); // 88 98 A8 B8 89 99 A9 B9 8A 9A AA BA 8B 9B AB BB
  __m128i yB = _mm_unpackhi_epi16(x9, xB); // 8C 9C AC BC 8D 9D AD BD 8E 9E AE BE 8F 9F AF BF
  __m128i yC = _mm_unpacklo_epi16(xC, xE); // C0 D0 E0 F0 C1 D1 E1 F1 C2 D2 E2 F2 C3 D3 E3 F3
  __m128i yD = _mm_unpackhi_epi16(xC, xE); // C4 D4 E4 F4 C5 D5 E5 F5 C6 D6 E6 F6 C7 D7 E7 F7
  __m128i yE = _mm_unpacklo_epi16(xD, xF); // C8 D8 E8 F8 C9 D9 E9 F9 CA DA EA FA CB DB EB FB
  __m128i yF = _mm_unpackhi_epi16(xD, xF); // CC DC EC FC CD DD ED FD CE DE EE FE CF DF EF FF

  x0 = _mm_unpacklo_epi32(y0, y4); // 00 10 20 30 40 50 60 70 01 11 21 31 41 51 61 71
  x1 = _mm_unpackhi_epi32(y0, y4); // 02 12 22 32 42 52 62 72 03 13 23 33 43 53 63 73
  x2 = _mm_unpacklo_epi32(y1, y5); // 04 14 24 34 44 54 64 74 05 15 25 35 45 55 65 75
  x3 = _mm_unpackhi_epi32(y1, y5); // 06 16 26 36 46 56 66 76 07 17 27 37 47 57 67 77
  x4 = _mm_unpacklo_epi32(y2, y6); // 08 18 28 38 48 58 68 78 09 19 29 39 49 59 69 79
  x5 = _mm_unpackhi_epi32(y2, y6); // 0A 1A 2A 3A 4A 5A 6A 7A 0B 1B 2B 3B 4B 5B 6B 7B
  x6 = _mm_unpacklo_epi32(y3, y7); // 0C 1C 2C 3C 4C 5C 6C 7C 0D 1D 2D 3D 4D 5D 6D 7D
  x7 = _mm_unpackhi_epi32(y3, y7); // 0E 1E 2E 3E 4E 5E 6E 7E 0F 1F 2F 3F 4F 5F 6F 7F
  x8 = _mm_unpacklo_epi32(y8, yC); // 80 90 A0 B0 C0 D0 E0 F0 81 91 A1 B1 C1 D1 E1 F1
  x9 = _mm_unpackhi_epi32(y8, yC); // 82 92 A2 B2 C2 D2 E2 F2 83 93 A3 B3 C3 D3 E3 F3
  xA = _mm_unpacklo_epi32(y9, yD); // 84 94 A4 B4 C4 D4 E4 F4 85 95 A5 B5 C5 D5 E5 F5
  xB = _mm_unpackhi_epi32(y9, yD); // 86 96 A6 B6 C6 D6 E6 F6 87 97 A7 B7 C7 D7 E7 F7
  xC = _mm_unpacklo_epi32(yA, yE); // 88 98 A8 B8 C8 D8 E8 F8 89 99 A9 B9 C9 D9 E9 F9
  xD = _mm_unpackhi_epi32(yA, yE); // 8A 9A AA BA CA DA EA FA 8B 9B AB BB CB DB EB FB
  xE = _mm_unpacklo_epi32(yB, yF); // 8C 9C AC BC CC DC EC FC 8D 9D AD BD CD DD ED FD
  xF = _mm_unpackhi_epi32(yB, yF); // 8E 9E AE BE CE DE EE FE 8F 9F AF BF CF DF EF FF

  w0 = _mm_unpacklo_epi64(x0, x8); // 00 10 20 30 40 50 60 70 80 90 A0 B0 C0 D0 E0 F0
  w1 = _mm_unpackhi_epi64(x0, x8); // 01 11 21 31 41 51 61 71 81 91 A1 B1 C1 D1 E1 F1
  w2 = _mm_unpacklo_epi64(x1, x9); // 02 12 22 32 42 52 62 72 82 92 A2 B2 C2 D2 E2 F2
  w3 = _mm_unpackhi_epi64(x1, x9); // 03 13 23 33 43 53 63 73 83 93 A3 B3 C3 D3 E3 F3
  w4 = _mm_unpacklo_epi64(x2, xA); // ...
  w5 = _mm_unpackhi_epi64(x2, xA);
  w6 = _mm_unpacklo_epi64(x3, xB);
  w7 = _mm_unpackhi_epi64(x3, xB);
  w8 = _mm_unpacklo_epi64(x4, xC);
  w9 = _mm_unpackhi_epi64(x4, xC);
  wA = _mm_unpacklo_epi64(x5, xD);
  wB = _mm_unpackhi_epi64(x5, xD);
  wC = _mm_unpacklo_epi64(x6, xE);
  wD = _mm_unpackhi_epi64(x6, xE);
  wE = _mm_unpacklo_epi64(x7, xF);
  wF = _mm_unpackhi_epi64(x7, xF);
}

}

}
