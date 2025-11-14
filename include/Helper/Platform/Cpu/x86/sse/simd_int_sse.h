#pragma once

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

template<typename T> template<bool aligned>
inline typename SseIntSimd<T>::Type SseIntSimd<T>::load(const T* src)
{
  return Type::fromNativeType(aligned ? _mm_load_si128((const __m128i*)src) : _mm_loadu_si128((const __m128i*)src));
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
  return SseSimdIntType<T>::fromNativeType(_mm_setzero_si128());
}

template<typename T>
inline typename SseIntSimd<T>::Type SseIntSimd<T>::load(const T* src)
{
  return Type::fromNativeType(_mm_load_si128((const __m128i*)src));
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

}

}
