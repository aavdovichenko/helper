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
  return AvxSimdIntType<T>::fromNativeType(this->value);
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

}

}
