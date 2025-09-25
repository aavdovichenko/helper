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
  template<bool aligned> inline void store(T* dst);
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
struct AvxIntSimd : public x86Simd, public IntSimd<T, __m256i, __m256i>
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

  static bool isSupported();

  template<bool aligned> static inline Type load(const T* src);
  static inline Type load(const T* src);
  static inline void store(T* dst, ParamType value);
  static inline void storeLow(T* dst, ParamType value);
  static inline void storeHigh(T* dst, ParamType value);

  static inline Type select(ConditionType condition, Type a, Type b);

  constexpr static int PreferedAlignment = 32;
};

// implementation

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
inline void BaseAvxSimdIntType<T, Implementation>::store(T* dst)
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
  return Implementation::fromNativeType(_mm256_and_si256(this->value, other.value));
}

template<typename T, typename Implementation>
inline Implementation BaseAvxSimdIntType<T, Implementation>::operator|(const Implementation& other) const
{
  return Implementation::fromNativeType(_mm256_or_si256(this->value, other.value));
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

template<typename T>
inline bool AvxIntSimd<T>::isSupported()
{
  static bool avxEnabled = isAVXEnabled();
  static bool avx2Enabled = isAVX2Enabled();
  return avxEnabled && avx2Enabled; // AVX2 required
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

}

}
