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
  template<bool aligned> inline void store(T* dst);
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

  template<bool aligned> static inline Type load(const T* src);
  static inline Type load(const T* src);
  static inline void store(T* dst, ParamType value);

  static inline Type select(ConditionType condition, Type a, Type b);
};

// implementation

// SseSimdIntType<T>

template<typename T>
inline SseSimdIntType<T> SseSimdIntConditionType<T>::mask() const
{
  return SseSimdIntType<T>::fromNativeType(this->value);
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
inline void BaseSseSimdIntType<T, Implementation>::store(T* dst)
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

}

}
