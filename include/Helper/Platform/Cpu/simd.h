#pragma once

#include <cstdint>
#include <type_traits>

#include "math.h"
#include "rounding.h"

#define SIMD_INT_MAX_WIDTH 1
#define SIMD_FLOAT_MAX_WIDTH 1
#define SIMD_DOUBLE_MAX_WIDTH 1

namespace Platform
{

namespace Cpu
{

enum SimdFeature
{
  Multiplication  = 0x0001,
  Abs             = 0x0010,
  MulSign         = 0x0020,
  InitFromUint8   = 0x0100,
  RevertByteOrder = 0x1000,
};

typedef uint64_t SimdFeatures;

struct GenericSimd
{
  static constexpr bool isPointerAligned(const void*)
  {
    return true;
  }

  template<typename T2>
  static inline T2* allocMemory(size_t count);
  static inline void freeMemory(void* p);
};

template<typename T> struct GenericExtendedIntegerType;

template<>
struct GenericExtendedIntegerType<int16_t>
{
  typedef int32_t Type;
};

template<>
struct GenericExtendedIntegerType<int32_t>
{
  typedef int64_t Type;
};

template<>
struct GenericExtendedIntegerType<uint32_t>
{
  typedef uint64_t Type;
};

template <typename T, int width>
struct SIMD : public GenericSimd
{
  typedef uint64_t ConditionType;
  struct Type
  {
    T values[width];

    Type() {}
    Type(const typename SIMD<typename std::make_signed<T>::type, width>::Type& other);
    Type(const typename SIMD<typename std::make_unsigned<T>::type, width>::Type& other);

    template<bool aligned> static inline Type load(const T* src);
    inline void store(T* dst) const;

    inline Type operator-() const;
    inline Type operator+(const Type& other) const;
    inline Type operator-(const Type& other) const;
    inline Type operator*(const Type& other) const;
    inline Type operator*(T factor) const;
    inline Type operator/(const Type& other) const;
    inline Type operator<<(int count) const;
    inline Type operator>>(int count) const;

    inline Type& operator+=(const Type& other);
    inline Type& operator*=(T factor);

    inline Type operator&(const Type& other) const;

    inline ConditionType operator<(const Type& other) const;
    inline ConditionType operator<=(const Type& other) const;
    inline ConditionType operator>(const Type& other) const;
    inline ConditionType operator>=(const Type& other) const;
  };
  typedef Type NativeType;

  struct ExtendedType
  {
    typedef typename GenericExtendedIntegerType<T>::Type ScalarType;
    typename SIMD<ScalarType, width>::Type lo, hi;
  };

  static inline constexpr bool isSupported(SimdFeatures features);

  static inline Type zero();

  // populate all SIMD vector components with given value
  static inline Type populate(T value);

  template <typename... Args>
  static inline Type create(Args...);

  static inline ExtendedType mulExtended(Type a, Type b);
  static inline Type mulExtended(Type a, Type b, Type& abhi);

  static inline Type min(Type a, Type b);
  static inline Type max(Type a, Type b);
  static inline Type abs(Type a);
  static inline Type mulSign(Type a, Type sign);
  // per component selection by per component condition
  static inline Type select(ConditionType condition, Type a, Type b);

  static inline Type sqrt(Type value);

  static inline typename SIMD<int, width>::Type ifloor(Type value);
  static inline typename SIMD<int, width>::Type iceil(Type value);

  static inline Type load(const T* src);
  static inline Type loadUnaligned(const T* src);
  static inline void store(T* dst, const Type& value);
  static inline void storeUnaligned(T* dst, const Type& value);

  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose(Type* dst, const T* src);

  static inline T reductionSum(const Type& value);

private:
  template <int index, typename... Args>
  static inline void setValuesFromIndex(Type& dst, T value, Args...);
  template <int index>
  static inline void setValuesFromIndex(Type& dst, T value);
};

template <typename T>
struct SIMD<T, 1> : public GenericSimd
{
  typedef T Type;
  typedef T NativeType;
  typedef T ParamType;
  typedef bool ConditionType;
  typedef typename GenericExtendedIntegerType<T>::Type ExtendedType;
  typedef typename GenericExtendedIntegerType<T>::Type ExtendedItemType;

  static inline constexpr bool isSupported(SimdFeatures features);

  static inline constexpr Type zero();

  // populate all SIMD vector components with given value
  static inline Type populate(T value);
  // rotate SIMD vector components in the direction of the least component
  static inline Type rotate(Type value);
  // extract least SIMD vector component
  static inline T least(Type value);

  static inline Type mulSigned(Type a, Type b, Type& abhi);
  static inline Type mulUnsigned(Type a, Type b, Type& abhi);

  static inline Type min(Type a, Type b);
  static inline Type max(Type a, Type b);
  // per component selection by per component condition
  static inline Type select(ConditionType condition, Type a, Type b);

  static inline Type sqrt(Type value);

  static inline int ifloor(Type value);
  static inline int iceil(Type value);

  template <bool aligned> static inline Type load(const T* src);
  static inline Type load(const T* src);
  static inline Type loadUnaligned(const T* src);
  template <bool aligned> static inline void store(T* dst, Type value);
  static inline void store(T* dst, Type value);
  static inline void storeUnaligned(T* dst, Type value);

  template <bool aligned, typename T2>
  static inline T loadAndConvert(const T2* p);
  template <bool aligned, typename T2>
  static inline void convertAndStore(T2* p, T value);

  static inline T reductionSum(Type value);

  template<bool aligned, int dstStride = 1, int srcStride = 1>
  static inline void transpose(Type* dst, const Type* src);

  static inline Type interleaveLow16Bit(Type a, Type b);
};

template<typename T> struct SimdDetector;

// no simd fallback functions

template<typename FloatType> static FloatType mulVectorsNoSimd(const FloatType* a, const FloatType* b, int size);

// implementation

// SIMD<T, width>

template<typename T, int width>
inline constexpr bool SIMD<T, width>::isSupported(SimdFeatures)
{
  return false;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::zero()
{
  Type result = {};
  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::populate(T value)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = value;

  return result;
}

template<typename T, int width>
template<typename ...Args>
inline typename SIMD<T, width>::Type SIMD<T, width>::create(Args... args)
{
  Type result;

  setValuesFromIndex<0>(result, args...);

  return result;
}

template<typename T, int width>
template<bool aligned, int dstStride, int srcStride>
inline void SIMD<T, width>::transpose(Type* dst, const T* src)
{
  for(int i = 0; i < width; i++)
  {
    for (int j = 0; j < width; j++)
      dst[j * dstStride].values[i] = src[i * srcStride * width + j];
  }
}

template<typename T, int width>
template<int index, typename ...Args>
inline void SIMD<T, width>::setValuesFromIndex(Type& dst, T value, Args... args)
{
  static_assert(index < width - 1, "too many arguments");
  dst.values[index] = value;
  setValuesFromIndex<index + 1>(dst, args...);
}

template<typename T, int width>
template<int index>
inline void SIMD<T, width>::setValuesFromIndex(Type& dst, T value)
{
  static_assert(index == width - 1, "not enough arguments");
  dst.values[index] = value;
}

template<typename T, int width>
inline typename SIMD<T, width>::ExtendedType SIMD<T, width>::mulExtended(Type a, Type b)
{
  ExtendedType result;
  typedef typename ExtendedType::ScalarType ExtendedScalarType;

  for (int i = 0; i < width/2; i++)
    result.lo.values[i] = (ExtendedScalarType)a.values[i] * (ExtendedScalarType)b.values[i];
  for (int i = width/2; i < width; i++)
    result.hi.values[i - width/2] = (ExtendedScalarType)a.values[i] * (ExtendedScalarType)b.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::mulExtended(Type a, Type b, Type& abhi)
{
  typedef typename ExtendedType::ScalarType ExtendedScalarType;
  Type ablo;
  constexpr int shift = (8 * sizeof(T));
  for (int i = 0; i < width; i++)
  {
    ExtendedScalarType result = (ExtendedScalarType)a.values[i] * (ExtendedScalarType)b.values[i];
    ablo.values[i] = result & ((~(ExtendedScalarType)0) >> shift);
    abhi.values[i] = (result >> shift) & ((~(ExtendedScalarType)0) >> shift);
  }

  return ablo;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::min(Type a, Type b)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = qMin(a.values[i], b.values[i]);

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::max(Type a, Type b)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = qMax(a.values[i], b.values[i]);

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::abs(Type a)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = a.values[i] < 0 ? -a.values[i] : a.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::mulSign(Type a, Type sign)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = sign.values[i] < 0 ? -a.values[i] : a.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::select(ConditionType condition, Type a, Type b)
{
  static_assert(width <= 64, "not implemented");
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = (condition & (1ull << i)) ? a.values[i] : b.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::sqrt(Type a)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = Platform::Cpu::sqrt<T>(a.values[i]);

  return result;
}

template<typename T, int width>
inline typename SIMD<int, width>::Type SIMD<T, width>::ifloor(Type a)
{
  typename SIMD<int, width>::Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = Platform::Cpu::ifloor<T>(a.values[i]);

  return result;
}

template<typename T, int width>
inline typename SIMD<int, width>::Type SIMD<T, width>::iceil(Type a)
{
  typename SIMD<int, width>::Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = Platform::Cpu::iceil<T>(a.values[i]);

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::load(const T* src)
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = src[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::loadUnaligned(const T* src)
{
  return load(src);
}

template<typename T, int width>
inline void SIMD<T, width>::store(T* dst, const Type& value)
{
  for (int i = 0; i < width; i++)
    dst[i] = value.values[i];
}

template<typename T, int width>
inline void SIMD<T, width>::storeUnaligned(T* dst, const Type& value)
{
  store(dst, value);
}

template<typename T, int width>
inline T SIMD<T, width>::reductionSum(const Type& value)
{
  T result = 0;

  for (int i = 0; i < width; i++)
    result += value.values[i];

  return result;
}

template<typename T, int width>
inline SIMD<T, width>::Type::Type(const typename SIMD<typename std::make_signed<T>::type, width>::Type& other)
{
  for (int i = 0; i < width; i++)
    values[i] = other.values[i];
}

template<typename T, int width>
inline SIMD<T, width>::Type::Type(const typename SIMD<typename std::make_unsigned<T>::type, width>::Type& other)
{
  for (int i = 0; i < width; i++)
    values[i] = other.values[i];
}

template<typename T, int width> template<bool aligned>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::load(const T* src)
{
  Type value;
  memcpy(value.values, src, sizeof(value.values));
  return value;
}

template<typename T, int width>
inline void SIMD<T, width>::Type::store(T* dst) const
{
  memcpy(dst, values, sizeof(values));
}

template<typename T2>
inline T2* GenericSimd::allocMemory(size_t count)
{
  return (T2*)malloc(sizeof(T2) * count);
}

inline void GenericSimd::freeMemory(void* p)
{
  free(p);
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator-() const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = -values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator+(const Type& other) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] + other.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator-(const Type& other) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] - other.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator*(const Type& other) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] * other.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator*(T factor) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] * factor;

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator/(const Type& other) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] / other.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator<<(int count) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] << count;

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator>>(int count) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] >> count;

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type& SIMD<T, width>::Type::operator+=(const Type& other)
{
  for (int i = 0; i < width; i++)
    values[i] += other.values[i];

  return *this;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type& SIMD<T, width>::Type::operator*=(T factor)
{
  for (int i = 0; i < width; i++)
    values[i] *= factor;

  return *this;
}

template<typename T, int width>
inline typename SIMD<T, width>::Type SIMD<T, width>::Type::operator&(const Type& other) const
{
  Type result;

  for (int i = 0; i < width; i++)
    result.values[i] = values[i] & other.values[i];

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::ConditionType SIMD<T, width>::Type::operator<(const Type& other) const
{
  static_assert(width <= 64, "not implemented");
  ConditionType result = 0;

  for (int i = 0; i < width; i++)
    result |= (values[i] < other.values[i] ? 1ull : 0ull) << i;

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::ConditionType SIMD<T, width>::Type::operator<=(const Type& other) const
{
  static_assert(width <= 64, "not implemented");
  ConditionType result = 0;

  for (int i = 0; i < width; i++)
    result |= (values[i] <= other.values[i] ? 1ull : 0ull) << i;

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::ConditionType SIMD<T, width>::Type::operator>(const Type& other) const
{
  static_assert(width <= 64, "not implemented");
  ConditionType result = 0;

  for (int i = 0; i < width; i++)
    result |= (values[i] > other.values[i] ? 1ull : 0ull) << i;

  return result;
}

template<typename T, int width>
inline typename SIMD<T, width>::ConditionType SIMD<T, width>::Type::operator>=(const Type& other) const
{
  static_assert(width <= 64, "not implemented");
  ConditionType result = 0;

  for (int i = 0; i < width; i++)
    result |= (values[i] > other.values[i] ? 1ull : 0ull) << i;

  return result;
}

// SIMD<T, 1>

template<typename T>
inline constexpr bool SIMD<T, 1>::isSupported(SimdFeatures)
{
  return true;
}

template<typename T>
inline constexpr typename SIMD<T, 1>::Type SIMD<T, 1>::zero()
{
  return 0;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::populate(T value)
{
  return value;
}

template<typename T>
inline T SIMD<T, 1>::least(Type value)
{
  return value;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::rotate(Type value)
{
  return value;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::mulSigned(Type a, Type b, Type& abhi)
{
  static_assert(std::is_integral<Type>() && std::is_signed<Type>() && sizeof(T) <= 4, "32-bit (or less) signed integral type required");
  int64_t ab = (int64_t)(typename std::make_signed<T>::type)a * (typename std::make_signed<T>::type)b;
  constexpr int64_t mask = (1ull << (sizeof(T) * 8)) - 1;
  abhi = (T)((ab >> (sizeof(T) * 8)) & mask);
  return (T)(ab & mask);
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::mulUnsigned(Type a, Type b, Type& abhi)
{
  static_assert(std::is_integral<Type>() && sizeof(T) <= 4, "32-bit (or less) integral type required");
  uint64_t ab = (uint64_t)(typename std::make_unsigned<T>::type)a * (typename std::make_unsigned<T>::type)b;
  constexpr uint64_t mask = (1ull << (sizeof(T) * 8)) - 1;
  abhi = (T)((ab >> (sizeof(T) * 8)) & mask);
  return (T)(ab & mask);
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::min(Type a, Type b)
{
  return a < b ? a : b;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::max(Type a, Type b)
{
  return a > b ? a : b;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::select(ConditionType condition, Type a, Type b)
{
  return condition ? a : b;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::sqrt(Type value)
{
  return Platform::Cpu::sqrt<Type>(value);
}

template<typename T>
inline int SIMD<T, 1>::ifloor(Type value)
{
  return Platform::Cpu::ifloor<Type>(value);
}

template<typename T>
inline int SIMD<T, 1>::iceil(Type value)
{
  return Platform::Cpu::iceil<Type>(value);
}

template<typename T> template<bool aligned>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::load(const T* src)
{
  return *src;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::load(const T* src)
{
  return *src;
}

template<typename T>
inline typename SIMD<T, 1>::Type SIMD<T, 1>::loadUnaligned(const T* src)
{
  return *src;
}

template<typename T> template<bool aligned>
inline void SIMD<T, 1>::store(T* dst, Type value)
{
  *dst = value;
}

template<typename T>
inline void SIMD<T, 1>::store(T* dst, Type value)
{
  *dst = value;
}

template<typename T>
inline void SIMD<T, 1>::storeUnaligned(T* dst, Type value)
{
  *dst = value;
}

template<typename T> template<bool aligned, typename T2>
inline T SIMD<T, 1>::loadAndConvert(const T2* p)
{
  return (T)*p;
}

template<typename T> template<bool aligned, typename T2>
inline void SIMD<T, 1>::convertAndStore(T2* p, T value)
{
  *p = (T2)value;
}


template<typename T>
inline T SIMD<T, 1>::reductionSum(Type value)
{
  return value;
}

template<typename T> template<bool aligned, int dstStride, int srcStride>
inline void SIMD<T, 1>::transpose(Type* dst, const Type* src)
{
  *dst = *src;
}

template<typename T>
inline T SIMD<T, 1>::interleaveLow16Bit(Type a, Type b)
{
  static_assert(std::is_integral<Type>() && sizeof(T) >= 4, "32-bit (or more) integral type required");
  return (T)(a & 0xffff) | ((T)((b & 0xffff)) << 16);
}

static inline float mul_add(float t1, float m1, float m2)
{
  return t1 + m1*m2;
}

static inline float mul_sub(float me, float m1, float m2)
{
  return me - m1 * m2;
}

static inline bool cond_not(bool a)
{
  return !a;
}

static inline bool cond_and(bool a, bool b)
{
  return a && b;
}

static inline bool cond_or(bool a, bool b)
{
  return a || b;
}

template<typename FloatType> static inline FloatType mulVectorsNoSimd(const FloatType* a, const FloatType* b, int size)
{
  FloatType sum = 0.0f;

  for (int j = 0; j < size; j++)
    sum += a[j] * b[j];

  return sum;
}

}

}

#ifdef PLATFORM_CPU_X86
#  include "x86/sse/simd_sse.h"
#  ifndef int128_t
#    define int128_t __m128i
#  endif
#  ifndef PLATFORM_CPU_FEATURE_NO_AVX
#    include "x86/avx/simd_avx.h"
#  endif
#else
#  ifndef int128_t
#    define int128_t typename SIMD<int64_t, 2>::Type
#  endif
#endif

namespace Platform
{

namespace Cpu
{

typedef typename SIMD<float, 4>::Type floatx4_t;
typedef typename SIMD<float, 8>::Type floatx8_t;

typedef typename SIMD<int8_t, 16>::Type int8x16_t;
typedef typename SIMD<int16_t, 8>::Type int16x8_t;
typedef typename SIMD<int32_t, 4>::Type int32x4_t;
typedef typename SIMD<int64_t, 2>::Type int64x2_t;

typedef typename SIMD<int8_t, 32>::Type int8x32_t;
typedef typename SIMD<int16_t, 16>::Type int16x16_t;
typedef typename SIMD<int16_t, 16>::ConditionType int16x16_cond_t;
typedef typename SIMD<int32_t, 8>::Type int32x8_t;
typedef typename SIMD<int64_t, 4>::Type int64x4_t;
typedef typename SIMD<int128_t, 2>::Type int128x2_t;

typedef typename SIMD<uint8_t, 16>::Type uint8x16_t;
typedef typename SIMD<uint16_t, 8>::Type uint16x8_t;
typedef typename SIMD<uint32_t, 4>::Type uint32x4_t;

typedef typename SIMD<uint8_t, 32>::Type uint8x32_t;
typedef typename SIMD<uint16_t, 16>::Type uint16x16_t;
typedef typename SIMD<uint32_t, 8>::Type uint32x8_t;
typedef typename SIMD<uint64_t, 4>::Type uint64x4_t;

#ifdef PLATFORM_CPU_X86
template<>
struct SimdDetector<int8_t>
{
  static int maxSimdLength(SimdFeatures features, int lengthLimit = 0x7fffffff)
  {
    if (lengthLimit >= 32 && SIMD<int8_t, 32>::isSupported(features))
      return 32;
    else if (lengthLimit >= 16 && SIMD<int8_t, 16>::isSupported(features))
      return 16;
    return 1;
  }
};

template<>
struct SimdDetector<int16_t>
{
  static int maxSimdLength(SimdFeatures features, int lengthLimit = 0x7fffffff)
  {
    if (lengthLimit >= 16 && SIMD<int16_t, 16>::isSupported(features))
      return 16;
    else if (lengthLimit >= 8 && SIMD<int16_t, 8>::isSupported(features))
      return 8;
    return 1;
  }
};

template<>
struct SimdDetector<int32_t>
{
  static int maxSimdLength(SimdFeatures features, int lengthLimit = 0x7fffffff)
  {
    if (lengthLimit >= 8 && SIMD<int32_t, 8>::isSupported(features))
      return 8;
    else if (lengthLimit >= 4 && SIMD<int32_t, 4>::isSupported(features))
      return 4;
    return 1;
  }
};

template<>
struct SimdDetector<int64_t>
{
  static int maxSimdLength(SimdFeatures features, int lengthLimit = 0x7fffffff)
  {
    if (lengthLimit >= 4 && SIMD<int64_t, 4>::isSupported(features))
      return 4;
    else if (lengthLimit >= 2 && SIMD<int64_t, 2>::isSupported(features))
      return 2;
    return 1;
  }
};
#endif

}

#ifdef PLATFORM_CPU_X86
static inline void* aligned_alloc(size_t alignment, size_t size)
{
  return _mm_malloc(size, alignment);
}

static inline void aligned_free(void* p)
{
  return _mm_free(p);
}
#else
static inline void* aligned_alloc(size_t alignment, size_t size)
{
  return std::aligned_alloc(size, alignment);
}

static inline void aligned_free(void* p)
{
  return std::free(p);
}
#endif

}
