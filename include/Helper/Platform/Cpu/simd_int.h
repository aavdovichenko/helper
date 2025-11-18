#pragma once

#include <utility>

namespace Platform
{

namespace Cpu
{

template <typename T, typename SimdType, typename Implementation>
struct SimdIntType
{
  SimdType value;
//    SimdIntType() {}
//    SimdIntType(SimdType v) : value(v) {}
  inline SimdIntType& operator=(SimdType v) { value = v; return *this; }

  inline static Implementation fromNativeType(SimdType&& v) { return Implementation{std::move(v)}; }

  template<int i> inline Implementation& insert(T x);
  template<int i0, int i1, int ...i, typename ...Args> inline Implementation& insert(T x0, T x1, Args... x);

  operator SimdType() const { return value; }
};

template <typename T, typename SimdType, typename SimdConditionType>
struct IntSimd
{
  struct ConditionType
  {
    SimdConditionType value;
    operator SimdConditionType() const { return value; }
  };
#if defined(DEBUG) || defined(_DEBUG)
  typedef const ConditionType& ConditionParamType;
#else
  typedef ConditionType ConditionParamType;
#endif
  typedef SimdType NativeType;
};

template<typename T, typename SimdType, typename Implementation> template<int i0, int i1, int ...i, typename ...Args>
inline Implementation& SimdIntType<T, SimdType, Implementation>::insert(T x0, T x1, Args... x)
{
  return insert<i0>(x0).template insert<i1, i...>(x1, x...);
}

}

}
