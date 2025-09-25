#pragma once

namespace Platform
{

namespace Cpu
{

template <typename T, typename SimdType, typename Implementation>
struct SimdConditionType
{
  SimdType value;
  inline SimdConditionType& operator=(SimdType v) { value = v; return *this; }

  inline static Implementation fromNativeType(SimdType v) { Implementation x; x.value = v; return x; }

  operator SimdType() const { return value; }
};

}

}
