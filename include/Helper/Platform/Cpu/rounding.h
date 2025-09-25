#pragma once

#include "cpu.h"

#ifdef PLATFORM_CPU_X86
#  include "x86/rounding_x86.h"
#else
#  include "emulated/rounding_emulated.h"
#endif

namespace Platform
{

namespace Cpu
{

template <typename T> static inline int ifloor(T x);
template <typename T> static inline int iceil(T x);

template<> inline int ifloor<float>(float x)
{
  return ifloorf(x);
}

template<> inline int ifloor<double>(double x)
{
  return ifloord(x);
}

template<> inline int iceil<float>(float x)
{
  return iceilf(x);
}

template<> inline int iceil<double>(double x)
{
  return iceild(x);
}

}

}
