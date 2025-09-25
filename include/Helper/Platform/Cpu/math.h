#pragma once

#include "cpu.h"

#ifdef PLATFORM_CPU_X86
#  include "x86/math_x86.h"
#else
#  include "emulated/math_emulated.h"
#endif

namespace Platform
{

namespace Cpu
{

template <typename T> static inline T sqrt(T x);
template <typename T> static inline T rsqrt(T x);

static inline float sqrtf(float x);
static inline double sqrtd(double x);
static inline float rsqrtf(float x);
static inline double rsqrtd(double x);

template<> inline float sqrt<float>(float x)
{
  return sqrtf(x);
}

template<> inline double sqrt<double>(double x)
{
  return sqrtd(x);
}

template<> inline float rsqrt<float>(float x)
{
  return rsqrtf(x);
}

template<> inline double rsqrt<double>(double x)
{
  return rsqrtd(x);
}

}

}
