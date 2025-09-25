#pragma once

#include <math.h>

namespace Platform
{

namespace Cpu
{

 static inline float sqrtf(float x)
{
  return ::sqrtf(x);
}

static inline double sqrtd(double x)
{
  return ::sqrt(x);
}

static inline float rsqrtf(float x)
{
  return 1.0f / ::sqrtf(x);
}

static inline double rsqrtd(double x)
{
  return 1.0f / ::sqrt(x);
}

}

}
