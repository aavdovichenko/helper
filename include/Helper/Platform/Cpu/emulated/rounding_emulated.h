#pragma once

namespace Platform
{

namespace Cpu
{

static inline int ifloorf(float x)
{
  int i = (int)x;

  if ((float)i > x)
    i -= 1;

  return i;
}

static inline int ifloord(double x)
{
  int i = (int)x;

  if ((double)i > x)
    i -= 1;

  return i;
}

static inline int iceilf(float x)
{
  int i = (int)x;

  if ((float)i < x)
    i += 1;

  return i;
}

static inline int iceild(double x)
{
  int i = (int)x;

  if ((float)i < x)
    i += 1;

  return i;
}

}

}
