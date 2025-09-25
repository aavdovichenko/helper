#pragma once

#include "../../compiler.h"

#if defined(PLATFORM_COMPILER_MSVC)
#  include <intrin.h>
#endif

namespace Platform
{

namespace Cpu
{

static inline uint32_t x86cpuid(uint32_t eax, uint32_t ecx, int wordIndex)
{
  int info[4];
#if defined(PLATFORM_COMPILER_GNU) && !defined(PLATFORM_COMPILER_EMSCRIPTEN)
  asm("cpuid\n"
    : "=a" (info[0]), "=b" (info[1]), "=c" (info[2]), "=d" (info[3])
    : "a" (eax), "c"(ecx));
#elif defined(PLATFORM_COMPILER_MSVC)
  __cpuidex(info, eax, ecx);
#elif defined(PLATFORM_COMPILER_GHS)
  __CPUIDEX(eax, ecx, info);
#else
#error "unsupported compiler"
#endif
  return info[wordIndex];
}

static inline uint32_t getx86CpuFeaturesWord(int wordIndex)
{
  return x86cpuid(1, 0, wordIndex);
}

static inline uint32_t getx86CpuExtendedFeaturesWord(int wordIndex)
{
  if (x86cpuid(0, 0, 0) < 7)
    return 0;

  return x86cpuid(7, 0, wordIndex);
}

}

}
