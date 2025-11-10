#pragma once

#include <cstdint>

namespace Platform
{

namespace Cpu
{

template <int alignment>
struct x86Simd
{
  constexpr static int byteAlignment()
  {
    return alignment;
  }

  static bool isPointerAligned(const void* p)
  {
    return ((intptr_t)p & (alignment - 1)) == 0;
  }

  template<typename T>
  static inline T* allocMemory(size_t count)
  {
    return (T*)_mm_malloc(sizeof(T) * count, alignment);
  }

  static inline void freeMemory(void* p)
  {
    _mm_free(p);
  }
};

}

}
