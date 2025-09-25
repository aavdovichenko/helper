#pragma once

namespace Platform
{

namespace Cpu
{

struct x86Simd
{
  static bool isPointerAligned(const void* p)
  {
    return ((intptr_t)p & 0xf) == 0;
  }

  template<typename T>
  static inline T* allocMemory(size_t count)
  {
    return (T*)_mm_malloc(sizeof(T) * count, 16);
  }

  static inline void freeMemory(void* p)
  {
    _mm_free(p);
  }
};

}

}
