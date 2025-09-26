#pragma once

#include <cstdint>

#include "../compiler.h"
#include "cpu.h"

#if !defined(PLATFORM_COMPILER_MSVC) && !defined(PLATFORM_COMPILER_GNU)
#include <cassert>
#endif

#ifdef PLATFORM_CPU_X86
#  include "x86/x86.h"
#endif

namespace Platform
{

namespace Cpu
{

template <typename T> inline bool havePopcntInstruction();
template <typename T> inline int popcnt(T word);
template <typename T> inline int leastSignificantSetBit(T word);
template <typename T> inline int mostSignificantSetBit(T word);

// implementation

template <typename T> inline bool havePopcntInstruction()
{
#ifdef PLATFORM_CPU_X86
#ifdef PLATFORM_CPU_X86_32
  if (sizeof(T) == 8)
    return false;
#endif
  return getx86CpuFeaturesWord(2) & (1 << 23);
#else
  return false;
#endif
}

template<> inline int popcnt<uint16_t>(uint16_t word)
{
#ifdef PLATFORM_COMPILER_MSVC
  return __popcnt16(word);
#elif defined(PLATFORM_COMPILER_GNU)
  return __builtin_popcount(word);
#else
  (void)word;
  assert(false); // popcnt() should not be called if not supported
  return 0;
#endif
}

template<> inline int popcnt<uint32_t>(uint32_t word)
{
#ifdef PLATFORM_COMPILER_MSVC
  return __popcnt(word);
#elif defined(PLATFORM_COMPILER_GNU)
  return __builtin_popcount(word);
#else
  return popcnt<uint16_t>(word & 0xffff) + popcnt<uint16_t>((word >> 16) & 0xffff);
#endif
}

template<> inline int popcnt<uint64_t>(uint64_t word)
{
#ifdef PLATFORM_COMPILER_MSVC
  return (int)__popcnt64(word);
#elif defined(PLATFORM_COMPILER_GNU)
  return __builtin_popcountll(word);
#else
  return popcnt<uint32_t>(word & 0xffffffff) + popcnt<uint32_t>((word >> 32) & 0xffffffff);
#endif
}

template<typename T> inline int leastSignificantSetBit(T word)
{
  constexpr size_t WordBits = sizeof(T) * 8;
#if defined(PLATFORM_COMPILER_GNU) || defined(PLATFORM_COMPILER_CLANG)
  static_assert(WordBits <= 64, "leastSignificantSetBit() unsupported word size");
  if (WordBits <= 32)
    return __builtin_ctz(word);
  else if (WordBits <= 64)
    return __builtin_ctzll(word);
  return -1;
#elif defined(PLATFORM_COMPILER_MSVC)
  static_assert(WordBits <= 64, "leastSignificantSetBit() unsupported word size");
  unsigned long num = 0;
  if (WordBits <= 32)
    _BitScanForward(&num, (unsigned long)word);
  else if (WordBits <= 64)
  {
#if PLATFORM_CPU_WORD_BITS >= 64
    return (int)_tzcnt_u64(word);
//    _BitScanForward64(&num, word);
#else
    if (word & 0xffffffff)
      _BitScanForward(&num, (uint32_t)(word & 0xffffffff));
    else
    {
      _BitScanForward(&num, (uint32_t)(((uint64_t)word >> 32) & 0xffffffff));
      num += 32;
    }
#endif
  }
  return num;
#else
#error "unsupported compiler"
#endif
}

template<typename T> inline int mostSignificantSetBit(T word)
{
  constexpr size_t WordBits = sizeof(T) * 8;
#if defined(PLATFORM_COMPILER_GNU) || defined(PLATFORM_COMPILER_CLANG)
  static_assert(WordBits <= 64, "mostSignificantSetBit() unsupported word size");
  if (WordBits <= 32)
    return 31 - __builtin_clz(word);
  else if (WordBits <= 64)
    return 63 - __builtin_clzll(word);
  return -1;
#elif defined(PLATFORM_COMPILER_MSVC)
  static_assert(WordBits <= 64, "mostSignificantSetBit() unsupported word size");
  unsigned long num;
  if (WordBits <= 32)
    _BitScanReverse(&num, (unsigned long)word);
  else if (WordBits <= 64)
  {
#if PLATFORM_CPU_WORD_BITS >= 64
    _BitScanReverse64(&num, word);
#else
    if (word & 0xffffffff00000000ull)
    {
      _BitScanReverse(&num, (uint32_t)((uint64_t)word >> 32) & 0xffffffff);
      num += 32;
    }
    else
      _BitScanReverse(&num, (uint32_t)(word & 0xffffffff));
#endif
  }
  return num;
#else
#error "unsupported compiler"
#endif
}

}

}
