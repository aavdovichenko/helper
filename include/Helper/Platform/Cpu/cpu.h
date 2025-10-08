#pragma once

#include "../compiler.h"

namespace Platform
{

namespace Cpu
{

enum class Family
{
  Unknown,
  i386,
  x86_32,
  x86_64,
  Arm32,
  Arm64,
  Mips32,
  Mips64,
};

enum class Architecture
{
  Unknown,
  x86,
  Arm,
  Mips
};

enum ByteOrder
{
  LittleEndian,
  BigEndian
};

#if defined(__arm__) || defined(__TARGET_ARCH_ARM) || defined(_M_ARM) || defined(_M_ARM64) || defined(__aarch64__) || defined(__ARM64__)
#  define PLATFORM_CPU_ARM
#  if defined(__aarch64__) || defined(__ARM64__) || defined(_M_ARM64)
#   define PLATFORM_CPU_ARM_64
    constexpr Family family = Family::Arm64;
#  else
#   define PLATFORM_CPU_ARM_32
    constexpr Family family = Family::Arm32;
#  endif
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
#  define PLATFORM_CPU_X86
#  define PLATFORM_CPU_X86_32
#  if defined(__i686__) || defined(__athlon__) || defined(__SSE__) || defined(__pentiumpro__)
    constexpr Family family = Family::x86_32;
#  else
    constexpr Family family = Family::i386;
#  endif
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
#  define PLATFORM_CPU_X86
#  define PLATFORM_CPU_X86_64
  constexpr Family family = Family::x86_64;
#elif defined(__mips) || defined(__mips__) || defined(_M_MRX000)
#  define PLATFORM_CPU_MIPS
#  if defined(_MIPS_ARCH_MIPS64) || defined(__mips64)
#   define PLATFORM_CPU_MIPS_64
    constexpr Family family = Family::Mips64;
#  else
#   define PLATFORM_CPU_MIPS_32
    constexpr Family family = Family::Mips32;
#  endif
#else
#  error "unknown target cpu architecture"
#endif

constexpr Architecture architecture = 
  (family == Family::i386 || family == Family::x86_32 || family == Family::x86_64) ? Architecture::x86 :
  (family == Family::Arm32 || family == Family::Arm64) ? Architecture::Arm :
  (family == Family::Mips32 || family == Family::Mips64) ? Architecture::Mips :
  Architecture::Unknown;

#if defined(__mips) || defined(__mips__) || defined(_M_MRX000)
#  if defined(__MIPSEL__)
    constexpr ByteOrder byteOrder = LittleEndian;
#  elif defined(__MIPSEB__)
    constexpr ByteOrder byteOrder = BigEndian;
#  endif
#else
constexpr ByteOrder byteOrder = LittleEndian;
#endif


#if defined(PLATFORM_CPU_X86_64) || defined(PLATFORM_CPU_ARM_64) || defined(PLATFORM_CPU_MIPS_64)
#define PLATFORM_CPU_WORD_BITS 64
#else
#define PLATFORM_CPU_WORD_BITS 32
#endif

constexpr int wordBits = (family == Family::x86_64 || family == Family::Arm64 || family == Family::Mips64) ? 64 : 32;

namespace Feature
{
#if defined(PLATFORM_CPU_X86)
#  if defined(PLATFORM_COMPILER_MSVC) || defined(__SSE3__)
#    define PLATFORM_CPU_FEATURE_SSE3
     constexpr bool sse3 = true;
#  else
     constexpr bool sse3 = false;
#  endif
#  if defined(PLATFORM_COMPILER_MSVC) || defined(__SSSE3__)
#    define PLATFORM_CPU_FEATURE_SSSE3
     constexpr bool ssse3 = true;
#  else
     constexpr bool ssse3 = false;
#  endif
#  if defined(PLATFORM_COMPILER_MSVC) || defined(__SSE4_1__)
#    define PLATFORM_CPU_FEATURE_SSE41
     constexpr bool sse41 = true;
#  else
     constexpr bool sse41 = false;
#  endif
#  if defined(PLATFORM_COMPILER_MSVC) || defined(__AVX__)
#    define PLATFORM_CPU_FEATURE_AVX
     constexpr bool avx = true;
#  else
     constexpr bool avx = false;
#  endif
#  if defined(PLATFORM_COMPILER_MSVC) || defined(__AVX2__)
#    define PLATFORM_CPU_FEATURE_AVX2
     constexpr bool avx2 = true;
#  else
     constexpr bool avx2 = false;
#  endif
#endif
}

}

}
