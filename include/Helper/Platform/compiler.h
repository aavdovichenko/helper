#pragma once

#if defined(_MSC_VER)
#  define PLATFORM_COMPILER_MSVC
#elif defined(__GNUC__)
#  define PLATFORM_COMPILER_GNU
#endif

#ifdef __clang__
#  define PLATFORM_COMPILER_CLANG
#endif
