#pragma once

#if defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)
#define PLATFORM_OS_WINDOWS
#elif defined(__linux__) || defined(__linux)
#define PLATFORM_OS_LINUX
#endif

#if defined(PLATFORM_OS_LINUX)
#define PLATFORM_OS_POSIX
#endif
