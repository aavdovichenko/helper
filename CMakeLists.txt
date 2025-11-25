project(helper CXX)

include_directories("${helper_SOURCE_DIR}/include")

# interface headers
set(HELPER_SOURCES
    include/Helper/Platform/Cpu/emulated/math_emulated.h
    include/Helper/Platform/Cpu/emulated/rounding_emulated.h
    include/Helper/Platform/Cpu/cpu.h
    include/Helper/Platform/Cpu/intrinsics.h
    include/Helper/Platform/Cpu/math.h
    include/Helper/Platform/Cpu/rounding.h
    include/Helper/Platform/Cpu/simd.h
    include/Helper/Platform/Cpu/simd_condition.h
    include/Helper/Platform/Cpu/simd_int.h
    include/Helper/Platform/compiler.h
    include/Helper/Platform/os.h
    include/Helper/FixedPoint.h
    include/Helper/ThreadPool.h
)

# x86 cpu headers
list(APPEND HELPER_SOURCES
    include/Helper/Platform/Cpu/x86/math_x86.h
    include/Helper/Platform/Cpu/x86/rounding_x86.h
    include/Helper/Platform/Cpu/x86/avx/simd_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_float_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_int_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_int8_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_int16_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_int32_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_int64_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_int128_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_uint8_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_uint16_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_uint32_avx.h
    include/Helper/Platform/Cpu/x86/avx/simd_uint64_avx.h
    include/Helper/Platform/Cpu/x86/sse/simd_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_float_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_int_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_int8_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_int16_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_int32_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_int64_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_uint8_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_uint16_sse.h
    include/Helper/Platform/Cpu/x86/sse/simd_uint32_sse.h
    include/Helper/Platform/Cpu/x86/simd_x86.h
    include/Helper/Platform/Cpu/x86/x86.h
)

add_library(helper INTERFACE)
