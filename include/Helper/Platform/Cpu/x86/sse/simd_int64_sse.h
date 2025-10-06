#pragma once

#include "simd_int_sse.h"

#define PLATFORM_CPU_FEATURE_INT64x2

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<int64_t> : public BaseSseSimdIntType<int64_t, SseSimdIntType<int64_t>>
{
  using BaseSseSimdIntType<int64_t, SseSimdIntType<int64_t>>::BaseSseSimdIntType;

  template<bool aligned> static inline SseSimdIntType<int64_t> loadLowWord(const int64_t* src);
  static inline SseSimdIntType<int64_t> loadLowWord(const int64_t* src);
  static inline SseSimdIntType<int64_t> loadLowWord(__m128i dst, const int64_t* src);

  template <int i>
  int64_t get() const;

  template <int i0, int i1>
  inline SseSimdIntType<int64_t> shuffled() const;
  template <int i0, int i1>
  static inline SseSimdIntType<int64_t> shuffle(__m128i a, __m128i b);

  inline SseSimdIntType<int64_t> revertedByteOrder() const;
};

template<>
struct SIMD<int64_t, 2> : public SseIntSimd<int64_t>
{
};

// implemetation

template<int i>
inline int64_t SseSimdIntType<int64_t>::get() const
{
  static_assert(i == 0 || i == 1, "invalid index");
  return i == 0 ? _mm_cvtsi128_si64(value) : _mm_extract_epi64(value, 1);
}

template<int i0, int i1>
inline SseSimdIntType<int64_t> SseSimdIntType<int64_t>::shuffled() const
{
  static_assert(i0 >= 0 && i0 <= 1 && i1 >= 0 && i1 <= 1, "invalid index");
  return SseSimdIntType<int64_t>{_mm_shuffle_epi32(value, (i0 == 0 ? 0x04 : 0xe) | (i1 == 0 ? 0x40 : 0xe0))};
}

template<int i0, int i1>
inline SseSimdIntType<int64_t> SseSimdIntType<int64_t>::shuffle(__m128i a, __m128i b)
{
  static_assert(i0 >= 0 && i0 <= 3 && i1 >= 0 && i1 <= 3, "invalid index");

  if (i0 == 0 && i1 == 3)
    return SseSimdIntType<int64_t>{_mm_castpd_si128(_mm_move_sd(_mm_castsi128_pd(b), _mm_castsi128_pd(a)))};
  if (i0 == 3 && i1 == 0)
    return SseSimdIntType<int64_t>{_mm_castpd_si128(_mm_move_sd(_mm_castsi128_pd(a), _mm_castsi128_pd(a)))};

  constexpr int imm8 = ((i1 < 2 ? i1 : i1 - 2) << 1) | (i0 < 2 ? i0 : i0 - 2);
  return SseSimdIntType<int64_t>{_mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(i0 < 2 ? a : b), _mm_castsi128_pd(i1 < 2 ? a : b), imm8))};
}

inline SseSimdIntType<int64_t> SseSimdIntType<int64_t>::revertedByteOrder() const
{
  __m128i indices = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
  return _mm_shuffle_epi8(value, indices);
}

template<bool aligned>
inline SseSimdIntType<int64_t> SseSimdIntType<int64_t>::loadLowWord(const int64_t* src)
{
  return SseSimdIntType<int64_t>::fromNativeType(aligned ? _mm_loadl_epi64((const __m128i*)src) : _mm_loadl_epi64((const __m128i*)src));
}

inline SseSimdIntType<int64_t> SseSimdIntType<int64_t>::loadLowWord(const int64_t* src)
{
  return SseSimdIntType<int64_t>::fromNativeType(_mm_loadl_epi64((const __m128i*)src));
}

inline SseSimdIntType<int64_t> SseSimdIntType<int64_t>::loadLowWord(__m128i dst, const int64_t* src)
{
  return SseSimdIntType<int64_t>::fromNativeType(_mm_castps_si128(_mm_loadl_pi(_mm_castsi128_ps(dst), (const __m64*)src)));
}

}

}
