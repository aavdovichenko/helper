#pragma once

#include "simd_int_sse.h"

namespace Platform
{

namespace Cpu
{

template<>
struct SseSimdIntType<int8_t> : public BaseSseSimdIntType<int8_t, SseSimdIntType<int8_t>>
{
  using BaseSseSimdIntType<int8_t, SseSimdIntType<int8_t>>::BaseSseSimdIntType;

  static SseSimdIntType<int8_t> populate(int8_t value);

  SseSimdIntType<int8_t> operator+(SseSimdIntType<int8_t> other) const;
  SseSimdIntType<int8_t> operator-(SseSimdIntType<int8_t> other) const;
  SseSimdIntType<int8_t>& operator+=(SseSimdIntType<int8_t> other);

  SseSimdIntConditionType<int8_t> operator==(const SseSimdIntType<int8_t>& other) const;
};

template<>
struct SIMD<int8_t, 16> : public SseIntSimd<int8_t>
{
  template<int dstStride = 1>
  static inline void transpose2x8x8(Type* dst, Type w0, Type w1, Type w2, Type w3, Type w4, Type w5, Type w6, Type w7);
};

inline SseSimdIntType<int8_t> SseSimdIntType<int8_t>::populate(int8_t value)
{
  return _mm_set1_epi8(value);
}

inline SseSimdIntType<int8_t> SseSimdIntType<int8_t>::operator+(SseSimdIntType<int8_t> other) const
{
  return _mm_add_epi8(value, other.value);
}

inline SseSimdIntType<int8_t> SseSimdIntType<int8_t>::operator-(SseSimdIntType<int8_t> other) const
{
  return _mm_sub_epi8(value, other.value);
}

inline SseSimdIntType<int8_t>& SseSimdIntType<int8_t>::operator+=(SseSimdIntType<int8_t> other)
{
  value = _mm_add_epi8(value, other.value);
  return *this;
}

inline SseSimdIntConditionType<int8_t> SseSimdIntType<int8_t>::operator==(const SseSimdIntType<int8_t>& other) const
{
  return SseSimdIntConditionType<int8_t>::fromNativeType(_mm_cmpeq_epi8(value, other.value));
}

template<int dstStride>
inline void SIMD<int8_t, 16>::transpose2x8x8(Type* dst, Type w0, Type w1, Type w2, Type w3, Type w4, Type w5, Type w6, Type w7)
{
  // 00 01 02 03 04 05 06 07 | 08 09 0A 0B 0C 0D 0E 0F
  // 10 11 12 13 14 15 16 17 | 18 19 1A 1B 1C 1D 1E 1F
  // 20 21 22 23 24 25 26 27 | 28 29 2A 2B 2C 2D 2E 2F
  // 30 31 32 33 34 35 36 37 | 38 39 3A 3B 3C 3D 3E 3F
  // 40 41 42 43 44 45 46 47 | 48 49 4A 4B 4C 4D 4E 4F
  // 50 51 52 53 54 55 56 57 | 58 59 5A 5B 5C 5D 5E 5F
  // 60 61 62 63 64 65 66 67 | 68 69 6A 6B 6C 6D 6E 6F
  // 70 71 72 73 74 75 76 77 | 78 79 7A 7B 7C 7D 7E 7F

  __m128i tmp0 = _mm_unpacklo_epi8(w0, w1); // 00 10 01 11 02 12 03 13 | 04 14 05 15 06 16 07 17
  __m128i tmp1 = _mm_unpackhi_epi8(w0, w1); // 08 18 09 19 0A 1A 0B 1B | 0C 1C 0D 1D 0E 1E 0F 1F
  __m128i tmp2 = _mm_unpacklo_epi8(w2, w3); // 20 30 21 31 22 32 23 33 | 24 34 25 35 26 36 27 37
  __m128i tmp3 = _mm_unpackhi_epi8(w2, w3); // 28 38 29 39 2A 3A 2B 3B | 2C 3C 2D 3D 2E 3E 2F 3F
  __m128i tmp4 = _mm_unpacklo_epi8(w4, w5); // 40 50 41 51 42 52 43 53 | 44 54 45 55 46 56 47 57
  __m128i tmp5 = _mm_unpackhi_epi8(w4, w5); // 48 58 49 59 4A 5A 4B 5B | 4C 5C 4D 5D 4E 5E 4F 5F
  __m128i tmp6 = _mm_unpacklo_epi8(w6, w7); // 60 70 61 71 62 72 63 73 | 64 74 65 75 66 76 67 77
  __m128i tmp7 = _mm_unpackhi_epi8(w6, w7); // 68 78 69 79 6A 7A 6B 7B | 6C 7C 6D 7D 6E 7E 6F 7F

  w0 = _mm_unpacklo_epi16(tmp0, tmp2); // 00 10 20 30 01 11 21 31 | 02 12 22 32 03 13 23 33
  w1 = _mm_unpackhi_epi16(tmp0, tmp2); // 04 14 24 34 05 15 25 35 | 06 16 26 36 07 17 27 37
  w2 = _mm_unpacklo_epi16(tmp1, tmp3); // 08 18 28 38 09 19 29 39 | 0A 1A 2A 3A 0B 1B 2B 3B
  w3 = _mm_unpackhi_epi16(tmp1, tmp3); // 0C 1C 2C 3C 0D 1D 2D 3D | 0E 1E 2E 3E 0F 1F 2F 3F
  w4 = _mm_unpacklo_epi16(tmp4, tmp6); // 40 50 60 70 41 51 61 71 | 42 52 62 72 43 53 63 73
  w5 = _mm_unpackhi_epi16(tmp4, tmp6); // 44 54 64 74 45 55 65 75 | 46 56 66 76 47 57 67 77
  w6 = _mm_unpacklo_epi16(tmp5, tmp7); // 48 58 68 78 49 59 69 79 | 4A 5A 6A 7A 4B 5B 6B 7B
  w7 = _mm_unpackhi_epi16(tmp5, tmp7); // 4C 5C 6C 7C 4D 5D 6D 7D | 4E 5E 6E 7E 4F 5F 6F 7F

  tmp0 = _mm_unpacklo_epi32(w0, w4); // 00 10 20 30 40 50 60 70 | 01 11 21 31 41 51 61 71
  tmp1 = _mm_unpackhi_epi32(w0, w4); // 02 12 22 32 42 52 62 72 | 03 13 23 33 43 53 63 73
  tmp2 = _mm_unpacklo_epi32(w1, w5); // 04 14 24 34 44 54 64 74 | 05 15 25 35 45 55 65 75
  tmp3 = _mm_unpackhi_epi32(w1, w5); // 06 16 26 36 46 56 66 76 | 07 17 27 37 47 57 67 77
  tmp4 = _mm_unpacklo_epi32(w2, w6); // 08 18 28 38 48 58 68 78 | 09 19 29 39 49 59 69 79
  tmp5 = _mm_unpackhi_epi32(w2, w6); // 0A 1A 2A 3A 4A 5A 6A 7A | 0B 1B 2B 3B 4B 5B 6B 7B
  tmp6 = _mm_unpacklo_epi32(w3, w7); // 0C 1C 2C 3C 4C 5C 6C 7C | 0D 1D 2D 3D 4D 5D 6D 7D
  tmp7 = _mm_unpackhi_epi32(w3, w7); // 0E 1E 2E 3E 4E 5E 6E 7E | 0F 1F 2F 3F 4F 5F 6F 7F

  dst[0 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp0), _mm_castsi128_pd(tmp4), 0x0)); // 00 10 20 30 40 50 60 70 | 08 18 28 38 48 58 68 78
  dst[1 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp0), _mm_castsi128_pd(tmp4), 0x3)); // 01 11 21 31 41 51 61 71 | 09 19 29 39 49 59 69 79
  dst[2 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp1), _mm_castsi128_pd(tmp5), 0x0)); // 02 12 22 32 42 52 62 72 | 0A 1A 2A 3A 4A 5A 6A 7A
  dst[3 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp1), _mm_castsi128_pd(tmp5), 0x3)); // 03 13 23 33 43 53 63 73 | 0B 1B 2B 3B 4B 5B 6B 7B
  dst[4 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp2), _mm_castsi128_pd(tmp6), 0x0)); // 04 14 24 34 44 54 64 74 | 0C 1C 2C 3C 4C 5C 6C 7C
  dst[5 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp2), _mm_castsi128_pd(tmp6), 0x3)); // 05 15 25 35 45 55 65 75 | 0D 1D 2D 3D 4D 5D 6D 7D
  dst[6 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp3), _mm_castsi128_pd(tmp7), 0x0)); // 06 16 26 36 46 56 66 76 | 0E 1E 2E 3E 4E 5E 6E 7E
  dst[7 * dstStride] = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(tmp3), _mm_castsi128_pd(tmp7), 0x3)); // 07 17 27 37 47 57 67 77 | 0F 1F 2F 3F 4F 5F 6F 7F
}

}

}
