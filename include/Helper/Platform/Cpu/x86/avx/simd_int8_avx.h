#pragma once

#include "simd_int_avx.h"

#define PLATFORM_CPU_FEATURE_INT8x32

namespace Platform
{

namespace Cpu
{

template<>
struct AvxSimdIntType<int8_t> : public BaseAvxSimdIntType<int8_t, AvxSimdIntType<int8_t>>
{
  using BaseAvxSimdIntType<int8_t, AvxSimdIntType<int8_t>>::BaseAvxSimdIntType;

  static AvxSimdIntType<int8_t> populate(int8_t value);
  static inline AvxSimdIntType<int8_t> create(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7,
    int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15,
    int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23, 
    int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31);

  template<uint8_t countLo, int8_t lo, int8_t hi>
  static AvxSimdIntType<int8_t> createWith2Runs();

  AvxSimdIntType<int8_t> operator+(AvxSimdIntType<int8_t> other) const;
  AvxSimdIntType<int8_t> operator-(AvxSimdIntType<int8_t> other) const;
  AvxSimdIntType<int8_t>& operator+=(AvxSimdIntType<int8_t> other);

  AvxSimdIntType<int8_t> operator<<(int count) const;

  AvxSimdIntConditionType<int8_t> operator==(const AvxSimdIntType<int8_t>& other) const;
  AvxSimdIntConditionType<int8_t> operator<(const AvxSimdIntType<int8_t>& other) const;
  AvxSimdIntConditionType<int8_t> operator>(const AvxSimdIntType<int8_t>& other) const;
};

template<>
struct SIMD<int8_t, 32> : public AvxIntSimd<int8_t>
{
  static AvxSimdIntType<int8_t> populate(int8_t value);

  template<uint8_t count, int8_t padding = 0>
  static Type shiftItemsLeft(Type value);
  template<uint8_t count>
  static Type shiftItemsLeft(Type value, Type carry);

  template<bool dstAligned, bool srcAligned>
  static inline void transpose(int8_t* dst, size_t dstStride, const int8_t* src, size_t srcStride);

  template<int dstStride = 1>
  static inline void transpose4x8x8(Type* dst, Type w0, Type w1, Type w2, Type w3, Type w4, Type w5, Type w6, Type w7);

  static Type create4BitLookupTable(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7,
    int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15);
  static Type lookup4BitKeyValues(Type keys, Type table);
};

// AvxSimdIntType<int8_t>

inline AvxSimdIntType<int8_t> AvxSimdIntType<int8_t>::populate(int8_t value)
{
  return AvxSimdIntType<int8_t>{_mm256_set1_epi8(value)};
}

inline AvxSimdIntType<int8_t> AvxSimdIntType<int8_t>::create(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7,
  int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15,
  int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23,
  int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31)
{
  return AvxSimdIntType<int8_t>{_mm256_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31)};
}

template<uint8_t n, int8_t l, int8_t r>
inline AvxSimdIntType<int8_t> AvxSimdIntType<int8_t>::createWith2Runs()
{
  static_assert(n > 0 && n < 32, "invalid value");
  return _mm256_setr_epi8(0 < n ? l : r, 1 < n ? l : r, 2 < n ? l : r, 3 < n ? l : r, 4 < n ? l : r, 5 < n ? l : r, 6 < n ? l : r, 7 < n ? l : r,
    8 < n ? l : r, 9 < n ? l : r, 10 < n ? l : r, 11 < n ? l : r, 12 < n ? l : r, 13 < n ? l : r, 14 < n ? l : r, 15 < n ? l : r,
    16 < n ? l : r, 17 < n ? l : r, 18 < n ? l : r, 19 < n ? l : r, 20 < n ? l : r, 21 < n ? l : r, 22 < n ? l : r, 23 < n ? l : r,
    24 < n ? l : r, 25 < n ? l : r, 26 < n ? l : r, 27 < n ? l : r, 28 < n ? l : r, 29 < n ? l : r, 30 < n ? l : r, 31 < n ? l : r);
}

inline AvxSimdIntType<int8_t> AvxSimdIntType<int8_t>::operator+(AvxSimdIntType<int8_t> other) const
{
  return AvxSimdIntType<int8_t>{_mm256_add_epi8(value, other.value)};
}

inline AvxSimdIntType<int8_t> AvxSimdIntType<int8_t>::operator-(AvxSimdIntType<int8_t> other) const
{
  return AvxSimdIntType<int8_t>{_mm256_sub_epi8(value, other.value)};
}

inline AvxSimdIntType<int8_t>& AvxSimdIntType<int8_t>::operator+=(AvxSimdIntType<int8_t> other)
{
  value = _mm256_add_epi8(value, other.value);
  return *this;
}

inline AvxSimdIntType<int8_t> AvxSimdIntType<int8_t>::operator<<(int count) const
{
  return AvxSimdIntType<int8_t>::fromNativeType(_mm256_and_si256(_mm256_slli_epi16(value, count), _mm256_set1_epi8((uint8_t)0xff << count)));
}

inline AvxSimdIntConditionType<int8_t> AvxSimdIntType<int8_t>::operator==(const AvxSimdIntType<int8_t>& other) const
{
  return AvxSimdIntConditionType<int8_t>::fromNativeType(_mm256_cmpeq_epi8(value, other.value));
}

inline AvxSimdIntConditionType<int8_t> AvxSimdIntType<int8_t>::operator<(const AvxSimdIntType<int8_t>& other) const
{
  return AvxSimdIntConditionType<int8_t>::fromNativeType(_mm256_cmpgt_epi8(other.value, value));
}

inline AvxSimdIntConditionType<int8_t> AvxSimdIntType<int8_t>::operator>(const AvxSimdIntType<int8_t>& other) const
{
  return AvxSimdIntConditionType<int8_t>::fromNativeType(_mm256_cmpgt_epi8(value, other.value));
}

// SIMD<int8_t, 32>

inline AvxSimdIntType<int8_t> SIMD<int8_t, 32>::populate(int8_t value)
{
  return AvxSimdIntType<int8_t>{_mm256_set1_epi8(value)};
}

template<uint8_t count, int8_t padding>
inline typename SIMD<int8_t, 32>::Type SIMD<int8_t, 32>::shiftItemsLeft(Type value)
{
  static_assert(count < 16, "not implemented");
  __m256i mask = Type::createWith2Runs<count, -1, 0>().value;
  __m256i carry = _mm256_and_si256(_mm256_srli_si256(value.value, 16 - count), mask);
  __m256i shifted = _mm256_or_si256(_mm256_slli_si256(value.value, count), _mm256_permute4x64_epi64(carry, _MM_SHUFFLE(1, 0, 2, 2)));
  if (padding == 0)
    return shifted;
  if (count == 1)
    return _mm256_or_si256(shifted, padding == -1 ? mask : _mm256_set1_epi8(padding));

  return _mm256_or_si256(shifted, Type::createWith2Runs<count, padding, 0>().value);
}

template<uint8_t count>
inline typename SIMD<int8_t, 32>::Type SIMD<int8_t, 32>::shiftItemsLeft(Type value, Type carry)
{
  static_assert(count < 16, "not implemented");
  __m256i mask = Type::createWith2Runs<32 - count, 0, -1>().value;
  __m256i c = _mm256_srli_si256(_mm256_and_si256(carry.value, mask), 16 - count);
  return _mm256_or_si256(shiftItemsLeft<count>(value).value, _mm256_permute4x64_epi64(c, _MM_SHUFFLE(0, 0, 3, 2)));
}

template<bool dstAligned, bool srcAligned>
inline void SIMD<int8_t, 32>::transpose(int8_t* dst, size_t dstStride, const int8_t* src, size_t srcStride)
{
  Type w0 = load<srcAligned>(src + 0 * srcStride), w1 = load<srcAligned>(src + 1 * srcStride);
  Type w2 = load<srcAligned>(src + 2 * srcStride), w3 = load<srcAligned>(src + 3 * srcStride);
  Type w4 = load<srcAligned>(src + 4 * srcStride), w5 = load<srcAligned>(src + 5 * srcStride);
  Type w6 = load<srcAligned>(src + 6 * srcStride), w7 = load<srcAligned>(src + 7 * srcStride);
  Type w8 = load<srcAligned>(src + 8 * srcStride), w9 = load<srcAligned>(src + 9 * srcStride);
  Type wA = load<srcAligned>(src + 10 * srcStride), wB = load<srcAligned>(src + 11 * srcStride);
  Type wC = load<srcAligned>(src + 12 * srcStride), wD = load<srcAligned>(src + 13 * srcStride);
  Type wE = load<srcAligned>(src + 14 * srcStride), wF = load<srcAligned>(src + 15 * srcStride);
  Type wG = load<srcAligned>(src + 16 * srcStride), wH = load<srcAligned>(src + 17 * srcStride);
  Type wI = load<srcAligned>(src + 18 * srcStride), wJ = load<srcAligned>(src + 19 * srcStride);
  Type wK = load<srcAligned>(src + 20 * srcStride), wL = load<srcAligned>(src + 21 * srcStride);
  Type wM = load<srcAligned>(src + 22 * srcStride), wN = load<srcAligned>(src + 23 * srcStride);
  Type wO = load<srcAligned>(src + 24 * srcStride), wP = load<srcAligned>(src + 25 * srcStride);
  Type wQ = load<srcAligned>(src + 26 * srcStride), wR = load<srcAligned>(src + 27 * srcStride);
  Type wS = load<srcAligned>(src + 28 * srcStride), wT = load<srcAligned>(src + 29 * srcStride);
  Type wU = load<srcAligned>(src + 30 * srcStride), wV = load<srcAligned>(src + 31 * srcStride);

  transposeAvxInt8(w0.value, w1.value, w2.value, w3.value, w4.value, w5.value, w6.value, w7.value, w8.value, w9.value, wA.value, wB.value, wC.value, wD.value, wE.value, wF.value,
    wG.value, wH.value, wI.value, wJ.value, wK.value, wL.value, wM.value, wN.value, wO.value, wP.value, wQ.value, wR.value, wS.value, wT.value, wU.value, wV.value);

  w0.store<dstAligned>(dst + 0 * dstStride); w1.store<dstAligned>(dst + 1 * dstStride);
  w2.store<dstAligned>(dst + 2 * dstStride); w3.store<dstAligned>(dst + 3 * dstStride);
  w4.store<dstAligned>(dst + 4 * dstStride); w5.store<dstAligned>(dst + 5 * dstStride);
  w6.store<dstAligned>(dst + 6 * dstStride); w7.store<dstAligned>(dst + 7 * dstStride);
  w8.store<dstAligned>(dst + 8 * dstStride); w9.store<dstAligned>(dst + 9 * dstStride);
  wA.store<dstAligned>(dst + 10 * dstStride); wB.store<dstAligned>(dst + 11 * dstStride);
  wC.store<dstAligned>(dst + 12 * dstStride); wD.store<dstAligned>(dst + 13 * dstStride);
  wE.store<dstAligned>(dst + 14 * dstStride); wF.store<dstAligned>(dst + 15 * dstStride);
  wG.store<dstAligned>(dst + 16 * dstStride); wH.store<dstAligned>(dst + 17 * dstStride);
  wI.store<dstAligned>(dst + 18 * dstStride); wJ.store<dstAligned>(dst + 19 * dstStride);
  wK.store<dstAligned>(dst + 20 * dstStride); wL.store<dstAligned>(dst + 21 * dstStride);
  wM.store<dstAligned>(dst + 22 * dstStride); wN.store<dstAligned>(dst + 23 * dstStride);
  wO.store<dstAligned>(dst + 24 * dstStride); wP.store<dstAligned>(dst + 25 * dstStride);
  wQ.store<dstAligned>(dst + 26 * dstStride); wR.store<dstAligned>(dst + 27 * dstStride);
  wS.store<dstAligned>(dst + 28 * dstStride); wT.store<dstAligned>(dst + 29 * dstStride);
  wU.store<dstAligned>(dst + 30 * dstStride); wV.store<dstAligned>(dst + 31 * dstStride);
}

template<int dstStride>
inline void SIMD<int8_t, 32>::transpose4x8x8(Type* dst, Type w0, Type w1, Type w2, Type w3, Type w4, Type w5, Type w6, Type w7)
{
  // 00 01 02 03 04 05 06 07 | 08 09 0A 0B 0C 0D 0E 0F
  // 10 11 12 13 14 15 16 17 | 18 19 1A 1B 1C 1D 1E 1F
  // 20 21 22 23 24 25 26 27 | 28 29 2A 2B 2C 2D 2E 2F
  // 30 31 32 33 34 35 36 37 | 38 39 3A 3B 3C 3D 3E 3F
  // 40 41 42 43 44 45 46 47 | 48 49 4A 4B 4C 4D 4E 4F
  // 50 51 52 53 54 55 56 57 | 58 59 5A 5B 5C 5D 5E 5F
  // 60 61 62 63 64 65 66 67 | 68 69 6A 6B 6C 6D 6E 6F
  // 70 71 72 73 74 75 76 77 | 78 79 7A 7B 7C 7D 7E 7F

  __m256i tmp0 = _mm256_unpacklo_epi8(w0, w1); // 00 10 01 11 02 12 03 13 | 04 14 05 15 06 16 07 17
  __m256i tmp1 = _mm256_unpackhi_epi8(w0, w1); // 08 18 09 19 0A 1A 0B 1B | 0C 1C 0D 1D 0E 1E 0F 1F
  __m256i tmp2 = _mm256_unpacklo_epi8(w2, w3); // 20 30 21 31 22 32 23 33 | 24 34 25 35 26 36 27 37
  __m256i tmp3 = _mm256_unpackhi_epi8(w2, w3); // 28 38 29 39 2A 3A 2B 3B | 2C 3C 2D 3D 2E 3E 2F 3F
  __m256i tmp4 = _mm256_unpacklo_epi8(w4, w5); // 40 50 41 51 42 52 43 53 | 44 54 45 55 46 56 47 57
  __m256i tmp5 = _mm256_unpackhi_epi8(w4, w5); // 48 58 49 59 4A 5A 4B 5B | 4C 5C 4D 5D 4E 5E 4F 5F
  __m256i tmp6 = _mm256_unpacklo_epi8(w6, w7); // 60 70 61 71 62 72 63 73 | 64 74 65 75 66 76 67 77
  __m256i tmp7 = _mm256_unpackhi_epi8(w6, w7); // 68 78 69 79 6A 7A 6B 7B | 6C 7C 6D 7D 6E 7E 6F 7F

  w0 = _mm256_unpacklo_epi16(tmp0, tmp2); // 00 10 20 30 01 11 21 31 | 02 12 22 32 03 13 23 33
  w1 = _mm256_unpackhi_epi16(tmp0, tmp2); // 04 14 24 34 05 15 25 35 | 06 16 26 36 07 17 27 37
  w2 = _mm256_unpacklo_epi16(tmp1, tmp3); // 08 18 28 38 09 19 29 39 | 0A 1A 2A 3A 0B 1B 2B 3B
  w3 = _mm256_unpackhi_epi16(tmp1, tmp3); // 0C 1C 2C 3C 0D 1D 2D 3D | 0E 1E 2E 3E 0F 1F 2F 3F
  w4 = _mm256_unpacklo_epi16(tmp4, tmp6); // 40 50 60 70 41 51 61 71 | 42 52 62 72 43 53 63 73
  w5 = _mm256_unpackhi_epi16(tmp4, tmp6); // 44 54 64 74 45 55 65 75 | 46 56 66 76 47 57 67 77
  w6 = _mm256_unpacklo_epi16(tmp5, tmp7); // 48 58 68 78 49 59 69 79 | 4A 5A 6A 7A 4B 5B 6B 7B
  w7 = _mm256_unpackhi_epi16(tmp5, tmp7); // 4C 5C 6C 7C 4D 5D 6D 7D | 4E 5E 6E 7E 4F 5F 6F 7F

  tmp0 = _mm256_unpacklo_epi32(w0, w4); // 00 10 20 30 40 50 60 70 | 01 11 21 31 41 51 61 71
  tmp1 = _mm256_unpackhi_epi32(w0, w4); // 02 12 22 32 42 52 62 72 | 03 13 23 33 43 53 63 73
  tmp2 = _mm256_unpacklo_epi32(w1, w5); // 04 14 24 34 44 54 64 74 | 05 15 25 35 45 55 65 75
  tmp3 = _mm256_unpackhi_epi32(w1, w5); // 06 16 26 36 46 56 66 76 | 07 17 27 37 47 57 67 77
  tmp4 = _mm256_unpacklo_epi32(w2, w6); // 08 18 28 38 48 58 68 78 | 09 19 29 39 49 59 69 79
  tmp5 = _mm256_unpackhi_epi32(w2, w6); // 0A 1A 2A 3A 4A 5A 6A 7A | 0B 1B 2B 3B 4B 5B 6B 7B
  tmp6 = _mm256_unpacklo_epi32(w3, w7); // 0C 1C 2C 3C 4C 5C 6C 7C | 0D 1D 2D 3D 4D 5D 6D 7D
  tmp7 = _mm256_unpackhi_epi32(w3, w7); // 0E 1E 2E 3E 4E 5E 6E 7E | 0F 1F 2F 3F 4F 5F 6F 7F

  dst[0 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp0), _mm256_castsi256_pd(tmp4), 0x0)); // 00 10 20 30 40 50 60 70 | 08 18 28 38 48 58 68 78
  dst[1 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp0), _mm256_castsi256_pd(tmp4), 0xf)); // 01 11 21 31 41 51 61 71 | 09 19 29 39 49 59 69 79
  dst[2 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp1), _mm256_castsi256_pd(tmp5), 0x0)); // 02 12 22 32 42 52 62 72 | 0A 1A 2A 3A 4A 5A 6A 7A
  dst[3 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp1), _mm256_castsi256_pd(tmp5), 0xf)); // 03 13 23 33 43 53 63 73 | 0B 1B 2B 3B 4B 5B 6B 7B
  dst[4 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp2), _mm256_castsi256_pd(tmp6), 0x0)); // 04 14 24 34 44 54 64 74 | 0C 1C 2C 3C 4C 5C 6C 7C
  dst[5 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp2), _mm256_castsi256_pd(tmp6), 0xf)); // 05 15 25 35 45 55 65 75 | 0D 1D 2D 3D 4D 5D 6D 7D
  dst[6 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp3), _mm256_castsi256_pd(tmp7), 0x0)); // 06 16 26 36 46 56 66 76 | 0E 1E 2E 3E 4E 5E 6E 7E
  dst[7 * dstStride].value = _mm256_castpd_si256(_mm256_shuffle_pd(_mm256_castsi256_pd(tmp3), _mm256_castsi256_pd(tmp7), 0xf)); // 07 17 27 37 47 57 67 77 | 0F 1F 2F 3F 4F 5F 6F 7F
}

inline typename SIMD<int8_t, 32>::Type SIMD<int8_t, 32>::create4BitLookupTable(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15)
{
  return _mm256_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
}

inline typename SIMD<int8_t, 32>::Type SIMD<int8_t, 32>::lookup4BitKeyValues(Type keys, Type table)
{
  return _mm256_shuffle_epi8(table, keys);
}

}

}
