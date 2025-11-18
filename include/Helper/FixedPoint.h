#pragma once

#include <cstdint>

namespace Helper
{

template<typename BaseType, typename AccType, int fraction>
struct FixedPoint
{
  typedef BaseType Type;
  typedef AccType AccumulatorType;

  constexpr static int fractionBits = fraction;
  constexpr static Type oneHalf = (Type)1 << (fraction - 1);

  static constexpr Type fromFloat(float x);
  static constexpr Type fromDouble(double x);
  static constexpr Type fromInt32(int32_t x);
  static constexpr AccumulatorType accumulatorFromInt32(int32_t x);

  static constexpr float toFloat(Type x);
  static constexpr double toDouble(Type x);
  static constexpr int32_t toInt32(Type x);

  static constexpr Type round(AccumulatorType x);
};

template<typename BaseType, typename AccType, int fraction>
inline constexpr BaseType FixedPoint<BaseType, AccType, fraction>::fromFloat(float x)
{
  return (AccType)(x * 2 * ((AccType)1 << fractionBits) + 1) >> 1;
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr BaseType FixedPoint<BaseType, AccType, fraction>::fromDouble(double x)
{
  return (AccType)(x * 2 * ((AccType)1 << fractionBits) + 1) >> 1;
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr BaseType FixedPoint<BaseType, AccType, fraction>::fromInt32(int32_t x)
{
  return ((Type)x) << fractionBits;
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr AccType FixedPoint<BaseType, AccType, fraction>::accumulatorFromInt32(int32_t x)
{
  return ((AccumulatorType)x) << fractionBits;
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr float FixedPoint<BaseType, AccType, fraction>::toFloat(Type x)
{
  return (float)x / ((Type)1 << fractionBits);
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr double FixedPoint<BaseType, AccType, fraction>::toDouble(Type x)
{
  return (double)x / ((Type)1 << fractionBits);
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr int32_t FixedPoint<BaseType, AccType, fraction>::toInt32(Type x)
{
  return x >> fractionBits;
}

template<typename BaseType, typename AccType, int fraction>
inline constexpr BaseType FixedPoint<BaseType, AccType, fraction>::round(AccumulatorType x)
{
  return (x + oneHalf) >> fractionBits;
}

}
