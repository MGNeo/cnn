#pragma once

#include "i_value_generator.hpp"

#include <random>
#include <time.h>
#include <chrono>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class ValueGenerator : public IValueGenerator<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ValueGenerator<T>>;

        ValueGenerator(const T minValue, const T maxValue);

        T GetMinValue() const override;
        T GetMaxValue() const override;

        T Generate() override;

      private:
        
        std::uniform_real_distribution<T> UDR;
        std::default_random_engine DRE;

      };

      template <typename T>
      ValueGenerator<T>::ValueGenerator(const T minValue, const T maxValue)
        :
        UDR{ minValue, maxValue },
        DRE{ static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) }
      {
      }

      template <typename T>
      T ValueGenerator<T>::GetMinValue() const
      {
        return UDR.min();
      }

      template <typename T>
      T ValueGenerator<T>::GetMaxValue() const
      {
        return UDR.max();
      }

      template <typename T>
      T ValueGenerator<T>::Generate()
      {
        return UDR(DRE);
      }
    }
  }
}