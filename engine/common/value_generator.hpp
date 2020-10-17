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

        ValueGenerator(const T minValue, const T maxValue);// TODO: We must not use the parameters in the constructor.

        T Generate() override;

        typename IValueGenerator<T>::Uptr Clone() const override;

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
      T ValueGenerator<T>::Generate()
      {
        return UDR(DRE);
      }

      template <typename T>
      typename IValueGenerator<T>::Uptr ValueGenerator<T>::Clone() const
      {
        auto valueGenerator = std::make_unique<ValueGenerator<T>>(*this);
        return valueGenerator;
      }

    }
  }
}