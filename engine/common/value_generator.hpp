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

        ValueGenerator();

        T GetMinValue() const override;
        void SetMinValue(const T minValue) override;

        T GetMaxValue() const override;
        void SetMaxValue(const T maxValue) override;

        T Generate() override;

        typename IValueGenerator<T>::Uptr Clone() const override;

      private:
        
        T MinValue;
        T MaxValue;

        std::default_random_engine DRE;

      };

      template <typename T>
      ValueGenerator<T>::ValueGenerator()
        :
        MinValue{ static_cast<T>(-1L) },
        MaxValue{ static_cast<T>(+1L) },
        DRE{ static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) }
      {
      }

      template <typename T>
      T ValueGenerator<T>::GetMinValue() const
      {
        return MinValue;
      }

      template <typename T>
      void ValueGenerator<T>::SetMinValue(const T minValue)
      {
        if (minValue >= MaxValue)
        {
          throw std::invalid_argument("cnn::engine::common::ValueGenerator::SetMinValue(), minValue >= MaxValue.");
        }
        MinValue = minValue;
      }

      template <typename T>
      T ValueGenerator<T>::GetMaxValue() const
      {
        return MaxValue;
      }

      template <typename T>
      void ValueGenerator<T>::SetMaxValue(const T maxValue)
      {
        if (maxValue <= MinValue)
        {
          throw std::invalid_argument("cnn::engine::common::ValueGenerator::SetMaxValue(), maxValue <= MinValue.");
        }
        MaxValue = maxValue;
      }

      template <typename T>
      T ValueGenerator<T>::Generate()
      {
        std::uniform_real_distribution<T> URD{ MinValue, MaxValue };
        return URD(DRE);
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