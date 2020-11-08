#pragma once

#include <random>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class ValueGenerator
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ValueGenerator();

        ValueGenerator(const ValueGenerator& valueGenerator) = default;

        ValueGenerator(ValueGenerator&& valueGenerator) noexcept;

        ValueGenerator& operator=(const ValueGenerator& valueGenerator) = default;

        ValueGenerator& operator=(ValueGenerator&& valueGenerator) noexcept;

        T GetMinValue() const noexcept;

        // Exception guarantee: strong for this.
        void SetMinValue(const T minValue);

        T GetMaxValue() const noexcept;

        // Exception guarantee: strong for this.
        void SetMaxValue(const T maxValue);

        // We expect that the method throws an exception never.
        T Generate();

        // We expect that the method throws an exception never.
        void SetSeed(const unsigned int seed);

        // We expect that the method throws an exception never.
        void Clear();

      private:
        
        T MinValue;
        T MaxValue;

        std::default_random_engine DRE;

      };

      template <typename T>
      ValueGenerator<T>::ValueGenerator()
      {
        Clear();
      }

      template <typename T>
      ValueGenerator<T>::ValueGenerator(ValueGenerator&& valueGenerator) noexcept
        :
        MinValue{ valueGenerator.MinValue },
        MaxValue{ valueGenerator.MaxValue },
        DRE{ valueGenerator.DRE }
      {
        valueGenerator.Clear();
      }

      template <typename T>
      ValueGenerator<T>& ValueGenerator<T>::operator=(ValueGenerator&& valueGenerator) noexcept
      {
        if (this != &valueGenerator)
        {
          MinValue = valueGenerator.MinValue;
          MaxValue = valueGenerator.MaxValue;
          DRE = std::move(valueGenerator.DRE);

          valueGenerator.Clear();
        }
        return *this;
      }

      template <typename T>
      T ValueGenerator<T>::GetMinValue() const noexcept
      {
        return MinValue;
      }

      template <typename T>
      void ValueGenerator<T>::SetMinValue(const T minValue)
      {
        if (minValue > MaxValue)
        {
          throw std::invalid_argument("cnn::engine::common::ValueGenerator::SetMinValue(), minValue > MaxValue.");
        }
        MinValue = minValue;
      }

      template <typename T>
      T ValueGenerator<T>::GetMaxValue() const noexcept
      {
        return MaxValue;
      }

      template <typename T>
      void ValueGenerator<T>::SetMaxValue(const T maxValue)
      {
        if (maxValue < MinValue)
        {
          throw std::invalid_argument("cnn::engine::common::ValueGenerator::SetMaxValue(), maxValue < MinValue.");
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
      void ValueGenerator<T>::SetSeed(const unsigned int seed)
      {
        DRE.seed(seed);
      }

      template <typename T>
      void ValueGenerator<T>::Clear()
      {
        MinValue = static_cast<T>(0.L);
        MaxValue = static_cast<T>(0.L);
        DRE.seed(0);
      }
    }
  }
}