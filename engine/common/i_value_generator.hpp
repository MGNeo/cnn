#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class IValueGenerator
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IValueGenerator<T>>;

        virtual T GetMinValue() const = 0;
        virtual T GetMaxValue() const = 0;

        virtual T Generate() = 0;

        virtual ~IValueGenerator() = default;

      };
    }
  }
}