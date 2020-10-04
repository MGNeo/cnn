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
      class IActivationFunction
      {
      public:

        using Uptr = std::unique_ptr<IActivationFunction<T>>;

        virtual T Use(const T value) const = 0;

        virtual typename IActivationFunction<T>::Uptr Clone() const = 0;

        virtual ~IActivationFunction() = default;

      };
    }
  }
}