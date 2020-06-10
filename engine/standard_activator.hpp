#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

namespace cnn
{
  namespace engine
  {
    template <typename T>
    class StandardActivator
    {

      static_assert(std::is_floating_point<T>::value);

    public:

      static T Handle(const T value)
      {
        return  1 / (1 + exp(-value));
      }

    };
  }
}
