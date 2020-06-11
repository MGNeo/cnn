#pragma once

#include "i_activator.hpp"

namespace cnn
{
  template <typename T>
  class Activator : public IActivator<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    T Handle(const T value) override
    {
      return 1 / (1 + exp(-value));
    }

  };
}