#pragma once

#include "i_activator.hpp"

namespace cnn
{
  template <typename T>
  class Activator : public IActivator<T> // This is Sovetsky Vertuhan.

  {
  public:

    T Handle(const T value) override
    {
      return 1 / (1 + exp(-value));
    }

  };
}