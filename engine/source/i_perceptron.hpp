#pragma once

#include <memory>

//#include "i_layer.hpp"

namespace cnn
{
  template <typename T>
  class IPerceptron
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<IPerceptron<T>>;

    virtual size_t GetLayerCount() const = 0;

    virtual 

  };
}