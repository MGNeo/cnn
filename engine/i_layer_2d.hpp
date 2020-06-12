#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class ILayer2D
  {

    static_assert(std::is_floating_point<T>::value);

  public:



    virtual ~ILayer2D() {};

  };
}