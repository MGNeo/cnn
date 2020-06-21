#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class IActivator
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<IActivator<T>>;

    virtual T Handle(const T value) const = 0;

    virtual ~IActivator<T>() {};

  };
}