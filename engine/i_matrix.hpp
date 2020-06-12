#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class IMatrix
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    virtual size_t GetCount() const = 0;

    virtual T GetCell(const size_t index) const = 0;
    virtual void SetCell(const size_t index, const T value) = 0;

    virtual ~IMatrix() {};

  };
}