#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class ILayer2D
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<ILayer2D<T>>;

    virtual size_t GetWidth() const = 0;
    virtual size_t GetHeight() const = 0;

    virtual T GetCell(const size_t x, const size_t y) const = 0;
    virtual void SetCell(const size_t x, const size_t y, const T value) = 0;

    virtual ~ILayer2D() {};

  };
}