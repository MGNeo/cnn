#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class ILesson2D
  {
    
    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<ILesson2D<T>>;

    virtual size_t GetInputWidth() const = 0;
    virtual size_t GetIntputHeight() const = 0;

    virtual size_t GetOutputWidth() const = 0;
    virtual size_t GetOutputHeight() const = 0;

    virtual T GetInput(const size_t x, const size_t y) const = 0;
    virtual void SetInput(const size_t x, const size_t y, const T value) = 0;

    virtual T GetOutput(const size_t x, const size_t y) const = 0;
    virtual void SetOutput(const size_t x, const size_t y, const T value) = 0;

    virtual ~ILesson2D() {};

  };
}