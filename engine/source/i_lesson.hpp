#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class ILesson
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<ILesson<T>>;

    virtual size_t GetInputCount() const = 0;
    virtual size_t GetOutputCount() const = 0;

    virtual T GetInput(const size_t index) const = 0;
    virtual void SetInput(const size_t index, const T value) = 0;

    virtual T GetOutput(const size_t index) const = 0;
    virtual void SetOutput(const size_t index, const T value) = 0;

    virtual ~ILesson() {};
  };
}