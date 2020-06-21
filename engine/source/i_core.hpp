#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class ICore
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<ICore<T>>;

    virtual size_t GetCount() const = 0;

    virtual T GetInput(const size_t index) const = 0;
    virtual void SetInput(const size_t index, const T value) = 0;

    virtual T GetWeight(const size_t index) const = 0;
    virtual void SetWeight(const size_t index, const T value) = 0;

    virtual void GenerateOutput() = 0;

    virtual T GetOutput() const = 0;

    virtual ~ICore() {};

  };
}