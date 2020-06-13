#pragma once

#include <memory>

namespace cnn
{
  template <typename T>
  class IMap
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<IMap<T>>;

    virtual size_t GetCount() const = 0;

    virtual T GetValue(const size_t index) const = 0;
    virtual void SetValue(const size_t index, const T value) = 0;

    virtual void Clear() = 0;

    virtual ~IMap() {};

  };
}