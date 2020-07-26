#pragma once

#include <memory>

#include "i_layer_2d.hpp"
#include "activator.hpp"

namespace cnn
{
  template <typename T>
  class INetwork2D
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<INetwork2D<T>>;

    virtual size_t GetInputCount() const = 0;
    virtual size_t GetInputWidth() const = 0;
    virtual size_t GetInputHeight() const = 0;

    virtual const IMap2D<T>& GetInput(const size_t index) const = 0;
    virtual IMap2D<T>& GetInput(const size_t index) = 0;

    virtual size_t GetLayerCount() const = 0;

    virtual const ILayer2D<T>& GetLayer(const size_t index) const = 0;
    virtual ILayer2D<T>& GetLayer(const size_t index) = 0;

    virtual const ILayer2D<T>& GetLastLayer() const = 0;
    virtual ILayer2D<T>& GetLastLayer() = 0;

    virtual void PushLayer(const size_t filterCount,
                           const size_t filterWidth,
                           const size_t filterHeight,
                           typename const IActivator<T>& activator) = 0;

    virtual void Process() = 0;

    virtual ~INetwork2D() {};

  };
}