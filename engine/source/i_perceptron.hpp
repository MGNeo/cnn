#pragma once

#include <memory>

#include "i_layer.hpp"

namespace cnn
{
  template <typename T>
  class IPerceptron
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<IPerceptron<T>>;

    virtual size_t GetInputCount() const = 0;

    virtual T GetInput(const size_t index) const = 0;
    virtual void SetInput(const size_t index, const T value) = 0;

    virtual size_t GetLayerCount() const = 0;

    virtual const ILayer<T>& GetLayer(const size_t index) const = 0;
    virtual ILayer<T>& GetLayer(const size_t index) = 0;

    virtual void PushLayer(const size_t neuronCount) = 0;

    virtual void Process() = 0;

    virtual ~IPerceptron() {};

  };
}