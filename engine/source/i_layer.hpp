#pragma once

#include <memory>

#include "i_neuron.hpp"

namespace cnn
{
  template <typename T>
  class ILayer
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<ILayer<T>>;

    virtual size_t GetNeuronCount() const = 0;
    
    virtual const INeuron<T>& GetNeuron(const size_t index) const = 0;
    virtual INeuron<T>& GetNeuron(const size_t index) = 0;

    virtual ~ILayer() {}

  };
}