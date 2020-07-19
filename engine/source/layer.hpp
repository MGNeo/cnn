#pragma once

#include <stdexcept>

#include "i_layer.hpp"

namespace cnn
{
  template <typename T>
  class Layer : public ILayer<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Layer(const size_t neuronCount, const size_t inputCount);

    size_t GetNeuronCount() const override;

    const INeuron<T>& GetNeuron(const size_t index) const override;
    INeuron<T>& GetNeuron(const size_t index) override;

  private:

    const size_t NeuronCount;
    const std::unique_ptr<typename INeuron<T>::Uptr[]> Neurons;

  };

  template <typename T>
  Layer<T>::Layer(const size_t neuronCount, const size_t inputCount)
    :
    NeuronCount{ neuronCount },
    Neurons{ std::make_unique<typename Neuron<T>::Uptr[]>(NeuronCount) }
  {
    if (NeuronCount == 0)
    {
      throw std::invalid_argument("cnn::Layer::Layer(), NeuronCount == 0.");
    }
    if (inputCount == 0)
    {
      throw std::invalid_argument("cnn::Layer::Layer(), inputCount == 0.");
    }
    for (size_t i = 0; i < NeuronCount; ++i)
    {
      Neurons[i] = std::make_unique<Neuron<T>>(inputCount);
    }
  }

  template <typename T>
  size_t Layer<T>::GetNeuronCount() const
  {
    return NeuronCount;
  }

  template <typename T>
  const INeuron<T>& Layer<T>::GetNeuron(const size_t index) const
  {
    if (index >= NeuronCount)
    {
      throw std::range_error("cnn::Layer::GetNeuron() const, index >= NeuronCount.");
    }
    return *(Neurons[index]);
  }

  template <typename T>
  INeuron<T>& Layer<T>::GetNeuron(const size_t index)
  {
    if (index >= NeuronCount)
    {
      throw std::range_error("cnn::Layer::GetNeuron(), index >= NeuronCount.");
    }
    return *(Neurons[index]);
  }
}