#pragma once

#include <vector>

#include "i_perceptron.hpp"
#include "layer.hpp"

namespace cnn
{
  template <typename T>
  class Perceptron : public IPerceptron<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Perceptron(const size_t inputCount);

    size_t GetInputCount() const override;

    size_t GetLayerCount() const override;

    const ILayer<T>& GetLayer(const size_t index) const override;
    ILayer<T>& GetLayer(const size_t index) override;

    void PushLayer(const size_t count) override;

    void Process() override;

  private:

    const size_t InputCount;
    std::vector<typename ILayer<T>::Uptr> Layers;

  };

  template <typename T>
  Perceptron<T>::Perceptron(const size_t inputCount)
    :
    InputCount{ inputCount }
  {
    if (InputCount == 0)
    {
      throw std::invalid_argument("cnn::Perceptron::Perceptron(), InputCount == 0.");
    }
  }

  template <typename T>
  size_t Perceptron<T>::GetInputCount() const
  {
    return InputCount;
  }

  template <typename T>
  size_t Perceptron<T>::GetLayerCount() const
  {
    return Layers.size();
  }

  template <typename T>
  const ILayer<T>& Perceptron<T>::GetLayer(const size_t index) const
  {
    if (index >= GetLayerCout())
    {
      throw std::range_error("cnn::Perceptron::GetLayer(), index >= GetLayerCount().");
    }
    return *(Layers[index]);
  }

  template <typename T>
  ILayer<T>& Perceptron<T>::GetLayer(const size_t index)
  {
    if (index >= GetLayerCount())
    {
      throw std::range_error("cnn::Perceptron::GetLayer(), index >= GetLayerCount().");
    }
    return *(Layers[index]);
  }

  template <typename T>
  void Perceptron<T>::PushLayer(const size_t neuronCount)
  {
    if (neuronCount == 0)
    {
      throw std::invalid_argument("cnn::Perceptron::PushLayer(), neuronCount == 0.");
    }
    ILayer<T>::Uptr layer{};
    if (GetLayerCount() == 0)
    {
      layer = std::make_unique<Layer<T>>(neuronCount, InputCount)
    } else {
      auto inputCount = GetLayer(GetLayerCount() - 1);
      layer = std::make_unique<Layer<T>>(neuronCount, inputCount);
    }
    Layers.push_back(std::move(layer));
  }

  template <typename T>
  void Perceptron<T>::Process()
  {
    /*
    // TODO: We need to reset all layers (inputs and outputs of neurons) if exception was generated.
    for (size_t l = 0; l < GetLayerCount(); ++l)
    {
      auto& layer = Layers[l];
      if (l == 0)
      {
        for (size_t n = 0; n < layer->GetNeuronCount(); ++n)
        {
          auto& neuron = layer->GetNeuron(n);
          for (size_t i = 0; i < neuron.GetInputCount(); ++i)
          {
            neuron.SetInput(Input)
          }
        }
      } else {

      }
    }
    */
  }
}