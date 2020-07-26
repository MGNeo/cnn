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

    T GetInput(const size_t index) const override;
    void SetInput(const size_t index, const T value) override;

    size_t GetLayerCount() const override;

    const ILayer<T>& GetLayer(const size_t index) const override;
    ILayer<T>& GetLayer(const size_t index) override;

    const ILayer<T>& GetLastLayer() const override;
    ILayer<T>& GetLastLayer() override;

    void PushLayer(const size_t count) override;

    void Process() override;

  private:

    const size_t InputCount;
    const std::unique_ptr<T[]> Inputs;
    std::vector<typename ILayer<T>::Uptr> Layers;

  };

  template <typename T>
  Perceptron<T>::Perceptron(const size_t inputCount)
    :
    InputCount{ inputCount },
    Inputs{ std::make_unique<T[]>(InputCount) }
  {
    if (InputCount == 0)
    {
      throw std::invalid_argument("cnn::Perceptron::Perceptron(), InputCount == 0.");
    }
    memset(Inputs.get(), 0, sizeof(T) * InputCount);
  }

  template <typename T>
  size_t Perceptron<T>::GetInputCount() const
  {
    return InputCount;
  }

  template <typename T>
  T Perceptron<T>::GetInput(const size_t index) const
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Perceptron::GetInput(), index >= InputCount.");
    }
    return Inputs[index];
  }

  template <typename T>
  void Perceptron<T>::SetInput(const size_t index, const T value)
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Perceptron::SetInput(), index >= InputCount.");
    }
    Inputs[index] = value;
  }

  template <typename T>
  size_t Perceptron<T>::GetLayerCount() const
  {
    return Layers.size();
  }

  template <typename T>
  const ILayer<T>& Perceptron<T>::GetLayer(const size_t index) const
  {
    if (index >= GetLayerCount())
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
  const ILayer<T>& Perceptron<T>::GetLastLayer() const
  {
    if (Layers.size() == 0)
    {
      throw std::logic_error("cnn::Perceptron::GetLastLayer() const, Layers.size() == 0.");
    }
    return *(Layers.back());
  }

  template <typename T>
  ILayer<T>& Perceptron<T>::GetLastLayer()
  {
    if (Layers.size() == 0)
    {
      throw std::logic_error("cnn::Perceptron::GetLastLayer(), Layers.size() == 0.");
    }
    return *(Layers.back());
  }

  template <typename T>
  void Perceptron<T>::PushLayer(const size_t neuronCount)
  {
    if (neuronCount == 0)
    {
      throw std::invalid_argument("cnn::Perceptron::PushLayer(), neuronCount == 0.");
    }
    typename ILayer<T>::Uptr layer{};
    if (GetLayerCount() == 0)
    {
      layer = std::make_unique<Layer<T>>(neuronCount, InputCount);
    } else {
      const size_t inputCount = GetLayer(GetLayerCount() - 1).GetNeuronCount();
      layer = std::make_unique<Layer<T>>(neuronCount, inputCount);
    }
    Layers.push_back(std::move(layer));
  }

  template <typename T>
  void Perceptron<T>::Process()
  {
    if (Layers.size() == 0)
    {
      return;
    }
    // TODO: We need to reset all layers (inputs and outputs of neurons) if exception was generated.

    // Process first layer.
    {
      auto& layer = Layers[0];
      for (size_t n = 0; n < layer->GetNeuronCount(); ++n)
      {
        auto& neuron = layer->GetNeuron(n);
        for (size_t i = 0; i < neuron.GetInputCount(); ++i)
        {
          neuron.SetInput(i, Inputs[i]);
        }
        neuron.GenerateOutput();
      }
    }

    // Process other layers.
    for (size_t l = 1; l < GetLayerCount(); ++l)
    {
      auto& previousLayer = Layers[l - 1];
      auto& currentLayer = Layers[l];
      for (size_t n = 0; n < currentLayer->GetNeuronCount(); ++n)
      {
        auto& currentNeuron = currentLayer->GetNeuron(n);
        for (size_t i = 0; i < currentNeuron.GetInputCount(); ++i)
        {
          auto& previousNeuron = previousLayer->GetNeuron(i);
          currentNeuron.SetInput(i, previousNeuron.GetOutput());
        }
        currentNeuron.GenerateOutput();
      }
    }
  }
}