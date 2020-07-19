#pragma once

#include <stdexcept>

#include "i_neuron.hpp"

namespace cnn
{
  template <typename T>
  class Neuron : public INeuron<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Neuron(const size_t inputCount);

    size_t GetInputCount() const override;

    T GetInput(const size_t index) const override;
    void SetInput(const size_t index, const T value) override;

    T GetWeight(const size_t index) const override;
    void SetWeight(const size_t index, const T value) override;

    void GenerateOutput() override;

    T GetOutput() const override;

  private:

    const size_t InputCount;
    std::unique_ptr<T[]> Inputs;
    std::unique_ptr<T[]> Weights;
    T Output;

  };

  template <typename T>
  Neuron<T>::Neuron(const size_t inputCount)
    :
    InputCount{ inputCount },
    Inputs{ std::make_unique<T[]>(InputCount) },
    Weights{ std::make_unique<T[]>(InputCount) },
    Output{}
  {
    if (InputCount == 0)
    {
      throw std::invalid_argument("cnn::Neuron::Neorun(), InputCount == 0.");
    }
    memset(Inputs.get(), 0, sizeof(T) * InputCount);
    memset(Weights.get(), 0, sizeof(T) * InputCount);
  }

  template <typename T>
  size_t Neuron<T>::GetInputCount() const
  {
    return InputCount;
  }

  template <typename T>
  T Neuron<T>::GetInput(const size_t index) const
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Neuron::GetInput() const, index >= InputCount.");
    }
    return Inputs[index];
  }

  template <typename T>
  void Neuron<T>::SetInput(const size_t index, const T value)
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Neuron::SetInput(), index >= InputCount.");
    }
    Inputs[index] = value;
  }

  template <typename T>
  T Neuron<T>::GetWeight(const size_t index) const
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Neuron::GetWeight() const, index >= InputCount.");
    }
    return Weights[index];
  }

  template <typename T>
  void Neuron<T>::SetWeight(const size_t index, const T value)
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Neuron::SetWeight(), index >= InputCount.");
    }
    Weights[index] = value;
  }

  template <typename T>
  void Neuron<T>::GenerateOutput()
  {
    Output = 0;
    for (size_t i = 0; i < InputCount; ++i)
    {
      Output += Inputs[i] * Weights[i];
    }
    // TODO: Use activate function.
  }

  template <typename T>
  T Neuron<T>::GetOutput() const
  {
    return Output;
  }
}