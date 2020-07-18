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

    Neuron(const size_t count);

    size_t GetCount() const override;

    T GetInput(const size_t index) const override;
    void SetInput(const size_t index, const T value) override;

    T GetWeight(const size_t index) const override;
    void SetWeight(const size_t index, const T value) override;

    T GetOutput() const override;

    void Process() override;

  private:

    const size_t Count;
    std::unique_ptr<T[]> Inputs;
    std::unique_ptr<T[]> Weights;
    T Output;

  };

  template <typename T>
  Neuron<T>::Neuron(const size_t count)
    :
    Count{ count },
    Inputs{ std::make_unique<T[]>(Count) },
    Outputs{ std::make_unique<T[]>(Count) },
    Output{}
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Neuron::Neorun(), Count == 0.");
    }
  }

  template <typename T>
  size_t Neuron<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Neuron<T>::GetInput(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Neuron::GetInput(), index >= Count.");
    }
    return Inputs[index];
  }

  template <typename T>
  void Neuron<T>::SetInput(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Neuron::SetInput(), index >= Count.");
    }
    Inputs[index] = value;
  }

  template <typename T>
  T Neuron<T>::GetWeight(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Neuron::GetWeight(), index >= Count.");
    }
    return Weights[index];
  }

  template <typename T>
  void Neuron<T>::SetWeight(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::invalid_argument]("cnn::Neuron::SetWeight(), index >= Count.");
    }
    Weights[index] = value;
  }

  template <typename T>
  T Neuron<T>::GetOutput() const
  {
    return Output;
  }

  template <typename T>
  void Neuron<T>::Process()
  {
    Output = 0;
    for (size_t i = 0; i < Count; ++i)
    {
      Output += Inputs[i] * Weights[i];
    }
    // TODO: Use activate function.
  }
}