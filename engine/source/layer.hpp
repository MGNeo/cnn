#pragma once

#include <stdexcept>

#include "i_layer.hpp"

namespace cnn
{
  template <typename T>
  class Layer : ILayer<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Layer(const size_t count);

    size_t GetCount() const override;

    T GetInput(const size_t index) const override;
    void SetInput(const size_t index, const T value) override;

    T GetWeight(const size_t index) const override;
    void SetWeight(const size_t index, const T value) override;

  private:

    const size_t Count;
    const std::unique_ptr<T[]> Inputs;
    const std::unique_ptr<T[]> Weights;

  };

  template <typename T>
  Layer<T>::Layer(const size_t count)
    :
    Count{ count },
    Inputs{ std::make_unique<T[]>(Count) },
    Weights{ std::make_unique<T[]>(Count) }
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Layer::Layer(), Count == 0.");
    }
  }

  template <typename T>
  size_t Layer<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Layer<T>::GetInput(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Layer::GetInput(), index >= Count.");
    }
    return Inputs[index];
  }

  template <typename T>
  void Layer<T>::SetInput(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Layer::SetInput(), index >= Count.");
    }
    Inputs[index] = value;
  }

  template <typename T>
  T Layer<T>::GetWeight(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Layer::GetWeight(), index >= Count.");
    }
    return Weights[index];
  }

  template <typename T>
  void Layer<T>::SetWeight(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::invalid_argument("cnn::Layer::SetWeight(), index >= Count.");
    }
    Weights[index] = value;
  }
}