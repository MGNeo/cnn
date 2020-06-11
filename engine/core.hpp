#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace cnn
{
  template <typename T>
  class Core
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Core(const size_t count);

    size_t GetCount() const;

    T GetInput(const size_t index) const;
    void SetInput(const size_t index, const T value);

    T GetWeight(const size_t index) const;
    void SetWeight(const size_t index, const T value);

    void GenerateOutput();

    T GetOutput() const;

  private:

    const size_t Count;
    std::unique_ptr<T[]> Inputs;
    std::unique_ptr<T[]> Weights;
    T Output;

  };

  template <typename T>
  Core<T>::Core(const size_t count)
    :
    Count{ count },
    Inputs{ std::make_unique<T[]>(Count) },
    Weights{ std::make_unique<T[]>(Count) },
    Output{}
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Core::Core(), Count == 0.");
    }
    memset(Inputs.get(), 0, sizeof(T) * Count);
    memset(Weights.get(), 0, sizeof(T) * Count);
  }

  template <typename T>
  size_t Core<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Core<T>::GetInput(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Core::GetInput(), index >= Count.");
    }
    return Inputs[index];
  }

  template <typename T>
  void Core<T>::SetInput(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Core::SetInput(), index >= Count.");
    }
    Inputs[index] = value;
  }

  template <typename T>
  T Core<T>::GetWeight(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Core::GetWeight(), index >= Count.");
    }
    return Weights[index];
  }

  template <typename T>
  void Core<T>::SetWeight(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Core::SetWeight(), index >= Count.");
    }
    Weights[index] = value;
  }

  template <typename T>
  void Core<T>::GenerateOutput()
  {
    Output = 0;
    for (size_t i = 0; i < Count; ++i)
    {
      Output += Inputs[i] * Weights[i];
    }
  }

  template <typename T>
  T Core<T>::GetOutput() const
  {
    return Output;
  }
}
