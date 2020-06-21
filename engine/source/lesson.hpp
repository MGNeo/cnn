#pragma once

#include <stdexcept>

#include "i_lesson.hpp"

namespace cnn
{
  template <typename T>
  class Lesson : public ILesson<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Lesson(const size_t inputCount, const size_t outputCount);

    size_t GetInputCount() const override;
    size_t GetOutputCount() const override;

    T GetInput(const size_t index) const override;
    void SetInput(const size_t index, const T value) override;

    T GetOutput(const size_t index) const override;
    void SetOutput(const size_t index, const T value) override;

  private:

    const size_t InputCount;
    const size_t OutputCount;

    const std::unique_ptr<T[]> Inputs;
    const std::unique_ptr<T[]> Outputs;

  };

  template <typename T>
  Lesson<T>::Lesson(const size_t inputCount, const size_t outputCount)
    :
    InputCount{ inputCount },
    OutputCount{ outputCount },
    Inputs{ std::make_unique<T[]>(InputCount) },
    Outputs{ std::make_unique<T[]>(OutputCount) }
  {
    if (InputCount == 0)
    {
      throw std::invalid_argument("cnn::Lesson::Lesson(), InputCount == 0.");
    }
    if (OutputCount == 0)
    {
      throw std::invalid_argument("cnn::Lesson::Lesson(), OutputCount == 0.");
    }
    for (size_t i = 0; i < InputCount; ++i)
    {
      Inputs[i] = 0;
    }
    for (size_t i = 0; i < OutputCount; ++i)
    {
      Outputs[i] = 0;
    }
  }

  template <typename T>
  size_t Lesson<T>::GetInputCount() const
  {
    return InputCount;
  }

  template <typename T>
  size_t Lesson<T>::GetOutputCount() const
  {
    return OutputCount;
  }

  template <typename T>
  T Lesson<T>::GetInput(const size_t index) const
  {
    if (index >= InputCount)
    {
      throw std::invalid_argument("cnn::Lesson::GetInput(), index >= InputCount.");
    }
    return Inputs[index];
  }

  template <typename T>
  void Lesson<T>::SetInput(const size_t index, const T value)
  {
    if (index >= InputCount)
    {
      throw std::invalid_argument("cnn::Lesson::SetInput(), index >= InputCount.");
    }
    Inputs[index] = value;
  }

  template <typename T>
  T Lesson<T>::GetOutput(const size_t index) const
  {
    if (index >= OutputCount)
    {
      throw std::invalid_argument("cnn::Lesson::GetOutput(), index >= OutputCount.");
    }
    return Outputs[index];
  }

  template <typename T>
  void Lesson<T>::SetOutput(const size_t index, const T value)
  {
    if (index >= OutputCount)
    {
      throw std::invalid_argument("cnn::Lesson::SetOutput(), index >= OutputCount.");
    }
    Outputs[index] = value;
  }
}