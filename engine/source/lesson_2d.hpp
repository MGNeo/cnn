#pragma once

#include "i_lesson_2d.hpp"
#include "lesson.hpp"

namespace cnn
{
  template <typename T>
  class Lesson2D : public ILesson2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Lesson2D(const size_t inputWidth,
             const size_t inputHeight,
             
             const size_t outputWidth,
             const size_t outputHeight);

    size_t GetInputWidth() const override;
    size_t GetIntputHeight() const override;

    size_t GetOutputWidth() const override;
    size_t GetOutputHeight() const override;

    T GetInput(const size_t x, const size_t y) const override;
    void SetInput(const size_t x, const size_t y, const T value) override;

    T GetOutput(const size_t x, const size_t y) const override;
    void SetOutput(const size_t x, const size_t y, const T value) override;

  private:

    const size_t InputWidth;
    const size_t InputHeight;
    const size_t OutputWidth;
    const size_t OutputHeight;
    const typename ILesson<T>::Uptr Lesson_;

    size_t ToInputIndex(const size_t x, const size_t y) const;
    size_t ToOutputIndex(const size_t x, const size_t y) const;

  };

  template <typename T>
  Lesson2D<T>::Lesson2D(const size_t inputWidth,
                        const size_t inputHeight,
    
                        const size_t outputWidth,
                        const size_t outputHeight)
    :
    InputWidth{ inputWidth },
    InputHeight{ inputHeight },
    OutputWidth{ outputWidth },
    OutputHeight{ outputHeight },
    Lesson_{ std::make_unique<Lesson<T>>(InputWidth * InputHeight, OutputWidth * OutputHeight) }
  {
    if (InputWidth == 0)
    {
      throw std::invalid_argument("cnn::Lesson2D::Lesson2D(), InputWidth == 0.");
    }
    if (InputHeight == 0)
    {
      throw std::invalid_argument("cnn::Lesson2D::Lesson2D(), InputHeight == 0.");
    }
    if (OutputWidth == 0)
    {
      throw std::invalid_argument("cnn::Lesson2D::Lesson2D(), OutputWidth == 0.");
    }
    if (OutputHeight == 0)
    {
      throw std::invalid_argument("cnn::Lesson2D::Lesson2D(), OutputHeight == 0.");
    }
    const size_t m1 = InputWidth * InputHeight;
    if ((m1 / InputWidth) != InputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::Lesson2D(), m1 was overflowed.");
    }
    const size_t m2 = OutputWidth * OutputHeight;
    if ((m2 / OutputWidth) != OutputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::Lesson2D(), m2 was overflowed.");
    }
  }

  template <typename T>
  size_t Lesson2D<T>::GetInputWidth() const
  {
    return InputWidth;
  }

  template <typename T>
  size_t Lesson2D<T>::GetIntputHeight() const
  {
    return InputHeight;
  }

  template <typename T>
  size_t Lesson2D<T>::GetOutputWidth() const
  {
    return OutputWidth;
  }

  template <typename T>
  size_t Lesson2D<T>::GetOutputHeight() const
  {
    return OutputHeight;
  }

  template <typename T>
  T Lesson2D<T>::GetInput(const size_t x, const size_t y) const
  {
    if (x >= InputWidth)
    {
      throw std::invalid_argument("cnn::Lesson2D::GetInput(), x >= InputWidth.");
    }
    if (y >= InputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::GetInput(), y >= InputHeight.");
    }
    return Lesson_->GetInput(ToInputIndex(x, y));
  }

  template <typename T>
  void Lesson2D<T>::SetInput(const size_t x, const size_t y, const T value)
  {
    if (x >= InputWidth)
    {
      throw std::invalid_argument("cnn::Lesson2D::SetInput(), x >= InputWidth.");
    }
    if (y >= InputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::SetInput(), y >= InputHeight.");
    }
    Lesson_->SetInput(ToInputIndex(x, y), value);
  }

  template <typename T>
  T Lesson2D<T>::GetOutput(const size_t x, const size_t y) const
  {
    if (x >= OutputWidth)
    {
      throw std::invalid_argument("cnn::Lesson2D::GetOutput(), x >= OutputWidth.");
    }
    if (y >= OutputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::GetOutput(), y >= OutputHeight.");
    }
    return Lesson_->GetOutput(ToOutputIndex(x, y));
  }

  template <typename T>
  void Lesson2D<T>::SetOutput(const size_t x, const size_t y, const T value)
  {
    if (x >= OutputWidth)
    {
      throw std::invalid_argument("cnn::Lesson2D::SetOutput(), x >= OutputWidth.");
    }
    if (y >= OutputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::SetOutput(), y >= OutputHeight.");
    }
    Lesson_->SetOutput(ToOutputIndex(x, y), value);
  }

  template <typename T>
  size_t Lesson2D<T>::ToInputIndex(const size_t x, const size_t y) const
  {
    if (x >= InputWidth)
    {
      throw std::invalid_argument("cnn::Lesson2D::ToInputIndex(), x >= InputWidth.");
    }
    if (y >= InputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::ToInputIndex(), y >= InputHeight.");
    }
    return x + y * InputWidth;
  }

  template <typename T>
  size_t Lesson2D<T>::ToOutputIndex(const size_t x, const size_t y) const
  {
    if (x >= OutputWidth)
    {
      throw std::invalid_argument("cnn::Lesson2D::ToOutputIndex(), x >= OutputWidth.");
    }
    if (y >= OutputHeight)
    {
      throw std::invalid_argument("cnn::Lesson2D::ToOutputIndex(), y >= OutputHeight.");
    }
    return x + y * OutputWidth;
  }

}