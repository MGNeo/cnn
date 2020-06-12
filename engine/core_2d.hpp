#pragma once

#include "i_core_2d.hpp"
#include "i_core.hpp"
#include "core.hpp"

namespace cnn
{
  template <typename T>
  class Core2D : public ICore2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Core2D(const size_t width, const size_t height);

    size_t GetWidth() const override;
    size_t GetHeight() const override;

    T GetInput(const size_t x, const size_t y) const override;
    void SetInput(const size_t x, const size_t y, const T value) override;

    T GetWeight(const size_t x, const size_t y) const override;
    void SetWeight(const size_t x, const size_t y, const T value) override;

    void GenerateOutput() override;

    T GetOutput() const override;

  private:

    const size_t Width;
    const size_t Height;    
    typename ICore<T>::Uptr Core_;

    size_t ToIndex(const size_t x, const size_t y) const;

  };

  template <typename T>
  Core2D<T>::Core2D(const size_t width, const size_t height)
    :
    Width{ width },
    Height{ height },
    Core_{ std::make_unique<Core<T>>(Width * Height) }
  {
    const size_t m = Width * Height;
    if ((Width != 0) && (Height != 0) && ((m / Width) != Height))
    {
      throw std::overflow_error("cnn::Core2D::Core2D(), m was overflowed.");
    }
  }

  template <typename T>
  size_t Core2D<T>::GetWidth() const
  {
    return Width;
  }

  template <typename T>
  size_t Core2D<T>::GetHeight() const
  {
    return Height;
  }

  template <typename T>
  T Core2D<T>::GetInput(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Core2D::GetInput(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Core2D::GetInput(), y >= Height.");
    }
    return Core_->GetInput(ToIndex(x, y));
  };

  template <typename T>
  void Core2D<T>::SetInput(const size_t x, const size_t y, const T value)
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Core2D::SetInput(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Core2D::SetInput(), y >= Height.");
    }
    Core_->SetInput(ToIndex(x, y), value);
  };

  template <typename T>
  T Core2D<T>::GetWeight(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Core2D::GetWeight(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Core2D::GetWeight(), y >= Height.");
    }
    return Core_->GetWeight(ToIndex(x, y));
  }

  template <typename T>
  void Core2D<T>::SetWeight(const size_t x, const size_t y, const T value)
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Core2D::SetWeight(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Core2D::setWeight(), y >= Height.");
    }
    Core_->SetWeight(ToIndex(x, y), value);
  }

  template <typename T>
  void Core2D<T>::GenerateOutput()
  {
    Core_->GenerateOutput();
  }

  template <typename T>
  T Core2D<T>::GetOutput() const
  {
    return Core_->GetOutput();
  }

  template <typename T>
  size_t Core2D<T>::ToIndex(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Core2D::ToIndex(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Core2D::ToIndex(), y >= Height.");
    }
    return x + Width * y;
  }

}