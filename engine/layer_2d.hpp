#pragma once

#include "i_layer_2d.hpp"
#include "layer.hpp"

namespace cnn
{
  template <typename T>
  class Layer2D : public ILayer2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Layer2D(const size_t width, const size_t height);

    size_t GetWidth() const override;
    size_t GetHeight() const override;

    T GetCell(const size_t x, const size_t y) const override;
    void SetCell(const size_t x, const size_t y, const T value) override;

  private:

    const size_t Width;
    const size_t Height;
    Layer<T> Layer_;

    size_t ToIndex(const size_t x, const size_t y) const;

  };

  template <typename T>
  Layer2D<T>::Layer2D(const size_t width, const size_t height)
    :
    Width{ width },
    Height{ height },
    Layer_{ Width * Height }
  {
    const size_t m = Width * Height;
    if ((Width > 0) && (Height > 0) && ((m / Width) != Height))
    {
      throw std::overflow_error("cnn::Layer2D::Layer2D(), m was overflowed.");
    }
  }

  template <typename T>
  size_t Layer2D<T>::GetWidth() const
  {
    return Width;
  }

  template <typename T>
  size_t Layer2D<T>::GetHeight() const
  {
    return Height;
  }

  template <typename T>
  T Layer2D<T>::GetCell(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Layer2D::GetCell(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Layer2D::GetCell(), y >= Height.");
    }
    return Layer_.GetCell(ToIndex(x, y));
  }

  template <typename T>
  void Layer2D<T>::SetCell(const size_t x, const size_t y, const T value)
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Layer2D::SetCell(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Layer2D::SetCell(), y >= Height.");
    }
    Layer_.SetCell(ToIndex(x, y), value);
  }

  template <typename T>
  size_t Layer2D<T>::ToIndex(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Layer2D::ToIndex(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Layer2D::ToIndex(), y >= Height.");
    }
    return x + y * Width;
  }
}