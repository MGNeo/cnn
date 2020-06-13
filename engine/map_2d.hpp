#pragma once

#include "i_Map_2d.hpp"
#include "Map.hpp"

namespace cnn
{
  template <typename T>
  class Map2D : public IMap2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Map2D(const size_t width, const size_t height);

    size_t GetWidth() const override;
    size_t GetHeight() const override;

    T GetValue(const size_t x, const size_t y) const override;
    void SetValue(const size_t x, const size_t y, const T value) override;

    void Clear() override;

  private:

    const size_t Width;
    const size_t Height;
    typename IMap<T>::Uptr Map_;

    size_t ToIndex(const size_t x, const size_t y) const;

  };

  template <typename T>
  Map2D<T>::Map2D(const size_t width, const size_t height)
    :
    Width{ width },
    Height{ height },
    Map_{ std::make_unique<Map<T>>(Width * Height) }
  {
    const size_t m = Width * Height;
    if ((Width > 0) && (Height > 0) && ((m / Width) != Height))
    {
      throw std::overflow_error("cnn::Map2D::Map2D(), m was overflowed.");
    }
  }

  template <typename T>
  size_t Map2D<T>::GetWidth() const
  {
    return Width;
  }

  template <typename T>
  size_t Map2D<T>::GetHeight() const
  {
    return Height;
  }

  template <typename T>
  T Map2D<T>::GetValue(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Map2D::GetValue(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Map2D::GetValue(), y >= Height.");
    }
    return Map_->GetValue(ToIndex(x, y));
  }

  template <typename T>
  void Map2D<T>::SetValue(const size_t x, const size_t y, const T value)
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Map2D::SetValue(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Map2D::SetValue(), y >= Height.");
    }
    Map_->SetValue(ToIndex(x, y), value);
  }

  template <typename T>
  size_t Map2D<T>::ToIndex(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Map2D::ToIndex(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Map2D::ToIndex(), y >= Height.");
    }
    return x + y * Width;
  }

  template <typename T>
  void Map2D<T>::Clear()
  {
    Map_->Clear();
  }
}