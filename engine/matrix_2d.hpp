#pragma once

#include "i_matrix_2d.hpp"
#include "matrix.hpp"

namespace cnn
{
  template <typename T>
  class Matrix2D : public IMatrix2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Matrix2D(const size_t width, const size_t height);

    size_t GetWidth() const override;
    size_t GetHeight() const override;

    T GetCell(const size_t x, const size_t y) const override;
    void SetCell(const size_t x, const size_t y, const T value) override;

  private:

    const size_t Width;
    const size_t Height;
    typename IMatrix<T> Matrix_;

    size_t ToIndex(const size_t x, const size_t y) const;

  };

  template <typename T>
  Matrix2D<T>::Matrix2D(const size_t width, const size_t height)
    :
    Width{ width },
    Height{ height },
    Matrix_{ std::make_unique<Matrix<T>>(Width * Height) }
  {
    const size_t m = Width * Height;
    if ((Width > 0) && (Height > 0) && ((m / Width) != Height))
    {
      throw std::overflow_error("cnn::Matrix2D::Matrix2D(), m was overflowed.");
    }
  }

  template <typename T>
  size_t Matrix2D<T>::GetWidth() const
  {
    return Width;
  }

  template <typename T>
  size_t Matrix2D<T>::GetHeight() const
  {
    return Height;
  }

  template <typename T>
  T Matrix2D<T>::GetCell(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Matrix2D::GetCell(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Matrix2D::GetCell(), y >= Height.");
    }
    return Matrix_.GetCell(ToIndex(x, y));
  }

  template <typename T>
  void Matrix2D<T>::SetCell(const size_t x, const size_t y, const T value)
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Matrix2D::SetCell(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Matrix2D::SetCell(), y >= Height.");
    }
    Matrix_.SetCell(ToIndex(x, y), value);
  }

  template <typename T>
  size_t Matrix2D<T>::ToIndex(const size_t x, const size_t y) const
  {
    if (x >= Width)
    {
      throw std::range_error("cnn::Matrix2D::ToIndex(), x >= Width.");
    }
    if (y >= Height)
    {
      throw std::range_error("cnn::Matrix2D::ToIndex(), y >= Height.");
    }
    return x + y * Width;
  }
}