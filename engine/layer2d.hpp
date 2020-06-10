#pragma once

#include "layer.hpp"

namespace cnn
{
  namespace engine
  {
    template<typename T, size_t W, size_t H>
    class Layer2D : private Layer<T, W * H>
    {

      static_assert(std::is_floating_point<T>::value);

      static_assert(W > 0);
      static_assert(H > 0);
      static_assert(((W * H) / W) == H);
      
    public:

      T GetValue(const size_t x, const size_t Y) const;
      void SetValue(const size_t x, const size_t y, const T value);

      size_t GetWidth() const;
      size_t GetHeight() const;

    private:

      size_t ToIndex(const size_t x, const size_t y) const;

    };

    template<typename T, size_t W, size_t H>
    size_t Layer2D<T, W, H>::ToIndex(const size_t x, const size_t y) const
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Layer2D::ToIndex(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Layer2D::ToIndex(), y >= H.");
      }
      return x + y * W;
    }

    template<typename T, size_t W, size_t H>
    T Layer2D<T, W, H>::GetValue(const size_t x, const size_t y) const
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Layer2D::GetValue(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Layer2D::GetValue(), y >= H.");
      }
      return Layer<T, W * H>::GetValue(ToIndex(x, y));
    }

    template<typename T, size_t W, size_t H>
    void Layer2D<T, W, H>::SetValue(const size_t x, const size_t y, const T value)
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Layer2D::SetValue(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Layer2D::SetValue(), y >= H.");
      }
      return Layer<T, W * H>::SetValue(ToIndex(x, y), value);
    }

    template<typename T, size_t W, size_t H>
    size_t Layer2D<T, W, H>::GetWidth() const
    {
      return W;
    }

    template<typename T, size_t W, size_t H>
    size_t Layer2D<T, W, H>::GetHeight() const
    {
      return H;
    }
  }
}
