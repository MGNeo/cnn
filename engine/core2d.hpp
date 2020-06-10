#pragma once

#include "core.hpp"

namespace cnn
{
  namespace engine
  {
    template <typename T, size_t W, size_t H, typename F = StandardActivator<T>>
    class Core2D : private Core<T, W * H, F>
    {
      static_assert(std::is_arithmetic<T>::value);

      static_assert(W > 0);
      static_assert(H > 0);
      static_assert(((W * H) / W) == H);

    public:
      
      T GetInput(const size_t x, const size_t y) const;
      void SetInput(const size_t x, const size_t y, const T value);

      T GetWeight(const size_t x, const size_t y) const;
      void SetWeight(const size_t x, const size_t y, const T value);

      size_t GetWidth() const;
      size_t GetHeight() const;

    private:

      size_t ToIndex(const size_t x, const size_t y) const;

    };

    template<typename T, size_t W, size_t H, typename F>
    size_t Core2D<T, W, H, F>::ToIndex(const size_t x, const size_t y) const
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Core2D::ToIndex(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Core2D::ToIndex(), y >= H.");
      }
      return (x % H) + W * (x / H);
    }

    template<typename T, size_t W, size_t H, typename F>
    T Core2D<T, W, H, F>::GetInput(const size_t x, const size_t y) const
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Core2D::GetInput(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Core2D::GetInput(), y >= H.");
      }
      return Core<T, W * H, F>::GetInput(ToIndex(x, y));
    }

    template<typename T, size_t W, size_t H, typename F>
    void Core2D<T, W, H, F>::SetInput(const size_t x, const size_t y, const T value)
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Core2D::SetInput(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Core2D::SetInput(), y >= H.");
      }
      return Core<T, W * H, F>::GetInput(ToIndex(x, y));
    }

    template<typename T, size_t W, size_t H, typename F>
    T Core2D<T, W, H, F>::GetWeight(const size_t x, const size_t y) const
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Core2D::GetWeight(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Core2D::GetWeight(), y >= H.");
      }
      return Core<T, W * H, F>::GetWeight(ToIndex(x, y));
    }

    template<typename T, size_t W, size_t H, typename F>
    void Core2D<T, W, H, F>::SetWeight(const size_t x, const size_t y, const T value)
    {
      if (x >= W)
      {
        throw std::range_error("cnn::engine::Core2D::SetWeight(), x >= W.");
      }
      if (y >= H)
      {
        throw std::range_error("cnn::engine::Core2D::SetWeight(), y >= H.");
      }
      Core<T, W * H, F>::SetWeight(ToIndex(x, y), value);
    }

    template<typename T, size_t W, size_t H, typename F>
    size_t Core2D<T, W, H, F>::GetWidth() const
    {
      return W;
    }

    template<typename T, size_t W, size_t H, typename F>
    size_t Core2D<T, W, H, F>::GetHeight() const
    {
      return H;
    }

  }
}
