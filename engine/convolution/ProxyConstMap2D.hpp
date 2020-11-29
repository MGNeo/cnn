#pragma once

#include "Map2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // ProxyConstMap2D is a protecting proxy, which protects a Map2D from changing.
      template <typename T>
      class ProxyConstMap2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyConstMap2D(const Map2D<T>& map) noexcept;

        ProxyConstMap2D(const ProxyConstMap2D& proxyConstMap) noexcept;

        ProxyConstMap2D(ProxyConstMap2D&& proxyConstMap) noexcept = delete;

        ProxyConstMap2D& operator=(const ProxyConstMap2D& proxyConstMap) = delete;

        ProxyConstMap2D& operator=(ProxyConstMap2D&& proxyConstMap) noexcept = delete;

        Size2D GetSize() const noexcept;

        T GetValue(const size_t x, const size_t y) const;

      private:

        const Map2D<T>& Map_;

      };

      template <typename T>
      ProxyConstMap2D<T>::ProxyConstMap2D(const Map2D<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      ProxyConstMap2D<T>::ProxyConstMap2D(const ProxyConstMap2D& proxyConstMap) noexcept
        :
        Map_{ proxyConstMap.Map_ }
      {
      }

      template <typename T>
      Size2D ProxyConstMap2D<T>::GetSize() const noexcept
      {
        return Map_.GetSize();
      }

      template <typename T>
      T ProxyConstMap2D<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map_.GetValue(x, y);
      }
    }
  }
}
