#pragma once

#include "Map2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ProxyMap2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyMap2D(Map2D<T>& map) noexcept;

        ProxyMap2D(const ProxyMap2D& proxyMap) noexcept;

        ProxyMap2D(ProxyMap2D&& proxyMap) = delete;

        ProxyMap2D& operator=(const ProxyMap2D& proxyMap) = delete;

        ProxyMap2D& operator=(ProxyMap2D&& proxyMap) = delete;

        Size2D GetSize() const noexcept;

        // Exception guarantee: strong for the map.
        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the map.
        void SetValue(const size_t x, const size_t y, const T value) const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // Exception guarantee: strong for the map.
        // Topologies of this and map must be equal.
        void FillFrom(const ProxyMap2D& proxyMap) const;

      private:

        Map2D<T>& Map;

      };

      template <typename T>
      ProxyMap2D<T>::ProxyMap2D(Map2D<T>& map) noexcept
        :
        Map{ map }
      {
      }

      template <typename T>
      ProxyMap2D<T>::ProxyMap2D(const ProxyMap2D& proxyMap) noexcept
        :
        Map{ proxyMap.Map }
      {
      }

      template <typename T>
      Size2D ProxyMap2D<T>::GetSize() const noexcept
      {
        return Map.GetSize();
      }

      template <typename T>
      T ProxyMap2D<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map.GetValue(x, y);
      }

      template <typename T>
      void ProxyMap2D<T>::SetValue(const size_t x, const size_t y, const T value) const
      {
        Map.SetValue(x, y, value);
      }

      template <typename T>
      void ProxyMap2D<T>::Clear() const noexcept
      {
        Map.Clear();
      }

      template <typename T>
      void ProxyMap2D<T>::FillFrom(const ProxyMap2D<T>& proxyMap) const
      {
        Map.FillFrom(proxyMap.Map);
      }
    }
  }
}