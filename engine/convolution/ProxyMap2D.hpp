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

        ProxyMap2D(Map2D<T>& map);

        // We delete copy-constructor too for saving of const-transitivity.
        // Now we can't build non-const object using const object.
        // What will you say about it, Elon Musk? ;-)
        ProxyMap2D(const ProxyMap2D& proxyMap) = delete;

        ProxyMap2D(ProxyMap2D&& proxyMap) = delete;

        ProxyMap2D& operator=(const ProxyMap2D& proxyMap) = delete;

        ProxyMap2D& operator=(ProxyMap2D&& proxyMap) = delete;

        Size2D<size_t> GetSize() const noexcept;

        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: strong for this.
        void SetValue(const size_t x, const size_t y, const T value);

        // It clears the state without changing of the topology.
        void Clear() noexcept;

      private:

        Map2D<T>& Map;

      };

      template <typename T>
      ProxyMap2D<T>::ProxyMap2D(Map2D<T>& map)
        :
        Map{ map }
      {
      }

      template <typename T>
      Size2D<size_t> ProxyMap2D<T>::GetSize() const noexcept
      {
        return Map.GetSize();
      }

      template <typename T>
      T ProxyMap2D<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map.GetValue(x, y);
      }

      template <typename T>
      void ProxyMap2D<T>::SetValue(const size_t x, const size_t y, const T value)
      {
        Map.SetValue(x, y, value);
      }

      template <typename T>
      void ProxyMap2D<T>::Clear() noexcept
      {
        Map.Clear();
      }
    }
  }
}