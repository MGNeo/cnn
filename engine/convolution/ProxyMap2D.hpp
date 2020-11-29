#pragma once

#include "Map2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // ProxyMap2D is a protecting proxy, which protects a Map2D from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Map2D as its part.
      template <typename T>
      class ProxyMap2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyMap2D(Map2D<T>& map) noexcept;

        ProxyMap2D(const ProxyMap2D& proxyMap) noexcept;

        ProxyMap2D(ProxyMap2D&& proxyMap) noexcept = delete;

        ProxyMap2D& operator=(const ProxyMap2D& proxyMap) = delete;

        ProxyMap2D& operator=(ProxyMap2D&& proxyMap) noexcept = delete;

        Size2D GetSize() const noexcept;

        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: strong for this.
        void SetValue(const size_t x, const size_t y, const T value) const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this.
        // The topologies of this and map must be equal.
        void FillFrom(const ProxyMap2D& proxyMap) const;

      private:

        Map2D<T>& Map_;

      };

      template <typename T>
      ProxyMap2D<T>::ProxyMap2D(Map2D<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      ProxyMap2D<T>::ProxyMap2D(const ProxyMap2D& proxyMap) noexcept
        :
        Map_{ proxyMap.Map_ }
      {
      }

      template <typename T>
      Size2D ProxyMap2D<T>::GetSize() const noexcept
      {
        return Map_.GetSize();
      }

      template <typename T>
      T ProxyMap2D<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map_.GetValue(x, y);
      }

      template <typename T>
      void ProxyMap2D<T>::SetValue(const size_t x, const size_t y, const T value) const
      {
        Map_.SetValue(x, y, value);
      }

      template <typename T>
      void ProxyMap2D<T>::Clear() const noexcept
      {
        Map_.Clear();
      }

      template <typename T>
      void ProxyMap2D<T>::Save(std::ostream& ostream) const
      {
        Map_.Save(ostream);
      }

      template <typename T>
      void ProxyMap2D<T>::FillFrom(const ProxyMap2D& proxyMap) const
      {
        Map_.FillFrom(proxyMap.Map_);
      }
    }
  }
}
