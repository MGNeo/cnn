#pragma once

#include "Map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // ProxyMap is a protecting proxy, which protects a Map from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Map as its part.
      template <typename T>
      class ProxyMap
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyMap(Map<T>& map) noexcept;

        ProxyMap(const ProxyMap<T>& proxyMap) noexcept;

        ProxyMap(ProxyMap&& proxyMap) = delete;

        ProxyMap& operator=(const ProxyMap<T>& proxyMap) = delete;

        ProxyMap& operator=(ProxyMap<T>&& proxyMap) = delete;

        // Exception guarantee: strong for the map.
        size_t GetValueCount() const noexcept;

        // Exception guarantee: strong for the map.
        T GetValue(const size_t index) const;

        // Exception guarantee: strong for the map.
        void SetValue(const size_t index, const T value) const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // Exception guarantee: strong for this.
        // Topologies of this and map must be equal.
        void FillFrom(const ProxyMap& proxyMap) const;

      private:

        Map<T>& Map_;

      };

      template <typename T>
      ProxyMap<T>::ProxyMap(Map<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      ProxyMap<T>::ProxyMap(const ProxyMap<T>& proxyMap) noexcept
        :
        Map_{ proxyMap.Map_ }
      {
      }

      template <typename T>
      size_t ProxyMap<T>::GetValueCount() const noexcept
      {
        return Map_.GetValueCount();
      }

      template <typename T>
      T ProxyMap<T>::GetValue(const size_t index) const
      {
        return Map_.GetValue(index);
      }

      template <typename T>
      void ProxyMap<T>::SetValue(const size_t index, const T value) const
      {
        Map_.SetValue(index, value);
      }

      template <typename T>
      void ProxyMap<T>::Clear() const noexcept
      {
        Map_.Clear();
      }

      template <typename T>
      void ProxyMap<T>::FillFrom(const ProxyMap<T>& proxyMap) const
      {
        Map_.FillFrom(proxyMap.Map_);
      }
    }
  }
}