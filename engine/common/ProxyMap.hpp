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

        ProxyMap(Map<T>& map);

        ProxyMap(const ProxyMap<T>& proxyMap);

        ProxyMap(ProxyMap&& proxyMap) = delete;

        ProxyMap& operator=(const ProxyMap<T>& proxyMap) = delete;

        ProxyMap& operator=(ProxyMap<T>&& proxyMap) = delete;

        // Exception guarantee: strong for this.
        size_t GetValueCount() const noexcept;

        // Exception guarantee: strong for this.
        T GetValue(const size_t index) const;

        void SetValue(const size_t index, const T value);

        // It clears the state without changing of the topology.
        void Clear() noexcept;

      private:

        Map<T>& Map_;

      };

      template <typename T>
      ProxyMap<T>::ProxyMap(Map<T>& map)
        :
        Map_{ map }
      {
      }

      template <typename T>
      ProxyMap<T>::ProxyMap(const ProxyMap<T>& proxyMap)
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
      void ProxyMap<T>::SetValue(const size_t index, const T value)
      {
        Map_.SetValue(index, value);
      }

      template <typename T>
      void ProxyMap<T>::Clear() noexcept
      {
        Map_.Clear();
      }
    }
  }
}