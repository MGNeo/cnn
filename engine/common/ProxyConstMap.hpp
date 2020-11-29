#pragma once

#include "Map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // ProxyConstMap is a protecting proxy, which protects a Map from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Map as its part.
      template <typename T>
      class ProxyConstMap
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyConstMap(const Map<T>& map) noexcept;

        ProxyConstMap(const ProxyConstMap& proxyConstMap) noexcept;

        ProxyConstMap(ProxyConstMap&& proxyConstMap) noexcept = delete;

        ProxyConstMap& operator=(const ProxyConstMap& proxyConstMap) noexcept = delete;

        ProxyConstMap& operator=(ProxyConstMap&& proxyConstMap) noexcept = delete;

        size_t GetValueCount() const noexcept;

        // Exception guarantee: strong for the map.
        T GetValue(const size_t index) const;

      private:

        const Map<T>& Map_;

      };

      template <typename T>
      ProxyConstMap<T>::ProxyConstMap(const Map<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      ProxyConstMap<T>::ProxyConstMap(const ProxyConstMap& proxyConstMap) noexcept
        :
        Map_{ proxyConstMap.Map_ }
      {
      }

      template <typename T>
      size_t ProxyConstMap<T>::GetValueCount() const noexcept
      {
        return Map_.GetValueCount();
      }

      template <typename T>
      T ProxyConstMap<T>::GetValue(const size_t index) const
      {
        return Map_.GetValue(index);
      }
    }
  }
}