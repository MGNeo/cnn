#pragma once

#include "Map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // MapProtectingReference is a type which implements semantics of protecting reference to Map.
      // The protecting reference proxies all methods of Map and doesn't allow to use methods, which change
      // the topology of the target map.
      // It allow to protect consistency of complex objects, which contain the target map as its part.
      template <typename T>
      class MapProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        MapProtectingReference(Map<T>& map) noexcept;

        MapProtectingReference(const MapProtectingReference& mapReference) noexcept;

        MapProtectingReference(MapProtectingReference&& mapReference) noexcept = delete;

        MapProtectingReference& operator=(const MapProtectingReference& mapReference) noexcept = delete;

        MapProtectingReference& operator=(MapProtectingReference&& mapReference) noexcept = delete;

        size_t GetValueCount() const noexcept;

        // Exception guarantee: strong for the map.
        T GetValue(const size_t index) const;

        // Exception guarantee: strong for the map.
        void SetValue(const size_t index, const T value) const;

        // It clears the state without changing of the topology of the map.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for the map.
        // Topologies of the Map and map must be equal.
        void FillFrom(const Map<T>& map) const;

      private:

        Map<T>& Map_;

      };

      template <typename T>
      MapProtectingReference<T>::MapProtectingReference(Map<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      MapProtectingReference<T>::MapProtectingReference(const MapProtectingReference& mapReference) noexcept
        :
        Map_{ mapReference.Map_ }
      {
      }

      template <typename T>
      size_t MapProtectingReference<T>::GetValueCount() const noexcept
      {
        return Map_.GetValueCount();
      }

      template <typename T>
      T MapProtectingReference<T>::GetValue(const size_t index) const
      {
        return Map_.GetValue(index);
      }

      template <typename T>
      void MapProtectingReference<T>::SetValue(const size_t index, const T value) const
      {
        Map_.SetValue(index, value);
      }

      template <typename T>
      void MapProtectingReference<T>::Clear() const noexcept
      {
        Map_.Clear();
      }

      template <typename T>
      void MapProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Map_.Save(ostream);
      }

      template <typename T>
      void MapProtectingReference<T>::FillFrom(const Map<T>& map) const
      {
        Map_.FillFrom(map);
      }
    }
  }
}