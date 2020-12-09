#pragma once

#include "Map2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // Map2DProtectingReference is a type which implements semantics of protecting reference to Map2D.
      // The protecting reference proxies all methods of Map2D and doesn't allow to use methods, which change
      // the topology of the target map.
      // It allow to protect consistency of complex objects, which contain the target map as its part.
      template <typename T>
      class Map2DProtectingReference
      {
        
        static_assert(std::is_floating_point<T>::value);

      public:

        Map2DProtectingReference(Map2D<T>& map) noexcept;

        Map2DProtectingReference(const Map2DProtectingReference& mapReference) noexcept;

        Map2DProtectingReference(Map2DProtectingReference&& mapReference) noexcept = delete;

        Map2DProtectingReference& operator=(const Map2DProtectingReference& mapReference) = delete;

        Map2DProtectingReference& operator=(Map2DProtectingReference&& mapReference) noexcept = delete;

        const Size2D& GetSize() const noexcept;

        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the map.
        void SetValue(const size_t x, const size_t y, const T value) const;

        // It clears the state without changing of the topology of the map.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for the map.
        // Topologies of the Map and map must be equal.
        void FillFrom(const Map2D<T>& map) const;

      private:

        Map2D<T>& Map;

      };

      template <typename T>
      Map2DProtectingReference<T>::Map2DProtectingReference(Map2D<T>& map) noexcept
        :
        Map{ map }
      {
      }

      template <typename T>
      Map2DProtectingReference<T>::Map2DProtectingReference(const Map2DProtectingReference& mapReference) noexcept
        :
        Map{ mapReference.Map }
      {
      }

      template <typename T>
      const Size2D& Map2DProtectingReference<T>::GetSize() const noexcept
      {
        return Map.GetSize();
      }

      template <typename T>
      T Map2DProtectingReference<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map.GetValue(x, y);
      }

      template <typename T>
      void Map2DProtectingReference<T>::SetValue(const size_t x, const size_t y, const T value) const
      {
        Map.SetValue(x, y, value);
      }

      template <typename T>
      void Map2DProtectingReference<T>::Clear() const noexcept
      {
        Map.Clear();
      }

      template <typename T>
      void Map2DProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Map.Save(ostream);
      }

      template <typename T>
      void Map2DProtectingReference<T>::FillFrom(const Map2D<T>& map) const
      {
        Map.FillFrom(map);
      }
    }
  }
}