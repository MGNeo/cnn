#pragma once

#include "Map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // ConstRefMap is a wrapper which implements semantics of a safe const reference on Map.
      // The safe const reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class ConstRefMap
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ConstRefMap(Map<T>& map) noexcept;

        ConstRefMap(const ConstRefMap<T>& constRefMap) noexcept;

        ConstRefMap(ConstRefMap<T>&& constRefMap) noexcept = delete;

        ConstRefMap& operator=(const ConstRefMap& constRefMap) noexcept = delete;

        ConstRefMap& operator=(ConstRefMap&& constRefMap) noexcept = delete;

        size_t GetValueCount() const noexcept;

        // Exception guarantee: strong for the map.
        T GetValue(const size_t index) const;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

      private:

        const Map<T>& Map_;

      };

      template <typename T>
      ConstRefMap<T>::ConstRefMap(Map<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      ConstRefMap<T>::ConstRefMap(const ConstRefMap<T>& constRefMap) noexcept
        :
        Map_{ constRefMap.Map_ }
      {
      }

      template <typename T>
      size_t ConstRefMap<T>::GetValueCount() const noexcept
      {
        return Map_.GetValueCount();
      }

      template <typename T>
      T ConstRefMap<T>::GetValue(const size_t index) const
      {
        return Map_.GetValue(index);
      }

      template <typename T>
      void ConstRefMap<T>::Save(std::ostream& ostream) const
      {
        Map_.Save(ostream);
      }
    }
  }
}