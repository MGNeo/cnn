#pragma once

#include "Map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // RefMap is a wrapper which implements semantics of a safe reference on Map.
      // The safe reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class RefMap
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        RefMap(Map<T>& map) noexcept;

        RefMap(const RefMap<T>& refMap) noexcept;

        RefMap(RefMap<T>&& refMap) noexcept = delete;

        RefMap& operator=(const RefMap& refMap) noexcept = delete;

        RefMap& operator=(RefMap&& refMap) noexcept = delete;

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
        // Topologies of this and map must be equal.
        void FillFrom(const Map<T>& map) const;

        // Exception guarantee: strong for the map.
        // Topologies of this and map must be equal.
        void FillFrom(const RefMap& refMap) const;

      private:

        Map<T>& Map_;

      };

      template <typename T>
      RefMap<T>::RefMap(Map<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      RefMap<T>::RefMap(const RefMap<T>& refMap) noexcept
        :
        Map_{ refMap.Map_ }
      {
      }

      template <typename T>
      size_t RefMap<T>::GetValueCount() const noexcept
      {
        return Map_.GetValueCount();
      }

      template <typename T>
      T RefMap<T>::GetValue(const size_t index) const
      {
        return Map_.GetValue(index);
      }

      template <typename T>
      void RefMap<T>::SetValue(const size_t index, const T value) const
      {
        Map_.SetValue(index, value);
      }

      template <typename T>
      void RefMap<T>::Clear() const noexcept
      {
        Map_.Clear();
      }

      template <typename T>
      void RefMap<T>::Save(std::ostream& ostream) const
      {
        Map_.Save(ostream);
      }

      template <typename T>
      void RefMap<T>::FillFrom(const Map<T>& map) const
      {
        Map_.FillFrom(map);
      }

      template <typename T>
      void RefMap<T>::FillFrom(const RefMap<T>& refMap) const
      {
        Map_.FillFrom(refMap.Map_);
      }
    }
  }
}