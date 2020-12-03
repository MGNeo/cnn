#pragma once

#include "Map2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // RefMap2D is a wrapper which implements semantics of a safe reference on Map2D.
      // The safe reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class RefMap2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        RefMap2D(Map2D<T>& map) noexcept;

        RefMap2D(const RefMap2D& refMap) noexcept;

        RefMap2D(RefMap2D&& refMap) noexcept = delete;

        RefMap2D& operator=(const RefMap2D<T>& refMap) noexcept = delete;

        RefMap2D<T>& operator=(RefMap2D<T>&& refMap) noexcept = delete;

        Size2D GetSize() const noexcept;

        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the map.
        void SetValue(const size_t x, const size_t y, const T value);

        // It clears the state without changing of the topology of the map.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this.
        // Topologies of this and map must be equal.
        void FillFrom(const Map2D<T>& map) const;

        // Exception guarantee: strong for this.
        // Topologies of this and refMap must be equal.
        void FillFrom(RefMap2D& refMap) const;

      private:

        Map2D<T>& Map_;

      };

      template <typename T>
      RefMap2D<T>::RefMap2D(Map2D<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      RefMap2D<T>::RefMap2D(const RefMap2D& refMap) noexcept
        :
        Map_{ refMap }
      {
      }

      template <typename T>
      Size2D RefMap2D<T>::GetSize() const noexcept
      {
        return Map_.GetSize();
      }

      template <typename T>
      T RefMap2D<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map_.GetValue(x, y);
      }

      template <typename T>
      void RefMap2D<T>::SetValue(const size_t x, const size_t y, const T value)
      {
        Map_.SetValue(x, y, value);
      }

      template <typename T>
      void RefMap2D<T>::Clear() const noexcept
      {
        Map_.Clear();
      }

      template <typename T>
      void RefMap2D<T>::Save(std::ostream& ostream) const
      {
        Map_.Save(ostream);
      }

      template <typename T>
      void RefMap2D<T>::FillFrom(const Map2D<T>& map) const
      {
        Map_.FillFrom(map);
      }

      template <typename T>
      void RefMap2D<T>::FillFrom(RefMap2D& refMap) const
      {
        Map_.FillFrom(refMap.Map_);
      }
    }
  }
}