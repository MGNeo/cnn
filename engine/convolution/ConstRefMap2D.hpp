#pragma once

#include "Map2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // ConstRefMap2D is a wrapper which implements semantics of a safe const reference on Map2D.
      // The safe const reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class ConstRefMap2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ConstRefMap2D(const Map2D<T>& map) noexcept;

        ConstRefMap2D(const ConstRefMap2D& constRefMap) noexcept;

        ConstRefMap2D(ConstRefMap2D&& constRefMap) noexcept = delete;

        ConstRefMap2D& operator=(const ConstRefMap2D<T>& constRefMap) noexcept = delete;

        ConstRefMap2D<T>& operator=(ConstRefMap2D<T>&& constRefMap) noexcept = delete;

        Size2D GetSize() const noexcept;

        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

      private:

        const Map2D<T>& Map_;

      };

      template <typename T>
      ConstRefMap2D<T>::ConstRefMap2D(const Map2D<T>& map) noexcept
        :
        Map_{ map }
      {
      }

      template <typename T>
      ConstRefMap2D<T>::ConstRefMap2D(const ConstRefMap2D& constRefMap) noexcept
        :
        Map_{ constRefMap }
      {
      }

      template <typename T>
      Size2D ConstRefMap2D<T>::GetSize() const noexcept
      {
        return Map_.GetSize();
      }

      template <typename T>
      T ConstRefMap2D<T>::GetValue(const size_t x, const size_t y) const
      {
        return Map_.GetValue(x, y);
      }

      template <typename T>
      void ConstRefMap2D<T>::Save(std::ostream& ostream) const
      {
        Map_.Save(ostream);
      }
    }
  }
}