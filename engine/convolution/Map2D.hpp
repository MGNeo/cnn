#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "../common/Map.hpp"
#include "Size2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Map2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Map2D(const Size2D<size_t> size = {});

        Map2D(const Map2D& map) = default;

        Map2D(Map2D&& map) noexcept = default;

        // Exception guarantee: strong for this.
        Map2D& operator=(const Map2D& map);

        Map2D& operator=(Map2D&& map) noexcept = default;

        Size2D<size_t> GetSize() const noexcept;

        // Exception guarantee: strong for this.
        void SetSize(const Size2D<size_t> size);

        T GetValue(const size_t x, const size_t y) const;

        // Exception guarantee: strong for this.
        void SetValue(const size_t x, const size_t y, const T value);

        // It clears the state without changing of the topology.
        void Clear() noexcept;

        // It resets the state to zero including the topology.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        Size2D<size_t> Size;

        common::Map<T> Map;

        size_t ToIndex(const size_t x, const size_t y) const;

      };

      template <typename T>
      Map2D<T>::Map2D(const Size2D<size_t> size)
        :
        Size{ size },
        Mep{ Size.GetArea() }
      {
        Map.Clear();
      }

      template <typename T>
      Map2D<T>& Map2D<T>::operator=(const Map2D& map)
      {
        if (this != &map)
        {
          Map2D<T> tmpMap{ map };
          // Beware, it is very intimate place for exception guarantee.
          std::swap(*this, tmpMap);
        }
        return *this;
      }

      template <typename T>
      Size2D<size_t> Map2D<T>::GetSize() const noexcept
      {
        return Size;
      }

      template <typename T>
      void Map2D<T>::SetSize(const Size2D<size_t> size)
      {
        Map2D<T> tmpMap{ size };
        // Beware, it is very intimate place for exception guarantee.
        std::swap(*this, tmpMap);
      }

      template <typename T>
      T Map2D<T>::GetValue(const size_t x, const size_t y) const
      {
        const size_t index = ToIndex(x, y);
        return Map.GetValue(index);
      }

      template <typename T>
      void Map2D<T>::SetValue(const size_t x, const size_t y, const T value)
      {
        const size_t index = ToIndex(x, y);
        Map.SetValue(index, value);
      }

      template <typename T>
      void Map2D<T>::Clear() noexcept
      {
        Map.Clear();
      }

      template <typename T>
      void Map2D<T>::Reset() noexcept
      {
        Map.Reset();
      }

      template <typename T>
      void Map2D<T>::Save(std::ostream& ostream) const
      {
        // TODO:
        // ...
      }

      template <typename T>
      void Map2D<T>::Load(std::istream& istream)
      {
        // TODO:
        // ...
      }

      template <typename T>
      size_t Map2D<T>::ToIndex(const size_t x, const size_t y) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (x >= Size.GetWidth())
        {
          throw std::range_error("cnn::engine::convolution::Map2D::ToIndex(), x >= Size.GetWidth().");
        }
        if (y >= Size.GetHeight())
        {
          throw std::range_error("cnn::engine::convolution::Map2D::ToIndex(), y >= Size.GetHeight().");
        }
#endif
        return x + y * Size.GetWidth();
      }
    }
  }
}
