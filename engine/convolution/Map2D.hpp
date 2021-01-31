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

        Map2D(const Size2D size = {});

        Map2D(const Map2D& map) = default;

        Map2D(Map2D&& map) noexcept = default;

        // Exception guarantee: strong for this.
        Map2D& operator=(const Map2D& map);

        Map2D& operator=(Map2D&& map) noexcept = default;

        const Size2D& GetSize() const noexcept;

        // Exception guarantee: strong for this.
        void SetSize(const Size2D& size);

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

        // Exception guarantee: strong for this.
        // Topologies of this and map must be equal.
        void FillFrom(const Map2D& map);

      private:

        Size2D Size;

        common::Map<T> Map;

        size_t ToIndex(const size_t x, const size_t y) const;

      };

      template <typename T>
      Map2D<T>::Map2D(const Size2D size)
        :
        Size{ size },
        Map{ Size.GetArea() }
      {
        Clear();
      }

      template <typename T>
      Map2D<T>& Map2D<T>::operator=(const Map2D& map)
      {
        if (this != &map)
        {
          Map2D<T> tmpMap{ map };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpMap);
        }
        return *this;
      }

      template <typename T>
      const Size2D& Map2D<T>::GetSize() const noexcept
      {
        return Size;
      }

      template <typename T>
      void Map2D<T>::SetSize(const Size2D& size)
      {
        Map2D<T> tmpMap{ size };
        // Beware, it is very intimate place for strong exception guarantee.
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
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Map2D::Save(), ostream.good() == false.");
        }
        Size.Save(ostream);
        Map.Save(ostream);
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Map2D::save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Map2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Map2D::Load(), istream.good() == false.");
        }
        decltype(Size) size;
        decltype(Map) map;

        size.Load(istream);
        map.Load(istream);

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Map2D::Load(), istream.good() == false.");
        }

        Size = std::move(size);
        Map = std::move(map);
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

      template <typename T>
      void Map2D<T>::FillFrom(const Map2D& map)
      {
        if (this != &map)
        {
          if (Size != map.Size)
          {
            throw std::invalid_argument("cnn::engine::convolution::Map2D::FillFrom(), Size != map.Size.");
          }
          Map.FillFrom(map.Map);
        }
      }
    }
  }
}
