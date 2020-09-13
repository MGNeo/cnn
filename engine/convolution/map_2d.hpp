#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "i_map_2d.hpp"
#include "../common/map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Map2D : public IMap2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Map2D(const size_t width, const size_t height);

        size_t GetWidth() const override;
        size_t GetHeight() const override;

        T GetValue(const size_t x, const size_t y) const override;
        void SetValue(const size_t x, const size_t y, const T value) override;

        void Clear() override;

        typename IMap2D<T>::Uptr Clone(const bool cloneState) const override;

        Map2D(const Map2D<T>& map, const bool cloneState);

      private:

        size_t Width;
        size_t Height;
        typename common::IMap<T>::Uptr Map_;

        size_t ToIndex(const size_t x, const size_t y) const;

      };

      template <typename T>
      Map2D<T>::Map2D(const size_t width, const size_t height)
        :
        Width{ width },
        Height{ height }
      {
        if (Width == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Map2D::Map2D(), Width == 0.");
        }
        if (Height == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Map2D::Map2D(), Height == 0.");
        }
        const size_t valueCount = Width * Height;
        if ((valueCount / Width) != Height)
        {
          throw std::overflow_error("cnn::engine::convolution::Map2D::Map2D(), valueCount was overflowed.");
        }
        Map_ = std::make_unique<common::Map<T>>(valueCount);
      }

      template <typename T>
      size_t Map2D<T>::GetWidth() const
      {
        return Width;
      }

      template <typename T>
      size_t Map2D<T>::GetHeight() const
      {
        return Height;
      }

      template <typename T>
      T Map2D<T>::GetValue(const size_t x, const size_t y) const
      {
        const size_t index = ToIndex(x, y);
        return Map_->GetValue(index);
      }

      template <typename T>
      void Map2D<T>::SetValue(const size_t x, const size_t y, const T value)
      {
        const size_t index = ToIndex(x, y);
        Map_->SetValue(index, value);
      }

      template <typename T>
      size_t Map2D<T>::ToIndex(const size_t x, const size_t y) const
      {
        if (x >= Width)
        {
          throw std::range_error("cnn::engine::convolution::Map2D::ToIndex(), x >= Width.");
        }
        if (y >= Height)
        {
          throw std::range_error("cnn::engine::convolution::Map2D::ToIndex(), y >= Height.");
        }
        return x + y * Width;
      }

      template <typename T>
      void Map2D<T>::Clear()
      {
        Map_->Clear();
      }

      template <typename T>
      typename IMap2D<T>::Uptr Map2D<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Map2D<T>>(*this, cloneState);
      }

      template <typename T>
      Map2D<T>::Map2D(const Map2D<T>& map, const bool cloneState)
        :
        Width{ map.GetWidth() },
        Height{ map.GetHeight() },
        Map_{ map.Map_->Clone(cloneState) }
      {
      }
    }
  }
}