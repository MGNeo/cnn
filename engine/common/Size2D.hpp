#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Size2D
      {
        
        static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value);

      public:
        
        Size2D() noexcept;

        Size2D(const Size2D& size) = default;

        Size2D(Size2D&& size) noexcept;

        Size2D& operator=(const Size2D& size) = default;

        Size2D& operator=(Size2D&& size) noexcept;

        T GetWidth() const noexcept;

        void SetWidth(const T width) noexcept;

        T GetHeight() const noexcept;

        void SetHeight(const T height) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

        T GetArea() const;

      private:

        T Width;
        T Height;

      };

      template <typename T>
      Size2D<T>::Size2D() noexcept
      {
        Clear();
      }

      template <typename T>
      Size2D<T>::Size2D(Size2D<T>&& size) noexcept
        :
        Width{ size.Width },
        Height{ size.Height }
      {
        size.Width = 0;
        size.Height = 0;
      }

      template <typename T>
      Size2D<T>& Size2D<T>::operator=(Size2D<T>&& size) noexcept
      {
        if (this != &size)
        {
          Width = size.Width;
          Height = size.Height;

          size.Width = 0;
          size.Height = 0;
        }
        return *this;
      }

      template <typename T>
      T Size2D<T>::GetWidth() const noexcept
      {
        return Width;
      }

      template <typename T>
      void Size2D<T>::SetWidth(const T width) noexcept
      {
        Width = width;
      }

      template <typename T>
      T Size2D<T>::GetHeight() const noexcept
      {
        return Height;
      }

      template <typename T>
      void Size2D<T>::SetHeight(const T height) noexcept
      {
        Height = height;
      }

      template <typename T>
      void Size2D<T>::Clear() noexcept
      {
        Width = 0;
        Height = 0;
      }

      template <typename T>
      void Size2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::common::Size2D::Save(), ostream.good() == false.");
        }
        ostream.write(reinterpret_cast<const char*const>(&Width), sizeof(Width));
        ostream.write(reinterpret_cast<const char* const>(&Height), sizeof(Height));
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::common::Size2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Size2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::common::Size2D::Load(), istream.good() == false.");
        }

        decltype(Width) width{};
        decltype(Height) height{};

        istream.read(reinterpret_cast<char*const>(&width), sizeof(width));
        istream.read(reinterpret_cast<char* const>(&height), sizeof(height));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::common::Size2D::Load(), istream.good() == false.");
        }

        SetWidth(width);
        SetHeight(height);
      }

      template <typename T>
      T Size2D<T>::GetArea() const
      {
        if ((Width == 0) || (Height == 0))
        {
          return 0;
        }

        const T area = Width * Height;
        if ((area / Width) != Height)
        {
          throw std::overflow_error("cnn::engine::common::Size2D::GetArea(), area has been overflowed.");
        }

        return area;
      }

    }
  }
}

