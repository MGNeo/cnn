#include "Size2D.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      
      Size2D::Size2D(const size_t width,
                     const size_t height) noexcept
        :
        Width{ width },
        Height{ height }
      {
      }

      Size2D::Size2D(Size2D&& size) noexcept
        :
        Width{ size.Width },
        Height{ size.Height }
      {
        size.Reset() ;
      }

      Size2D& Size2D::operator=(Size2D&& size) noexcept
      {
        if (this != &size)
        {
          Width = size.Width;
          Height = size.Height;

          size.Reset() ;
        }
        return *this;
      }

      bool Size2D::operator==(const Size2D& size) const noexcept
      {
        if ((Width == size.Width) && (Height == size.Height))
        {
          return true;
        } else {
          return false;
        }
      }

      bool Size2D::operator!=(const Size2D& size) const noexcept
      {
        if (*this == size)
        {
          return false;
        } else {
          return true;
        }
      }

      size_t Size2D::GetWidth() const noexcept
      {
        return Width;
      }

      void Size2D::SetWidth(const size_t width) noexcept
      {
        Width = width;
      }

      size_t Size2D::GetHeight() const noexcept
      {
        return Height;
      }

      void Size2D::SetHeight(const size_t height) noexcept
      {
        Height = height;
      }

      void Size2D::Reset() noexcept
      {
        Width = 0;
        Height = 0;
      }

      void Size2D::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Size2D::Save(), ostream.good() == false.");
        }
        ostream.write(reinterpret_cast<const char*const>(&Width), sizeof(Width));
        ostream.write(reinterpret_cast<const char* const>(&Height), sizeof(Height));
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Size2D::Save(), ostream.good() == false.");
        }
      }

      void Size2D::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Size2D::Load(), istream.good() == false.");
        }

        decltype(Width) width{};
        decltype(Height) height{};

        istream.read(reinterpret_cast<char*const>(&width), sizeof(width));
        istream.read(reinterpret_cast<char* const>(&height), sizeof(height));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Size2D::Load(), istream.good() == false.");
        }

        Width = width;
        Height = height;
      }

      size_t Size2D::GetArea() const
      {
        if ((Width == 0) || (Height == 0))
        {
          return 0;
        }

        const size_t area = Width * Height;
        if ((area / Width) != Height)
        {
          throw std::overflow_error("cnn::engine::convolution::Size2D::GetArea(), area has been overflowed.");
        }

        return area;
      }
    }
  }
}