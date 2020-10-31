#include "field.hpp"

#include <stdexcept>

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      Field::Field(const size_t width,
                   const size_t height)
        :
        Width{ width },
        Height{ height }
      {
        if (Width == 0)
        {
          throw std::invalid_argument("cnn::examples::complex_using::Field::Field(), Width == 0.");
        }
        if (Height == 0)
        {
          throw std::invalid_argument("cnn::examples::complex_using::Field::Field(), Height == 0.");
        }
        const size_t size = Width * Height;
        if (size / Width != Height)
        {
          throw std::overflow_error("cnn::examples::complex_using::Field::Field(), size has been overflowed.");
        }
        Values.resize(size);
      }

      size_t Field::GetWidth() const
      {
        return Width;
      }
      
      size_t Field::GetHeight() const
      {
        return Height;
      }

      float Field::GetValue(const size_t x,
                            const size_t y) const
      {
        return Values.at(ToIndex(x, y));
      }

      void Field::SetValue(const size_t x,
                           const size_t y,
                           const float value)
      {
        Values.at(ToIndex(x, y)) = value;
      }

      size_t Field::ToIndex(const size_t x,
                            const size_t y) const
      {
        const size_t m = y * Width;
        if ((y != 0) && (Width != 0) && ((m / y) != Width))
        {
          throw std::overflow_error("cnn::examples::complex_using::Field::ToIndex(), m has been overflowed.");
        }
        const size_t index = x + m;
        if (index < x)
        {
          throw std::overflow_error("cnn::examples::complex_using::Field::ToIndex(), index has been overflowed.");
        }
        return index;
      }
    }
  }
}