#pragma once

#include <cstddef>
#include <cstdint>

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      class Cursor
      {
      public:

        Cursor(const size_t width,
               const size_t height);

        size_t GetWidth() const;
        size_t GetHeight() const;

        size_t GetX() const;
        size_t GetY() const;

        void SetX(const size_t x);
        void SetY(const size_t y);

      private:

        size_t Width;
        size_t Height;

        size_t X;
        size_t Y;

      };
    }
  }
}
