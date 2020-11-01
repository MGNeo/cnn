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

        Cursor();

        size_t GetX() const;
        size_t GetY() const;

        void SetX(const size_t x);
        void SetY(const size_t y);

      private:

        size_t X;
        size_t Y;

      };
    }
  }
}
