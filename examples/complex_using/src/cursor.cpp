#include "cursor.hpp"

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      Cursor::Cursor()
      {
        SetX(0);
        SetY(0);
      }

      size_t Cursor::GetX() const
      {
        return X;
      }

      size_t Cursor::GetY() const
      {
        return Y;
      }

      void Cursor::SetX(const size_t x)
      {
        X = x;
      }

      void Cursor::SetY(const size_t y)
      {
        Y = y;
      }
    }
  }
}