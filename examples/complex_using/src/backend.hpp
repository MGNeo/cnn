#pragma once

#include "field.hpp"
#include "cursor.hpp"

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      class Backend
      {
      public:

        Backend();

      private:

        void LoadField();
        void LoadNetwork();

        Field Field_;
        Cursor Cursor_;
        // Network
        // ...

      };
    }
  }
}